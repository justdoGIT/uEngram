import os
import select
import socket
import sys
import time

import pytest

from uengram.host import hub as hubmod
from uengram.host.hub import Hub
from uengram.host.stream import StateStream

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


class Daemon:
    """A StateStream served the way the daemon's idle loop serves it."""

    def __init__(self, path):
        self.poller = select.poll()
        self.stream = StateStream(path)
        self.stream.register(self.poller)

    def serve(self, seconds=0.02):
        for fd, ev in self.poller.poll(seconds * 1000):
            self.stream.handle(fd, ev)


def _follow_cmd(path):
    return ["env", f"PYTHONPATH={SRC}", sys.executable, "-m", "uengram.host", "follow", "--socket", path]


def _until(cond, step, timeout=15.0):
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, "timed out"
        step()


@pytest.fixture
def short_backoff(monkeypatch):
    monkeypatch.setattr(hubmod, "BACKOFF_MIN", 0.2)
    monkeypatch.setattr(hubmod, "WRITE_EVERY", 0.0)


def test_hub_receives_pushed_state_ticks_and_stalls_and_survives_a_restart(tmp_path, short_backoff):
    sock, out = str(tmp_path / "s.sock"), str(tmp_path / "cluster.json")
    d = Daemon(sock)
    d.stream.publish({"type": "state", "t0": 1.0, "kind": "semantic"})  # before anyone connects
    hub = Hub({"n1": _follow_cmd(sock)}, out)

    def step():
        d.serve()
        hub.step(0.05)

    try:
        _until(lambda: hub.nodes["n1"]["state"] is not None, step)  # the latest state arrives on connect
        d.stream.publish({"type": "tick", "t": 2.0, "dt_s": 1.0, "cpu_busy_j": 10, "cpu_total_j": 400})
        d.stream.publish({"type": "pressure", "t": 3.0, "resource": "mem_pressure"})
        _until(lambda: hub.nodes["n1"]["pressure"].get("mem_pressure") == 3.0, step)
        assert hub.nodes["n1"]["ticks"][-1]["cpu_busy_j"] == 10
        assert os.path.exists(out)

        d.stream.close()  # daemon restart
        _until(lambda: not hub.nodes["n1"]["link"]["up"], step)
        d = Daemon(sock)
        d.stream.publish({"type": "state", "t0": 33.0, "kind": "episodic"})
        _until(lambda: hub.nodes["n1"]["link"]["up"] and hub.nodes["n1"]["state"]["t0"] == 33.0, step)
    finally:
        hub.close()
        d.stream.close()


def test_a_reader_that_stops_reading_is_dropped_without_blocking_the_daemon(tmp_path):
    d = Daemon(str(tmp_path / "s.sock"))
    c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    c.connect(str(tmp_path / "s.sock"))
    try:
        _until(lambda: len(d.stream.clients) == 1, d.serve)
        big = {"type": "tick", "pad": "x" * 65536}
        t0 = time.monotonic()
        for _ in range(64):  # 4 MB: far beyond the socket buffer of a reader that never reads
            d.stream.publish(big)
        assert time.monotonic() - t0 < 1.0
        assert d.stream.clients == {}
    finally:
        c.close()
        d.stream.close()
