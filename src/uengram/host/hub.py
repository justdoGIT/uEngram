"""Cluster hub: follow every node's state stream and keep one snapshot; stdlib.

One `ssh NODE uengram-host follow` per node (stream.py); every line updates
that node's entry. The snapshot (`/dev/shm/uengram-cluster-<user>.json` by
default: tmpfs, rewritten up to once a second, owner-only) holds per node the
latest window state, the last ticks (cluster.LOAD_TICKS), the time of the
last PSI trigger per resource and the link status. `uengram-host cluster`
reads it; placement is computed at read time (cluster.py), so a node whose
stream stopped shows up as stale instead of keeping its last advice.

A follower that exits (daemon restart, node reboot, network) is restarted
after 5 s, doubling up to 60 s while it keeps failing.
"""
from __future__ import annotations

import collections
import json
import os
import selectors
import subprocess
import sys
import time

from .cluster import LOAD_TICKS
from .stream import atomic_json

BACKOFF_MIN, BACKOFF_MAX = 5.0, 60.0
WRITE_EVERY = 1.0
REMOTE = "PYTHONPATH=$HOME/.local/share/uengram-host/src exec python3 -m uengram.host follow"


def ssh_follow(host: str) -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "-o", "ServerAliveInterval=10",
            "-o", "ServerAliveCountMax=3", host, REMOTE]


class Hub:
    def __init__(self, commands: dict[str, list[str]], out: str):
        self.commands, self.out = commands, out
        self.sel = selectors.DefaultSelector()
        self.nodes = {n: {"state": None, "ticks": collections.deque(maxlen=LOAD_TICKS), "pressure": {},
                          "link": {"up": False, "since": None, "error": None}} for n in commands}
        self.procs: dict[str, subprocess.Popen] = {}
        self.buf: dict[str, bytes] = {}
        self.retry_at = {n: 0.0 for n in commands}
        self.backoff = {n: BACKOFF_MIN for n in commands}
        self.dirty, self.written = True, 0.0

    def _start(self, name: str) -> None:
        p = subprocess.Popen(self.commands[name], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        os.set_blocking(p.stdout.fileno(), False)
        self.procs[name], self.buf[name] = p, b""
        self.sel.register(p.stdout, selectors.EVENT_READ, name)
        self.nodes[name]["link"] = {"up": True, "since": time.time(), "error": None}
        self.dirty = True

    def _ended(self, name: str) -> None:
        p = self.procs.pop(name)
        self.sel.unregister(p.stdout)
        p.stdout.close()
        try:
            tail = p.stderr.read().decode(errors="replace").strip().splitlines()
            err = tail[-1][-200:] if tail else ""
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
        link = self.nodes[name]["link"]
        lasted = time.time() - (link["since"] or time.time())
        delay = BACKOFF_MIN if lasted > BACKOFF_MAX else self.backoff[name]
        self.retry_at[name] = time.monotonic() + delay
        self.backoff[name] = min(2 * delay, BACKOFF_MAX)
        self.nodes[name]["link"] = {"up": False, "since": time.time(), "error": err or f"exit {p.returncode}"}
        print(f"{time.strftime('%H:%M:%S')} {name}: stream ended ({self.nodes[name]['link']['error']}); "
              f"retry in {delay:.0f} s", flush=True)
        self.dirty = True

    def _line(self, name: str, line: bytes) -> None:
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            return
        node, kind = self.nodes[name], msg.pop("type", None)
        if kind == "state":
            node["state"] = msg
        elif kind == "tick":
            node["ticks"].append(msg)
        elif kind == "pressure":
            node["pressure"][msg["resource"]] = msg["t"]
        self.dirty = True

    def _read(self, name: str) -> None:
        p = self.procs[name]
        try:
            chunk = os.read(p.stdout.fileno(), 65536)
        except BlockingIOError:
            return
        if not chunk:
            self._ended(name)
            return
        *lines, self.buf[name] = (self.buf[name] + chunk).split(b"\n")
        for ln in lines:
            self._line(name, ln)

    def snapshot(self) -> dict:
        return {"t": time.time(), "nodes": {n: {**v, "ticks": list(v["ticks"])} for n, v in self.nodes.items()}}

    def step(self, timeout: float = 1.0) -> None:
        now = time.monotonic()
        for name in self.commands:
            if name not in self.procs and now >= self.retry_at[name]:
                self._start(name)
        for key, _ in self.sel.select(timeout):
            self._read(key.data)
        if self.dirty and time.monotonic() - self.written >= WRITE_EVERY:
            atomic_json(self.out, self.snapshot())
            self.dirty, self.written = False, time.monotonic()

    def close(self) -> None:
        for p in self.procs.values():
            p.terminate()
        for p in self.procs.values():
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()


def run(commands: dict[str, list[str]], out: str, seconds: float | None = None) -> int:
    hub = Hub(commands, out)
    print(f"hub: following {', '.join(commands)}; snapshot {out}", file=sys.stderr, flush=True)
    t_end = None if seconds is None else time.monotonic() + seconds
    try:
        while t_end is None or time.monotonic() < t_end:
            hub.step()
    except KeyboardInterrupt:
        pass
    finally:
        hub.close()
    return 0
