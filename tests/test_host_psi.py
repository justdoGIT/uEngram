import os
import select
import subprocess
import sys
import time

import numpy as np
import pytest

from uengram.host.psi import PsiTriggers, fast_bounds


def _cols(t, psi_cpu):
    n = len(t)
    return {"t": np.asarray(t, float), "dt_s": np.ones(n), "psi_cpu_us": np.asarray(psi_cpu, float),
            "psi_mem_us": np.zeros(n), "psi_io_us": np.full(n, np.nan)}


def test_bound_is_the_largest_two_second_stall_share():
    b = fast_bounds(_cols([0, 1, 2, 3, 4], [0, 0, 9e5, 9e5, 0]), floor=0.01)
    assert b["cpu_pressure"] == pytest.approx(0.9)
    assert b["mem_pressure"] == 0.01  # never stalled: the floor
    assert b["io_pressure"] == 0.01  # no data: the floor


def test_ticks_across_a_gap_are_not_paired():
    # the two stalled seconds are 100 s apart: no 2 s window holds both
    b = fast_bounds(_cols([0, 1, 2, 102, 103], [0, 0, 9e5, 9e5, 0]), floor=0.01)
    assert b["cpu_pressure"] == pytest.approx(0.45)


def _writable(res: str) -> bool:
    return os.access(f"/proc/pressure/{res}", os.W_OK)


@pytest.mark.skipif(not _writable("cpu"), reason="no writable /proc/pressure/cpu (PSI triggers)")
def test_a_bound_that_cannot_be_exceeded_gets_no_trigger():
    trig = PsiTriggers({"cpu_pressure": 1.0, "mem_pressure": 0.5})
    try:
        assert set(trig.thresholds) == {"mem_pressure"}
        assert trig.thresholds["mem_pressure"] == 1_000_000  # 0.5 of a 2 s window, in us
    finally:
        trig.close()


@pytest.mark.skipif(not _writable("cpu"), reason="no writable /proc/pressure/cpu (PSI triggers)")
def test_cpu_contention_fires_the_trigger_within_its_window():
    trig = PsiTriggers({"cpu_pressure": 0.001})
    poller = select.poll()
    trig.register(poller)
    spin = [subprocess.Popen([sys.executable, "-c", "import time\nt=time.time()+4\nwhile time.time()<t: pass"])
             for _ in range(2 * os.cpu_count())]
    try:
        t0, fired = time.monotonic(), None
        while fired is None and time.monotonic() - t0 < 6:
            for fd, ev in poller.poll(500):
                if ev & select.POLLPRI:
                    fired = trig.resource(fd)
        assert fired == "cpu_pressure"
    finally:
        for p in spin:
            p.kill()
            p.wait()
        trig.close()
