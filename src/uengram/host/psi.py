"""Kernel PSI triggers: stall events within seconds, between windows.

A trigger `some S W` written to /proc/pressure/<resource> makes poll() report
POLLPRI when the `some` stall time grows by at least S us within a W us
window (Documentation/accounting/psi.rst). While the resource is stalled the
kernel samples the growth 10 times per window and notifies at most once per
window; unprivileged users may only use windows that are multiples of 2 s,
so W = 2 s. The daemon runs as an unprivileged user.

The thresholds are the 2 s analogue of the window saturation flag (node.py):
the largest 2 s stall share of the declared-healthy commissioning run,
floored at node.SAT_FLOOR. A 2 s share varies more than a 32 s share, so the
32 s bound is not reused. Shares are measured over every pair of consecutive
1 s ticks; the kernel's window slides, so the bound is the maximum over a
subset of its windows (slightly low: the trigger fires a little more often
than "never during commissioning"). A resource whose 2 s share reached 100 %
during commissioning cannot exceed it and gets no trigger.

This is not another attribute: it reads the same `some` totals that
cpu_pressure, mem_pressure and io_pressure own, at a shorter time scale, and
only for alerting; nothing of it enters the latent.
"""
from __future__ import annotations

import os
import select

WINDOW_S = 2
FILES = {"cpu_pressure": "cpu", "mem_pressure": "memory", "io_pressure": "io"}
COLUMNS = {"cpu_pressure": "psi_cpu_us", "mem_pressure": "psi_mem_us", "io_pressure": "psi_io_us"}
MAX_SHARE = 0.999  # at or above: no trigger (the bound cannot be exceeded)


def fast_bounds(cols, floor: float) -> dict[str, float]:
    """Largest stall share over two consecutive ticks, per resource (numpy columns)."""
    import numpy as np
    t, dt = cols["t"], cols["dt_s"]
    pair = np.diff(t) < 1.5  # consecutive ticks; excluded intervals and gaps break pairs
    span = dt[:-1] + dt[1:]
    out = {}
    for name, col in COLUMNS.items():
        stall = (cols[col][:-1] + cols[col][1:]) * 1e-6
        ok = pair & np.isfinite(stall) & (span > 0)
        share = stall[ok] / span[ok]
        out[name] = float(max(np.max(share), floor)) if len(share) else floor
    return out


class PsiTriggers:
    """One trigger fd per resource with a bound below MAX_SHARE."""

    def __init__(self, bounds: dict[str, float], window_s: int = WINDOW_S):
        self.window_s = window_s
        self.fds: dict[int, str] = {}
        self.thresholds: dict[str, int] = {}
        for name, share in bounds.items():
            if share >= MAX_SHARE:
                continue
            fd = os.open(f"/proc/pressure/{FILES[name]}", os.O_RDWR | os.O_NONBLOCK | os.O_CLOEXEC)
            us = max(1, int(share * window_s * 1e6))
            try:
                os.write(fd, f"some {us} {window_s * 1_000_000}\0".encode())
            except OSError:
                os.close(fd)
                self.close()
                raise
            self.fds[fd], self.thresholds[name] = name, us

    def register(self, poller: select.poll) -> None:
        for fd in self.fds:
            poller.register(fd, select.POLLPRI)

    def resource(self, fd: int) -> str | None:
        return self.fds.get(fd)

    def close(self) -> None:
        for fd in self.fds:
            os.close(fd)
        self.fds = {}
