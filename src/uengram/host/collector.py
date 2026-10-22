"""Tick scheduling and the JSONL tick format; stdlib only.

Ticks fire at wall-clock second + 0.5 s. Windows are aligned to multiples of
K seconds of wall time, so window boundaries sit 0.5 s away from every tick
and chrony offsets (tens of ms across the fleet) never move a tick into a
neighbouring window; windows of different nodes cover the same seconds.
Rates use the CLOCK_MONOTONIC interval measured by `Ticker`, never the
nominal period. A late tick (process stalled) simply has a longer dt.
"""
from __future__ import annotations

import json
import math
import os
import platform
import signal
import socket
import time
from typing import Callable

from . import schema
from .perf import PerfCounters
from .sources import Snapshot, Ticker

PHASE = 0.5


def open_collector(pmu: bool = True) -> Ticker:
    perf = PerfCounters(system=True) if pmu else None
    return Ticker(Snapshot(perf))


def host_meta(ticker: Ticker, period: float) -> dict:
    with open("/proc/cpuinfo") as f:
        model = next((ln.split(":", 1)[1].strip() for ln in f if ln.startswith("model name")), "")
    return {"host": socket.gethostname(), "kernel": platform.release(), "cpu_model": model,
            "period_s": period, "fields": list(schema.TICK_FIELDS), **ticker.snap.meta()}


def _sleep_until_phase(period: float) -> None:
    now = time.time()
    target = math.floor((now - PHASE) / period) * period + PHASE + period
    time.sleep(max(0.0, target - now))


def tick_loop(ticker: Ticker, on_tick: Callable[[float, dict], None], period: float = 1.0,
              seconds: float | None = None) -> None:
    """Call on_tick(wall_time, tick) every period until SIGTERM/SIGINT or `seconds`."""
    stop = False

    def _stop(*_):
        nonlocal stop
        stop = True

    old = {s: signal.signal(s, _stop) for s in (signal.SIGTERM, signal.SIGINT)}
    t_end = None if seconds is None else time.monotonic() + seconds
    try:
        _sleep_until_phase(period)
        ticker.tick()  # discard: its interval started at construction, off-phase
        while not stop and (t_end is None or time.monotonic() < t_end):
            _sleep_until_phase(period)
            if stop:
                break
            tick = ticker.tick()
            on_tick(round(time.time(), 3), tick)
    finally:
        for s, h in old.items():
            signal.signal(s, h)


class JsonlSink:
    """First line {"meta": ...}; then one {"t": wall_time, <tick fields>} per tick."""

    def __init__(self, path: str, meta: dict):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fresh = not os.path.exists(path) or os.path.getsize(path) == 0
        self.f = open(path, "a", buffering=1)
        if fresh:
            self.f.write(json.dumps({"meta": meta}) + "\n")

    def __call__(self, t: float, tick: dict) -> None:
        self.f.write(json.dumps({"t": t, **tick}, separators=(",", ":")) + "\n")

    def close(self) -> None:
        self.f.close()


def read_ticks(path: str) -> tuple[dict, list[dict]]:
    """Load a tick file written by JsonlSink (a truncated last line is skipped)."""
    meta, ticks = {}, []
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "meta" in rec:
                meta = meta or rec["meta"]
            else:
                ticks.append(rec)
    return meta, ticks
