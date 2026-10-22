"""Tick scheduling and the JSONL tick format; stdlib only.

Ticks fire at wall-clock second + 0.5 s. Windows are aligned to multiples of
K seconds of wall time, so window boundaries sit 0.5 s away from every tick
and chrony offsets (tens of ms across the fleet) never move a tick into a
neighbouring window; windows of different nodes cover the same seconds.
Rates use the CLOCK_MONOTONIC interval measured by `Ticker`, never the
nominal period. A late tick (process stalled) simply has a longer dt.
"""
from __future__ import annotations

import glob
import gzip
import json
import math
import os
import platform
import queue
import re
import shutil
import signal
import socket
import sys
import threading
import time
from typing import Callable

from . import schema
from .perf import PerfCounters
from .sources import Snapshot, Ticker

PHASE = 0.5


class AsyncWriter:
    """File writes on one background thread, in submission order, so a slow
    disk never delays a tick. A tick that misses its second lands in the next
    one, and its window then has K-1 ticks and is dropped whole. Measured on
    hp, whose root is an overlay on a USB stick: write() blocked for about a
    second every few minutes (97 missed ticks in 4.4 h, the collector using
    2 ms of CPU across each gap), which dropped 23 % of hp's windows.
    At most `maxsize` writes wait (an hour of ticks); beyond that new ones are
    dropped and counted. A failing write is logged and counted, not raised:
    the daemon keeps monitoring while its disk misbehaves."""

    def __init__(self, maxsize: int = 3600):
        self.q: queue.Queue = queue.Queue(maxsize)
        self.dropped = self.failed = 0
        self.thread = threading.Thread(target=self._run, name="writer", daemon=True)
        self.thread.start()

    def submit(self, fn: Callable, *args) -> None:
        try:
            self.q.put_nowait((fn, args))
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        while (item := self.q.get()) is not None:
            fn, args = item
            try:
                fn(*args)
            except OSError as e:
                self.failed += 1
                print(f"writer: {getattr(fn, '__qualname__', fn)}: {e}", file=sys.stderr, flush=True)

    def close(self, timeout: float = 60.0) -> None:
        """Finish the queued writes (up to `timeout` seconds)."""
        self.q.put(None, timeout=timeout)
        self.thread.join(timeout)


def open_collector(pmu: bool = True) -> Ticker:
    perf = PerfCounters(system=True) if pmu else None
    return Ticker(Snapshot(perf))


def host_meta(ticker: Ticker, period: float) -> dict:
    with open("/proc/cpuinfo") as f:
        model = next((ln.split(":", 1)[1].strip() for ln in f if ln.startswith("model name")), "")
    with open("/proc/meminfo") as f:
        mem_kb = next(int(ln.split()[1]) for ln in f if ln.startswith("MemTotal:"))
    return {"host": socket.gethostname(), "kernel": platform.release(), "cpu_model": model,
            "mem_total_kb": mem_kb, "period_s": period, "fields": list(schema.TICK_FIELDS),
            **ticker.snap.meta()}


def _sleep_until_phase(period: float, idle: Callable[[float], None]) -> None:
    now = time.time()
    target = math.floor((now - PHASE) / period) * period + PHASE + period
    idle(max(0.0, target - now))


def tick_loop(ticker: Ticker, on_tick: Callable[[float, dict], None], period: float = 1.0,
              seconds: float | None = None, idle: Callable[[float], None] = time.sleep) -> None:
    """Call on_tick(wall_time, tick) every period until SIGTERM/SIGINT or `seconds`.
    Between ticks `idle(timeout)` waits; the daemon passes one that serves its
    stream and PSI triggers (it must return within `timeout`)."""
    stop = False

    def _stop(*_):
        nonlocal stop
        stop = True

    old = {s: signal.signal(s, _stop) for s in (signal.SIGTERM, signal.SIGINT)}
    t_end = None if seconds is None else time.monotonic() + seconds
    try:
        _sleep_until_phase(period, idle)
        ticker.tick()  # discard: its interval started at construction, off-phase
        while not stop and (t_end is None or time.monotonic() < t_end):
            _sleep_until_phase(period, idle)
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


class SpoolSink:
    """Daily tick files `<dir>/ticks-<host>-YYYYMMDD.jsonl` (UTC day), each with
    its own meta line, so a validation run can replay any period. A finished
    day is gzip-compressed in a background thread (the tick loop never waits)
    and days older than `keep_days` are dropped: the spool is the collector's
    own bounded log, like svlogd's rotation (~62 MB/day raw, ~8 MB gzipped)."""

    def __init__(self, directory: str, meta: dict, keep_days: int = 14):
        self.dir, self.meta, self.keep_days = directory, meta, keep_days
        self.host = meta["host"]
        self.day, self.sink = None, None
        self._hk: threading.Thread | None = None
        os.makedirs(directory, exist_ok=True)

    def _path(self, day: str) -> str:
        return os.path.join(self.dir, f"ticks-{self.host}-{day}.jsonl")

    def __call__(self, t: float, tick: dict) -> None:
        day = time.strftime("%Y%m%d", time.gmtime(t))
        if day != self.day:
            if self.sink:
                self.sink.close()
            if self._hk:  # finished hours ago unless the day began right after a start (or in tests)
                self._hk.join()
            self.day, self.sink = day, JsonlSink(self._path(day), self.meta)
            cutoff = time.strftime("%Y%m%d", time.gmtime(t - self.keep_days * 86400))
            self._hk = threading.Thread(target=self._housekeep, args=(day, cutoff), daemon=True)
            self._hk.start()
        self.sink(t, tick)

    def _housekeep(self, today: str, cutoff: str) -> None:
        for p in glob.glob(os.path.join(self.dir, f"ticks-{self.host}-*.jsonl*")):
            day = os.path.basename(p)[len(f"ticks-{self.host}-"):][:8]
            if day < cutoff:
                os.remove(p)
            elif p.endswith(".jsonl") and day < today:  # never today's (or a later) file
                with open(p, "rb") as src, gzip.open(p + ".gz.tmp", "wb", compresslevel=6) as dst:
                    shutil.copyfileobj(src, dst)
                os.replace(p + ".gz.tmp", p + ".gz")
                os.remove(p)

    def close(self, wait: float | None = 5.0) -> None:
        """Close today's file; give a running compression `wait` seconds (an
        interrupted one leaves the .jsonl, which the next start compresses)."""
        if self.sink:
            self.sink.close()
        if self._hk:
            self._hk.join(wait)


def spool_files(directory: str, host: str) -> list[str]:
    """Spool files of one host in day order. A finished day is .jsonl.gz; a copy
    taken while that day was being compressed may also hold its .jsonl (partial
    or complete), so the .gz wins and temporary files are ignored."""
    pat = re.compile(rf"ticks-{re.escape(host)}-(\d{{8}})\.jsonl(\.gz)?$")
    days: dict[str, str] = {}
    for name in sorted(os.listdir(directory)) if os.path.isdir(directory) else []:
        m = pat.match(name)
        if m and (m.group(2) or m.group(1) not in days):
            days[m.group(1)] = os.path.join(directory, name)
    return [days[d] for d in sorted(days)]


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
