"""The online daemon: tick, window, remember, publish (numpy).

Every second: one tick, appended to the spool and pushed to stream readers
(its CPU and stall counts). Every window: the memory's record, appended to
events.jsonl, written to state.json and pushed. Between ticks the loop waits
in poll() on the stream socket and the PSI trigger fds, so a stall beyond the
commissioned 2 s bound is pushed within the trigger window instead of at the
end of the 32 s window. All file writes go through collector.AsyncWriter.
"""
from __future__ import annotations

import json
import os
import pickle
import select
import sys
import time

from .cluster import LOAD_FIELDS
from .collector import AsyncWriter, SpoolSink, host_meta, open_collector, tick_loop
from .node import NodeMemory
from .psi import PsiTriggers
from .stream import StateStream, atomic_json


def _write_bytes(path: str, blob: bytes) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(blob)
    os.replace(tmp, path)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def run(model: str, state: str, events_path: str, spool_dir: str | None, socket_path: str,
        save_every: int = 50) -> int:
    node = NodeMemory.load(model)
    ticker = open_collector(pmu=True)
    writer = AsyncWriter()
    spool = SpoolSink(spool_dir, host_meta(ticker, 1.0)) if spool_dir else None
    events = open(events_path, "a", buffering=1)
    poller = select.poll()
    stream = StateStream(socket_path)
    stream.register(poller)
    try:
        psi = PsiTriggers(node.fast_bound)
        psi.register(poller)
        _log(f"PSI triggers ('some' stall per {psi.window_s} s): {psi.thresholds} us")
    except OSError as e:  # a kernel without PSI or without unprivileged triggers
        psi = None
        _log(f"PSI triggers unavailable ({e}); stalls are flagged at window ends only")
    pressure: dict[str, float] = {}  # resource -> wall time of its last trigger
    last: dict[str, dict] = {}

    def publish(st: dict) -> None:
        st = {**st, "pressure_events": dict(pressure),
              "writer": {"dropped": writer.dropped, "failed": writer.failed}}
        last["state"] = st
        writer.submit(atomic_json, state, st, 0o644)
        stream.publish({"type": "state", **st})

    def idle(timeout: float) -> None:
        end = time.monotonic() + timeout
        while (left := end - time.monotonic()) > 0:
            for fd, ev in poller.poll(left * 1000.0):
                if stream.handle(fd, ev) or psi is None or (res := psi.resource(fd)) is None:
                    continue
                if ev & (select.POLLERR | select.POLLNVAL):  # the trigger is gone: stop polling it
                    poller.unregister(fd)
                    _log(f"PSI trigger for {res} failed (poll events {ev:#x}); no longer watched")
                    continue
                now = time.time()
                pressure[res] = now
                stream.publish({"type": "pressure", "host": node.host, "t": now, "resource": res,
                                "threshold_us": psi.thresholds[res], "window_s": psi.window_s})
                if "state" in last:
                    publish(last["state"])

    def on_tick(t: float, tick: dict) -> None:
        if spool:
            writer.submit(spool, t, tick)
        stream.publish({"type": "tick", "t": t, **{f: tick.get(f) for f in LOAD_FIELDS}})
        for rec in node.feed_ticks([{"t": t, **tick}]):
            writer.submit(events.write, json.dumps(rec) + "\n")
            publish(node.state(rec))
            if node.windows_seen % save_every == 0:
                writer.submit(_write_bytes, model, pickle.dumps(node, protocol=pickle.HIGHEST_PROTOCOL))

    try:
        tick_loop(ticker, on_tick, 1.0, idle=idle)
    finally:
        writer.close()
        node.save(model)
        events.close()
        if spool:
            spool.close()
        stream.close()
        if psi:
            psi.close()
    return 0
