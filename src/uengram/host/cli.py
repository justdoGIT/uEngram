"""uengram-host command line.

  probe                       check every source and print one tick
  record  [--out F | --spool D] [--seconds]   write ticks to a JSONL file or a daily spool (stdlib only)
  commission --ticks F... --model M   fit encoder, seed memory, calibrate alarms
  run     --model M [--spool D]  online: tick, window, encode, remember; state file + events
  replay  --model M --ticks F  offline: feed recorded ticks through a commissioned model
  status  [--state F]          print a node state file
  cluster --nodes ...          collect node states over ssh and print the cluster view
  campaign plan|tick|status|report --dir D   validation campaign (uengram.host.campaign)
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys

DEFAULT_DIR = os.path.expanduser("~/.local/state/uengram-host")


def _probe(args) -> int:
    from .collector import host_meta, open_collector
    from .perf import PerfUnavailable
    try:
        ticker = open_collector(pmu=True)
    except PerfUnavailable as e:
        print(f"PMU: {e}", file=sys.stderr)
        return 2
    import time
    time.sleep(1.0)
    print(json.dumps({"meta": host_meta(ticker, 1.0), "tick": ticker.tick()}, indent=1))
    return 0


def _record(args) -> int:
    from .collector import JsonlSink, SpoolSink, host_meta, open_collector, tick_loop
    ticker = open_collector(pmu=True)
    meta = host_meta(ticker, args.period)
    if args.spool:
        sink, where = SpoolSink(os.path.expanduser(args.spool), meta), args.spool
    else:
        where = args.out or os.path.join(DEFAULT_DIR, f"ticks-{socket.gethostname()}.jsonl")
        sink = JsonlSink(where, meta)
    print(f"recording to {where}", file=sys.stderr, flush=True)
    try:
        tick_loop(ticker, sink, args.period, args.seconds)
    finally:
        sink.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="uengram-host", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe")
    r = sub.add_parser("record")
    r.add_argument("--out")
    r.add_argument("--spool", help="directory of daily tick files (replaces --out)")
    r.add_argument("--seconds", type=float)
    r.add_argument("--period", type=float, default=1.0)
    from .cli_model import add_model_commands
    handlers = {"probe": _probe, "record": _record, **add_model_commands(sub)}
    args = p.parse_args(argv)
    return handlers[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
