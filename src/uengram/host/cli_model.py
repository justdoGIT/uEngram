"""Model subcommands (numpy is imported only when one of them runs)."""
from __future__ import annotations

import json
import os
import socket
import sys

from .cli import DEFAULT_DIR


def _commission(args) -> int:
    from .collector import read_ticks
    from .node import NodeMemory
    meta, ticks = read_ticks(args.ticks)
    node = NodeMemory.commission(ticks, meta, window=args.window, alpha=args.alpha,
                                 calib_frac=args.calib_frac)
    node.save(args.model)
    print(json.dumps(node.commission_report, indent=1))
    return 0


def _replay(args) -> int:
    from .collector import read_ticks
    from .node import NodeMemory
    node = NodeMemory.load(args.model)
    _, ticks = read_ticks(args.ticks)
    out = open(args.events, "w") if args.events else sys.stdout
    for rec in node.feed_ticks(ticks):
        out.write(json.dumps(rec) + "\n")
    if args.save:
        node.save(args.model)
    return 0


def _run(args) -> int:
    from .collector import open_collector, tick_loop
    from .node import NodeMemory
    node = NodeMemory.load(args.model)
    ticker = open_collector(pmu=True)
    events = open(args.events, "a", buffering=1)

    def on_tick(t, tick):
        for rec in node.feed_ticks([{"t": t, **tick}]):
            events.write(json.dumps(rec) + "\n")
            node.write_state(args.state, rec)
            if node.windows_seen % args.save_every == 0:
                node.save(args.model)

    try:
        tick_loop(ticker, on_tick, 1.0)
    finally:
        node.save(args.model)
        events.close()
    return 0


def _status(args) -> int:
    with open(args.state) as f:
        st = json.load(f)
    from .cluster import format_node
    print(format_node(st))
    return 0


def _cluster(args) -> int:
    from .cluster import collect, format_cluster
    states = collect(args.nodes, args.state)
    print(format_cluster(states))
    return 0


def add_model_commands(sub) -> dict:
    host = socket.gethostname()
    model = os.path.join(DEFAULT_DIR, f"model-{host}.pkl")
    state = os.path.join(DEFAULT_DIR, "state.json")
    c = sub.add_parser("commission")
    c.add_argument("--ticks", required=True)
    c.add_argument("--model", default=model)
    c.add_argument("--window", type=int, default=32, help="ticks per window (K)")
    c.add_argument("--alpha", type=float, default=0.01, help="alarm false-positive rate")
    c.add_argument("--calib-frac", type=float, default=0.3,
                   help="tail share of the commissioning run held out for alarm calibration")
    r = sub.add_parser("replay")
    r.add_argument("--model", default=model)
    r.add_argument("--ticks", required=True)
    r.add_argument("--events")
    r.add_argument("--save", action="store_true", help="keep what the memory learned")
    o = sub.add_parser("run")
    o.add_argument("--model", default=model)
    o.add_argument("--state", default=state)
    o.add_argument("--events", default=os.path.join(DEFAULT_DIR, "events.jsonl"))
    o.add_argument("--save-every", type=int, default=50, help="windows between model saves")
    s = sub.add_parser("status")
    s.add_argument("--state", default=state)
    k = sub.add_parser("cluster")
    k.add_argument("--nodes", nargs="+", required=True)
    k.add_argument("--state", default="~/.local/state/uengram-host/state.json")
    return {"commission": _commission, "replay": _replay, "run": _run, "status": _status,
            "cluster": _cluster}
