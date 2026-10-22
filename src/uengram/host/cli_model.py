"""Model subcommands (numpy is imported only when one of them runs)."""
from __future__ import annotations

import json
import os
import socket
import sys
import time

from .cli import DEFAULT_DIR


def _commission(args) -> int:
    from .features import read_columns
    from .node import NodeMemory
    meta, cols = read_columns([os.path.expanduser(p) for p in args.ticks])
    exclude = tuple(tuple(float(x) for x in e.split(":")) for e in args.exclude)
    node = NodeMemory.commission(cols, meta, window=args.window, alpha=args.alpha,
                                 calib_frac=args.calib_frac, exclude=exclude)
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
    from .collector import SpoolSink, host_meta, open_collector, tick_loop
    from .node import NodeMemory
    node = NodeMemory.load(args.model)
    ticker = open_collector(pmu=True)
    spool = SpoolSink(os.path.expanduser(args.spool), host_meta(ticker, 1.0)) if args.spool else None
    events = open(args.events, "a", buffering=1)

    def on_tick(t, tick):
        if spool:
            spool(t, tick)
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
        if spool:
            spool.close()
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


def _campaign(args) -> int:
    from . import campaign
    d = os.path.expanduser(args.dir)
    if args.action == "plan":
        start = time.time() if args.start == "now" else float(args.start)
        camp = campaign.plan(d, start, args.record_days, args.test_days, args.per_type, args.seed,
                             alpha=args.alpha, clean_days=args.clean_days)
        print(f"{len(camp['schedule'])} faults planned")
        print(campaign.status(d))
    elif args.action == "tick":  # the campaign service appends this to <dir>/campaign.log
        print(time.strftime("%Y-%m-%d %H:%M:%S"), campaign.tick(d), flush=True)
    elif args.action == "status":
        print(campaign.status(d))
    else:
        from .validate import write_report
        print(write_report(d, tuple(args.variants) if args.variants else None))
    return 0


def add_model_commands(sub) -> dict:
    host = socket.gethostname()
    model = os.path.join(DEFAULT_DIR, f"model-{host}.pkl")
    state = os.path.join(DEFAULT_DIR, "state.json")
    c = sub.add_parser("commission")
    c.add_argument("--ticks", nargs="+", required=True, help="tick files or a node's spool files (.jsonl[.gz])")
    c.add_argument("--model", default=model)
    c.add_argument("--window", type=int, default=32, help="ticks per window (K)")
    c.add_argument("--alpha", type=float, default=0.01, help="alarm false-positive rate")
    c.add_argument("--calib-frac", type=float, default=0.3,
                   help="tail share of the commissioning run held out for alarm calibration")
    c.add_argument("--exclude", action="append", default=[], metavar="LO:HI",
                   help="wall-time interval (epoch s) left out, e.g. an injected fault; repeatable")
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
    o.add_argument("--spool", help="also keep every tick in this daily spool directory")
    s = sub.add_parser("status")
    s.add_argument("--state", default=state)
    k = sub.add_parser("cluster")
    k.add_argument("--nodes", nargs="+", required=True)
    k.add_argument("--state", default="~/.local/state/uengram-host/state.json")
    g = sub.add_parser("campaign")
    g.add_argument("action", choices=["plan", "tick", "status", "report"])
    g.add_argument("--dir", required=True)
    g.add_argument("--start", default="now", help="plan: epoch seconds or 'now'")
    g.add_argument("--record-days", type=float, default=7, help="plan: fault-free recording before commissioning")
    g.add_argument("--clean-days", type=float, default=3,
                   help="plan: the report's offline false-alarm check commissions on these first days")
    g.add_argument("--test-days", type=float, default=7, help="plan: fault week(s) after commissioning")
    g.add_argument("--per-type", type=int, default=10, help="plan: faults per node and fault type")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--alpha", type=float, default=0.01)
    g.add_argument("--variants", nargs="*", help="report: subset of uengram.host.validate_variants.ALL")
    return {"commission": _commission, "replay": _replay, "run": _run, "status": _status,
            "cluster": _cluster, "campaign": _campaign}
