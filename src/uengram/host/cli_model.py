"""Model subcommands (numpy is imported only when one of them runs)."""
from __future__ import annotations

import json
import os
import socket
import sys
import time

from .cli import DEFAULT_DIR


def _commission(args) -> int:
    from .features import read_columns, select_kernel
    from .node import NodeMemory
    meta, cols = read_columns([os.path.expanduser(p) for p in args.ticks])
    if args.kernel is not None:
        cols = select_kernel(cols, args.kernel)
    exclude = tuple(tuple(float(x) for x in e.split(":")) for e in args.exclude)
    node = NodeMemory.commission(cols, meta, window=args.window, alpha=args.alpha,
                                 calib_frac=args.calib_frac, exclude=exclude)
    node.save(args.model)
    print(json.dumps(node.commission_report, indent=1))
    return 0


def _autocommission(args) -> int:
    """Recommission a deployed model once the running kernel has enough
    same-kernel spool data; otherwise a no-op. Writes the new model to
    `--out` rather than `--model`, so a caller (the uengram-recommission
    service) controls the daemon stop/swap/start and the model file is
    never replaced while uengram-hostd might still be writing it."""
    import platform
    from .collector import spool_files
    from .features import kernel_releases, read_columns, select_kernel
    from .node import NodeMemory

    running = args.kernel or platform.release()
    out = args.out or (args.model + ".staging")
    deployed_kernel = None
    if os.path.exists(args.model):
        try:
            deployed_kernel = NodeMemory.load(args.model).meta.get("kernel")
        except Exception as e:  # a corrupt or foreign pickle must not crash the checker
            print(json.dumps({"action": "error", "detail": f"cannot load {args.model}: {e}"}))
            return 1
    if deployed_kernel == running:
        print(json.dumps({"action": "up_to_date", "kernel": running}))
        return 0

    files = spool_files(os.path.expanduser(args.spool), socket.gethostname())
    if not files:
        print(json.dumps({"action": "insufficient_data", "reason": "no spool files", "kernel": running}))
        return 0
    meta, cols = read_columns(files)
    releases = kernel_releases(cols)
    if running not in releases:
        print(json.dumps({"action": "insufficient_data", "reason": "no ticks for the running kernel yet",
                           "kernel": running, "spool_releases": list(releases)}))
        return 0
    sel = select_kernel(cols, running)
    hours = len(sel["t"]) / 3600.0  # 1 Hz ticks
    if hours < args.min_hours:
        print(json.dumps({"action": "insufficient_data", "kernel": running,
                           "hours": round(hours, 2), "min_hours": args.min_hours}))
        return 0

    node = NodeMemory.commission(sel, meta, window=args.window, alpha=args.alpha, calib_frac=args.calib_frac)
    node.save(out)
    print(json.dumps({"action": "recommissioned", "kernel": running, "hours": round(hours, 2),
                       "out": out, "report": node.commission_report}))
    return 0


def _replay(args) -> int:
    from .collector import read_ticks
    from .node import NodeMemory
    node = NodeMemory.load(args.model)
    meta, ticks = read_ticks(args.ticks)
    expected = node.meta.get("kernel")
    releases = tuple(sorted({str(tick.get("kernel") or meta.get("kernel"))
                             for tick in ticks if tick.get("kernel") or meta.get("kernel")}))
    if expected and releases and releases != (str(expected),):
        raise ValueError(f"model kernel {expected!r} does not match replay data kernels {releases}")
    out = open(args.events, "w") if args.events else sys.stdout
    for rec in node.feed_ticks(ticks):
        out.write(json.dumps(rec) + "\n")
    if args.save:
        node.save(args.model)
    return 0


def _run(args) -> int:
    from .daemon import run
    return run(args.model, args.state, args.events, os.path.expanduser(args.spool) if args.spool else None,
               args.socket, args.save_every)


def _status(args) -> int:
    with open(args.state) as f:
        st = json.load(f)
    from .cluster import format_node
    print(format_node(st, st.get("pressure_events")))
    return 0


def _follow(args) -> int:
    from .stream import follow
    return follow(args.socket)


def _hub(args) -> int:
    import signal
    from .hub import run, ssh_follow
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # runit stop: close the followers
    return run({n: ssh_follow(n) for n in args.nodes}, os.path.expanduser(args.out))


def _cluster(args) -> int:
    from .cluster import format_cluster, read_snapshot
    try:
        snap = read_snapshot(os.path.expanduser(args.snapshot))
    except FileNotFoundError:
        print(f"no hub snapshot at {args.snapshot}: the hub runs on msi (uengram-host hub)", file=sys.stderr)
        return 1
    print(format_cluster(snap))
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


def _tune(args) -> int:
    from .tune import format_result, tune
    name = args.name or args.node.removesuffix("-worker")
    res = tune(args.node, name, args.knob, args.values, reps=args.reps, jobs=args.jobs,
               files_per_job=args.files_per_job, funcs=args.funcs, settle_s=args.settle,
               tolerance=args.tolerance, seed=args.seed, campaign_dir=args.campaign, apply=args.apply)
    print(format_result(res))
    return 0


def add_model_commands(sub) -> dict:
    host = socket.gethostname()
    model = os.path.join(DEFAULT_DIR, f"model-{host}.pkl")
    state = os.path.join(DEFAULT_DIR, "state.json")
    sock = os.path.join(DEFAULT_DIR, "stream.sock")
    snapshot = f"/dev/shm/uengram-cluster-{os.environ.get('USER') or os.getuid()}.json"
    c = sub.add_parser("commission")
    c.add_argument("--ticks", nargs="+", required=True, help="tick files or a node's spool files (.jsonl[.gz])")
    c.add_argument("--model", default=model)
    c.add_argument("--window", type=int, default=32, help="ticks per window (K)")
    c.add_argument("--alpha", type=float, default=0.01, help="alarm false-positive rate")
    c.add_argument("--calib-frac", type=float, default=0.3,
                   help="tail share of the commissioning run held out for alarm calibration")
    c.add_argument("--kernel", help="only commission ticks from this kernel release")
    c.add_argument("--exclude", action="append", default=[], metavar="LO:HI",
                   help="wall-time interval (epoch s) left out, e.g. an injected fault; repeatable")
    r = sub.add_parser("replay")
    r.add_argument("--model", default=model)
    r.add_argument("--ticks", required=True)
    r.add_argument("--events")
    r.add_argument("--save", action="store_true", help="keep what the memory learned")
    ac = sub.add_parser("autocommission")
    ac.add_argument("--model", default=model)
    ac.add_argument("--spool", default=os.path.join(DEFAULT_DIR, "spool"))
    ac.add_argument("--kernel", help="running kernel release (default: platform.release())")
    ac.add_argument("--min-hours", type=float, default=24.0,
                     help="hours of same-kernel spool data required before recommissioning")
    ac.add_argument("--out", help="write a ready model here instead of --model (default: MODEL.staging)")
    ac.add_argument("--window", type=int, default=32, help="ticks per window (K)")
    ac.add_argument("--alpha", type=float, default=0.01, help="alarm false-positive rate")
    ac.add_argument("--calib-frac", type=float, default=0.3,
                     help="tail share of the commissioning run held out for alarm calibration")
    o = sub.add_parser("run")
    o.add_argument("--model", default=model)
    o.add_argument("--state", default=state)
    o.add_argument("--events", default=os.path.join(DEFAULT_DIR, "events.jsonl"))
    o.add_argument("--save-every", type=int, default=50, help="windows between model saves")
    o.add_argument("--spool", help="also keep every tick in this daily spool directory")
    o.add_argument("--socket", default=sock, help="stream socket for `follow` readers")
    s = sub.add_parser("status")
    s.add_argument("--state", default=state)
    f = sub.add_parser("follow")
    f.add_argument("--socket", default=sock)
    h = sub.add_parser("hub")
    h.add_argument("--nodes", nargs="+", required=True, help="ssh names of the nodes to follow")
    h.add_argument("--out", default=snapshot)
    k = sub.add_parser("cluster")
    k.add_argument("--snapshot", default=snapshot, help="the hub's snapshot file")
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
    t = sub.add_parser("tune")
    t.add_argument("--node", required=True, help="ssh name of the node")
    t.add_argument("--name", help="Slurm node name (default: the ssh name without -worker)")
    t.add_argument("--knob", choices=["pl1", "epp", "scx"], required=True)
    t.add_argument("--values", nargs="+", required=True,
                   help="pl1: watts; epp: preference names; scx: scheduler names (scx_NAME) or none")
    t.add_argument("--reps", type=int, default=6, help="blocks; each runs every value once, in random order")
    t.add_argument("--jobs", type=int, help="parallel gcc processes (default: the node's CPUs)")
    t.add_argument("--files-per-job", type=int, default=16)
    t.add_argument("--funcs", type=int, default=150, help="functions per generated file")
    t.add_argument("--settle", type=float, default=20.0, help="seconds between setting a value and the build")
    t.add_argument("--tolerance", type=float, default=0.05, help="largest accepted slowdown (share)")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--campaign", default="~/Backup/uengram-host/campaign",
                   help="refuse while the campaign in this directory runs")
    t.add_argument("--apply", action="store_true", help="set the recommended pl1/epp until the next reboot")
    return {"commission": _commission, "autocommission": _autocommission, "replay": _replay, "run": _run,
            "status": _status, "follow": _follow, "hub": _hub, "cluster": _cluster, "campaign": _campaign,
            "tune": _tune}
