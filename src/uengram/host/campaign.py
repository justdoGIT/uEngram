"""Validation campaign: record, commission, scheduled fault matrix, report.

    plan    write <dir>/campaign.json: phases and a randomised fault schedule
    tick    advance the campaign by what is due (idempotent; the runit service
            `uengram-campaign` calls it every minute, so a reboot loses nothing)
    status  print the phase and fault counts

Phases (defaults): [start, commission) = 7 days in which the nodes record and
nothing is injected. The report uses that week for the clean false-alarm
check: commission offline on its first `clean_days` (3), replay the rest.
At `commission` every node is commissioned on its own whole week, which
covers the daily cycle, and the models are deployed. During the fault week
[commission, end) faults run on schedule, one at a time fleet-wide, each
logged with its actual start and end in faults.jsonl: every fault type the
node allows `per_type` (10) times, cycling over its strengths. After `end`
the spools and online events are collected and the report is written
(uengram.host.validate).

Safety: memory faults are confined to cgroups; CPU loops run at nice 19; disk
faults only read; power caps and netem undo themselves (fault-inject.sh); a
fault is postponed by 10 minutes while the node is >50 % busy or runs a Slurm
job, and skipped after 6 postponements. Faults that need root, and the model
deploy, wait (without spending postponements) while msi's kepr store is
locked, and raise an alert at most once an hour. The fault types per node are
in campaign.json (NODES: no disk faults on asus, whose disk is failing; no
disk, mem or netem faults on msi, the desktop, controller and NFS server).
"""
from __future__ import annotations

import fcntl
import json
import os
import random
import subprocess
import time

from .campaign_exec import SUDO_FAULTS, alert, execute, gate, need_mb, push_helper, run_ssh, vault_unlocked

DAY = 86400
DURATION = 180
FAULT_GAP = 600  # seconds between the end of one fault and the start of the next, fleet-wide
NODE_GAP = 2700  # seconds between two faults touching the same node
# Strengths per fault type, listed weakest first (the report's criteria use the
# first and last entry). cpu: share of CPUs spinning; mem: GB working set in a
# 1 GiB cgroup; disk, net: MB/s, 0 = unpaced; power: package PL1 as a share of
# TDP while all CPUs spin; netem: % egress loss (with NETEM_DELAY_MS) while
# streaming NETEM_MBPS to a peer; leak: MiB/s allocated and never freed.
LEVELS = {"cpu": [0.25, 0.5, 1.0], "mem": [1.25, 1.75, 2.5], "oom": [1], "disk": [10, 40, 0],
          "net": [2, 8, 0], "power": [0.5, 0.3], "netem": [1, 5], "leak": [2, 8]}
VAULT_ALERT_EVERY = 3600
NODES = {
    # msi's /home is nearly full: its ~/.local/state/uengram-host/spool is a symlink to the backup disk.
    # msi is the desktop, the campaign controller and the NFS server: no disk, mem or netem faults there.
    "msi": {"ssh": "msi-worker", "ip": "192.168.1.2", "faults": ["cpu", "oom", "net", "power", "leak"]},
    "hp": {"ssh": "hp", "ip": "192.168.1.3",
           "faults": ["cpu", "mem", "oom", "disk", "net", "power", "netem", "leak"]},
    # asus: no disk faults on its failing HDD
    "asus": {"ssh": "asus-worker", "ip": "192.168.1.5",
             "faults": ["cpu", "mem", "oom", "net", "power", "netem", "leak"]},
    # lenovo: no power faults, its RAPL power-limit zone is disabled by firmware (enabled=0, writes return EIO)
    "lenovo": {"ssh": "lenovo-worker", "ip": "192.168.1.4",
               "faults": ["cpu", "mem", "oom", "disk", "net", "netem", "leak"]},
    # gigabyte: no power faults. dev-setup's cpu-cap service holds every CPU at 1.5 GHz (issue #200), where
    # the package draws 6.8 W under full load; PL1 at 30 % of TDP is 16 W, so the cap never binds
    # (measured: the reference build took 70.5-70.8 s and 480 J at PL1 84, 35 and 25 W).
    "gigabyte": {"ssh": "gigabyte-worker", "ip": "192.168.1.6",
                 "faults": ["cpu", "mem", "oom", "disk", "net", "netem", "leak"]},
}


def _now() -> float:
    return time.time()


def plan(directory: str, start: float, record_days: float = 7, test_days: float = 7, per_type: int = 10,
         seed: int = 0, nodes: dict | None = None, alpha: float = 0.01, clean_days: float = 3) -> dict:
    nodes = nodes or NODES
    if not 0 < clean_days < record_days:
        raise ValueError("clean_days must lie inside the recording week")
    rng = random.Random(seed)
    t_c, t_end = start + record_days * DAY, start + (record_days + test_days) * DAY
    first, last = t_c + 3600, t_end - 1800  # leave the first hour to the freshly deployed models
    want = []  # (node, type, level): per_type of each allowed type, cycling over its levels
    for name, cfg in nodes.items():
        for ftype in cfg["faults"]:
            levels = list(LEVELS[ftype])
            rng.shuffle(levels)
            want += [(name, ftype, levels[i % len(levels)]) for i in range(per_type)]
    rng.shuffle(want)
    busy: dict[str, list[float]] = {n: [] for n in nodes}
    sched, t_all = [], []
    for i, (node, ftype, level) in enumerate(want):
        peer = None
        if ftype in ("net", "netem"):  # a stream needs a receiving peer
            peers = [p for p, c in nodes.items() if p != node and "net" in c["faults"]]
            peer = rng.choice(peers)
        touched = [node] + ([peer] if peer else [])
        for _ in range(500):
            t = rng.uniform(first, last)
            if all(abs(t - u) >= DURATION + FAULT_GAP for u in t_all) and \
                    all(abs(t - u) >= NODE_GAP for n in touched for u in busy[n]):
                break
        else:
            continue  # the test phase is full; fewer faults than asked
        t_all.append(t)
        for n in touched:
            busy[n].append(t)
        sched.append({"id": i, "type": ftype, "level": level, "node": node, "peer": peer,
                      "planned": round(t, 1), "status": "pending", "postponed": 0})
    sched.sort(key=lambda f: f["planned"])
    from .validate_criteria import criteria  # stdlib only
    camp = {"start": start, "clean_split": start + clean_days * DAY, "commission": t_c, "end": t_end,
            "alpha": alpha, "nodes": nodes, "levels": LEVELS, "duration_s": DURATION, "schedule": sched,
            "commissioned": None, "report": None,
            # pre-registered: written before any test data exists; the report evaluates exactly these
            "criteria": criteria(alpha), "criteria_registered": _now()}
    os.makedirs(directory, exist_ok=True)
    _save(directory, camp)
    return camp


def _save(directory: str, camp: dict) -> None:
    tmp = os.path.join(directory, "campaign.json.tmp")
    with open(tmp, "w") as f:
        json.dump(camp, f, indent=1)
    os.replace(tmp, os.path.join(directory, "campaign.json"))


def load(directory: str) -> dict:
    with open(os.path.join(directory, "campaign.json")) as f:
        return json.load(f)


def _run_due_faults(directory: str, camp: dict) -> None:
    for f in camp["schedule"]:
        if f["status"] != "pending" or f["planned"] > _now():
            continue
        if _now() > camp["end"] - 600:
            f["status"] = "skipped"
            f["reason"] = "test phase over"
            continue
        missing = [n for n in (f["node"], f["peer"]) if n and "error" in camp["commissioned"].get(n, {"error": ""})]
        if missing:
            f.update(status="skipped", reason=f"not commissioned: {missing}")
            _save(directory, camp)
            continue
        if f["type"] in SUDO_FAULTS and not _vault_ok(directory, camp):
            f.update(planned=_now() + 600, reason="kepr store on msi is locked")  # no postponement spent
            _save(directory, camp)
            continue
        last_end = max((g["end"] for g in camp["schedule"] if "end" in g), default=0.0)
        if _now() < last_end + FAULT_GAP:
            return  # a postponed fault waits until the previous one has faded
        names = [f["node"]] + ([f["peer"]] if f["peer"] else [])
        reasons = []
        try:
            for n in names:
                push_helper(camp["nodes"][n]["ssh"])
            senders = names if f["type"] not in ("net", "netem") else names[:1]  # the receiver only listens
            for n in senders:
                need = need_mb(f, camp["duration_s"]) if n == f["node"] else 0.0
                good, why = gate(n, camp["nodes"][n]["ssh"], need)
                reasons.append(f"{n}: {why}")
                if not good:
                    raise RuntimeError(f"{n} busy: {why}")
        except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as e:
            f["postponed"] += 1
            f["planned"] = _now() + 600
            f["status"] = "skipped" if f["postponed"] > 6 else "pending"
            f["reason"] = str(e)[:300]
            _save(directory, camp)
            continue
        t0 = _now()
        try:
            result = execute(camp, f)
            # a non-zero exit means the fault did not (fully) happen: kept out of detection, interval kept
            f.update(status="done" if result.get("rc") == 0 else "failed", result=result)
        except (OSError, subprocess.SubprocessError) as e:  # it may have run partly: keep its interval
            f.update(status="failed", reason=str(e)[:300])
        f.update(start=t0, end=_now(), gate="; ".join(reasons))
        with open(os.path.join(directory, "faults.jsonl"), "a") as log:
            log.write(json.dumps(f) + "\n")
        _save(directory, camp)
        return  # one fault per tick: the next waits for its own planned time


def _vault_ok(directory: str, camp: dict) -> bool:
    """True when sudo through kepr works; otherwise alert (at most hourly) and say no."""
    if vault_unlocked():
        return True
    if _now() - camp.get("vault_alerted", 0.0) >= VAULT_ALERT_EVERY:
        camp["vault_alerted"] = _now()
        _save(directory, camp)
        alert("CRITICAL", "uengram campaign blocked: kepr locked",
              "msi's kepr store is locked; faults needing sudo and the model deploy wait. "
              "Unlock: XDG_RUNTIME_DIR=/run/user/1000 kepr unlock")
    return False


def _state_dir(cfg: dict) -> str:
    return cfg.get("state", ".local/state/uengram-host")  # relative to the remote home


def _collect(directory: str, camp: dict) -> None:
    """rsync spools and online events of every node into <dir>/nodes/<name>/."""
    for name, cfg in camp["nodes"].items():
        dst = os.path.join(directory, "nodes", name)
        os.makedirs(os.path.join(dst, "spool"), exist_ok=True)
        state = _state_dir(cfg)
        # spool/ may be a symlink to a bigger disk (msi); the trailing slash follows it
        subprocess.run(["rsync", "-a", f"{cfg['ssh']}:{state}/spool/", f"{dst}/spool/"], timeout=3600)
        subprocess.run(["rsync", "-a", f"{cfg['ssh']}:{state}/events.jsonl", f"{dst}/"], timeout=600)


def _commission_and_deploy(directory: str, camp: dict) -> None:
    from .validate import commission_from_spool  # numpy is needed only here and in the report
    _collect(directory, camp)
    out = {}
    for name, cfg in camp["nodes"].items():
        try:
            node = commission_from_spool(directory, name, camp["start"], camp["commission"], camp["alpha"])
        except (OSError, ValueError) as e:  # no spool, too short: this node sits the test out
            out[name] = {"error": str(e)}
            continue
        path = os.path.join(directory, "models", f"model-{name}.pkl")
        node.save(path)
        remote = f"{_state_dir(cfg)}/model-{name}.pkl"
        # stop first: the daemon saves its in-memory model on SIGTERM, which
        # would overwrite a model copied in while it runs
        stop = run_ssh(cfg["ssh"], "sv stop uengram-hostd", sudo=True, timeout=120)
        cp = subprocess.run(["scp", "-q", path, f"{cfg['ssh']}:{remote}"], timeout=120)
        start = run_ssh(cfg["ssh"], "sv start uengram-hostd", sudo=True, timeout=120)
        ok = stop.returncode == 0 and cp.returncode == 0 and start.returncode == 0
        out[name] = {"deployed": _now(), "threshold": node.alarm_threshold, "report": node.commission_report}
        if not ok:
            out[name]["error"] = f"deploy rc stop={stop.returncode} scp={cp.returncode} start={start.returncode}"
    camp["commissioned"] = out


def tick(directory: str) -> str:
    with open(os.path.join(directory, ".lock"), "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return "busy: another tick is running"
        camp = load(directory)
        now = _now()
        if now < camp["commission"]:
            return "recording"
        if camp["commissioned"] is None:
            if not _vault_ok(directory, camp):  # the deploy restarts the daemons through sudo
                return "waiting: kepr store on msi is locked"
            _commission_and_deploy(directory, camp)
            _save(directory, camp)
            return "commissioned"
        if now < camp["end"]:
            _run_due_faults(directory, camp)
            return "testing"
        if camp["report"] is None:
            for f in camp["schedule"]:
                if f["status"] == "pending":
                    f.update(status="skipped", reason="test phase over")
            _collect(directory, camp)
            from .validate import write_report
            camp["report"] = write_report(directory)
            _save(directory, camp)
            return "reported"
        return "done"


def status(directory: str) -> str:
    camp = load(directory)
    counts: dict[str, int] = {}
    for f in camp["schedule"]:
        counts[f["status"]] = counts.get(f["status"], 0) + 1
    fmt = lambda t: time.strftime("%Y-%m-%d %H:%M", time.localtime(t))
    phase = ("recording" if _now() < camp["commission"] else "testing" if _now() < camp["end"]
             else "reported" if camp["report"] else "awaiting report")
    nxt = next((f for f in camp["schedule"] if f["status"] == "pending"), None)
    lines = [
        f"phase: {phase}; start {fmt(camp['start'])}, commission {fmt(camp['commission'])}, end {fmt(camp['end'])}",
        f"faults: {counts}",
        f"next: {nxt['type']} level {nxt['level']} on {nxt['node']} at {fmt(nxt['planned'])}" if nxt else "next: none",
        f"report: {camp['report']}" if camp["report"] else "",
        "" if vault_unlocked() else "WARNING: msi's kepr store is locked: faults needing sudo and the model "
                                    "deploy wait (XDG_RUNTIME_DIR=/run/user/1000 kepr unlock)",
    ]
    return "\n".join(x for x in lines if x)
