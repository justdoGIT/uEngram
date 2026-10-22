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
faults only read; a fault is postponed by 10 minutes while the node is >50 %
busy or runs a Slurm job, and skipped after 6 postponements. The fault types
per node are in campaign.json (default: no disk faults on asus, whose disk
is failing, and only CPU, OOM and network faults on the msi desktop).
"""
from __future__ import annotations

import fcntl
import json
import os
import random
import subprocess
import time

DAY = 86400
DURATION = 180
FAULT_GAP = 600  # seconds between the end of one fault and the start of the next, fleet-wide
NODE_GAP = 2700  # seconds between two faults touching the same node
LEVELS = {"cpu": [0.25, 0.5, 1.0], "mem": [1.25, 1.75, 2.5], "oom": [1], "disk": [10, 40, 0],
          "net": [2, 8, 0]}
NODES = {
    # msi's /home is nearly full: its ~/.local/state/uengram-host/spool is a symlink to the backup disk
    "msi": {"ssh": "msi-worker", "ip": "192.168.1.2", "faults": ["cpu", "oom", "net"]},
    "hp": {"ssh": "hp", "ip": "192.168.1.3", "faults": ["cpu", "mem", "oom", "disk", "net"]},
    "asus": {"ssh": "asus-worker", "ip": "192.168.1.5", "faults": ["cpu", "mem", "oom", "net"]},
    "lenovo": {"ssh": "lenovo-worker", "ip": "192.168.1.4", "faults": ["cpu", "mem", "oom", "disk", "net"]},
    "gigabyte": {"ssh": "gigabyte-worker", "ip": "192.168.1.6", "faults": ["cpu", "mem", "oom", "disk", "net"]},
}
HELPER = "/tmp/uengram-fault-inject.sh"
PACKAGED_HELPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fault-inject.sh")


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
        if ftype == "net":
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
    camp = {"start": start, "clean_split": start + clean_days * DAY, "commission": t_c, "end": t_end,
            "alpha": alpha, "nodes": nodes, "levels": LEVELS, "duration_s": DURATION, "schedule": sched,
            "commissioned": None, "report": None}
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


def _ssh(host: str, cmd: str, sudo: bool = False, timeout: float = 600) -> subprocess.CompletedProcess:
    """Run cmd on host; with sudo, the password goes from kepr straight into the
    ssh pipe (never through this process's memory, argv or a file)."""
    if not sudo:
        return subprocess.run(["ssh", "-o", "BatchMode=yes", host, cmd], capture_output=True, text=True,
                              timeout=timeout)
    kepr = subprocess.Popen(["kepr", "get", "KK_SUDO_PASSWORD", "--quiet"], stdout=subprocess.PIPE)
    try:
        return subprocess.run(["ssh", "-o", "BatchMode=yes", host, f"sudo -S -p '' {cmd}"], stdin=kepr.stdout,
                              capture_output=True, text=True, timeout=timeout)
    finally:
        kepr.stdout.close()
        kepr.wait()


def _push_helper(host: str) -> None:
    subprocess.run(["scp", "-q", PACKAGED_HELPER, f"{host}:{HELPER}"], check=True, timeout=60)


def _slurm_jobs(name: str) -> int:
    """Jobs on node `name`, asked from here (msi runs slurmctld; a worker's own
    squeue can stall a minute when its client config is missing)."""
    r = subprocess.run(["squeue", "-h", "-w", name], capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise RuntimeError(f"squeue failed: {r.stderr.strip()[:200]}")  # unknown: postpone
    return len(r.stdout.splitlines())


def _gate(name: str, host: str) -> tuple[bool, str]:
    r = _ssh(host, f"sh {HELPER} check", timeout=60)
    if r.returncode != 0:
        return False, f"check failed: {r.stderr.strip()[:200]}"
    kv = dict(x.split("=") for x in r.stdout.split())
    jobs = _slurm_jobs(name)
    why = f"{r.stdout.strip()} slurm_jobs={jobs}"
    return float(kv["busy"]) <= 0.5 and jobs == 0, why


def _execute(camp: dict, f: dict) -> dict:
    nodes = camp["nodes"]
    host, dur, lv = nodes[f["node"]]["ssh"], camp["duration_s"], f["level"]
    if f["type"] == "cpu":
        return {"rc": _ssh(host, f"sh {HELPER} spin {lv} {dur}").returncode}
    if f["type"] == "mem":
        r = _ssh(host, f"sh {HELPER} mem {lv} {dur}", sudo=True)
        return {"rc": r.returncode, "out": r.stdout.strip()[-200:]}
    if f["type"] == "oom":
        r = _ssh(host, f"sh {HELPER} oom", sudo=True)
        return {"rc": r.returncode, "out": r.stdout.strip()[-200:]}
    if f["type"] == "disk":
        return {"rc": _ssh(host, f"sh {HELPER} diskread {lv} {dur}", sudo=True).returncode}
    if f["type"] == "net":
        peer = nodes[f["peer"]]
        port = 47123
        rx = subprocess.Popen(["ssh", "-o", "BatchMode=yes", peer["ssh"], f"sh {HELPER} netrecv {port} {dur}"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        time.sleep(3)
        tx = _ssh(host, f"sh {HELPER} netsend {peer['ip']} {port} {lv} {dur}")
        rx.wait(timeout=dur + 120)
        return {"rc": tx.returncode, "rx_rc": rx.returncode}
    raise ValueError(f["type"])


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
        last_end = max((g["end"] for g in camp["schedule"] if "end" in g), default=0.0)
        if _now() < last_end + FAULT_GAP:
            return  # a postponed fault waits until the previous one has faded
        names = [f["node"]] + ([f["peer"]] if f["peer"] else [])
        reasons = []
        try:
            for n in names:
                _push_helper(camp["nodes"][n]["ssh"])
            for n in names if f["type"] != "net" else names[:1]:  # the sender must be idle; the receiver only listens
                good, why = _gate(n, camp["nodes"][n]["ssh"])
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
            f.update(status="done", result=_execute(camp, f))
        except (OSError, subprocess.SubprocessError) as e:  # it may have run partly: keep its interval
            f.update(status="failed", reason=str(e)[:300])
        f.update(start=t0, end=_now(), gate="; ".join(reasons))
        with open(os.path.join(directory, "faults.jsonl"), "a") as log:
            log.write(json.dumps(f) + "\n")
        _save(directory, camp)
        return  # one fault per tick: the next waits for its own planned time


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
        stop = _ssh(cfg["ssh"], "sv stop uengram-hostd", sudo=True, timeout=120)
        cp = subprocess.run(["scp", "-q", path, f"{cfg['ssh']}:{remote}"], timeout=120)
        start = _ssh(cfg["ssh"], "sv start uengram-hostd", sudo=True, timeout=120)
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
    return "\n".join([
        f"phase: {phase}; start {fmt(camp['start'])}, commission {fmt(camp['commission'])}, end {fmt(camp['end'])}",
        f"faults: {counts}",
        f"next: {nxt['type']} level {nxt['level']} on {nxt['node']} at {fmt(nxt['planned'])}" if nxt else "next: none",
        f"report: {camp['report']}" if camp["report"] else "",
    ]).rstrip()
