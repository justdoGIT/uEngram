"""Measure a power or scheduler knob on a fixed build, in randomized blocks; stdlib.

What is being decided: one setting per node for a knob that changes how the
same work is done, judged by joules per build and build time.
* pl1: package power limit (RAPL PL1), watts. msi, hp, asus, gigabyte (lenovo's
  firmware disables the limit).
* epp: HWP energy-performance preference. hp and asus (intel_pstate powersave);
  msi's intel_pstate `performance` mode pins EPP to 0.
* scx: a sched_ext scheduler (`none` = the kernel's EEVDF). Nodes on 7.x kernels.

Design. Only the same work can be compared: instructions per joule rewards
spin loops, and the cluster has no stream of identical jobs to compare (Slurm
accounting is off; sccache-dist work varies). So every run is the same
reference build: a fixed set of generated C files (knob.sh gen) compiled by
gcc -O2 with JOBS in parallel, as kk, on an otherwise idle node. Each rep runs
every value once in a fresh random order (a randomized complete block), so
slow drift (temperature, background) is balanced across values and each rep
gives a paired comparison with the baseline (the node's current setting).
Energy is the package RAPL counter over the build; time is wall time.

Per value v: d = log(E_v / E_base) per rep, mean and paired t interval (95 %),
likewise for time; reported as percent changes. A value is recommended when
its energy interval lies below 0 and its time interval's upper end is at most
log(1 + tolerance); among those the lowest mean energy wins. Otherwise the
baseline stays. Busy CPU time not spent by the build's processes ("foreign")
includes the kernel's own interrupt and housekeeping time: 6.3-6.4 % of
gigabyte's CPU time in five clean builds. A run is discarded when its foreign
share exceeds the experiment's median by more than 2 points (another job ran:
4 busy loops for 6 s of a 74 s build added 4.4 points and 5 % to its time).

Safety. Refuses while the validation campaign runs (it would put knob states
into the clean week). Every set arms a detached restore timer on the node; the
baseline is restored at the end and on any error. It stops when the node's
uEngram error-counter channel alarms (OOM kill, disk timeout, NIC errors,
thermal throttling) or the build reports a throttle. Latent (novelty) alarms
are expected: a new knob value is a new state.
"""
from __future__ import annotations

import json
import math
import os
import random
import statistics
import subprocess
import time

from .campaign_exec import run_ssh, slurm_jobs

HELPER = "/tmp/uengram-knob.sh"
PACKAGED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knob.sh")
BUILD_DIR = "/tmp/uengram-tune-build"
FOREIGN_MARGIN = 0.02
HOLD_S = 900  # restore timer: longer than one block, shorter than an unattended afternoon
# two-sided 97.5 % Student t quantiles, df 1..30
T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145,
        2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048,
        2.045, 2.042]


def campaign_active(directory: str, now: float) -> bool:
    path = os.path.join(directory, "campaign.json")
    if not os.path.exists(path):
        return False
    with open(path) as f:
        c = json.load(f)
    return c["start"] - 3600 <= now <= c["end"]


def paired_change(base: list[float], other: list[float]) -> dict:
    """Mean percent change of `other` vs `base` over paired reps, with a paired t interval on logs."""
    d = [math.log(o / b) for b, o in zip(base, other)]
    n = len(d)
    if n < 2:
        return {"n": n, "pct": float("nan"), "lo": float("nan"), "hi": float("nan")}
    m, s = statistics.fmean(d), statistics.stdev(d)
    h = (T975[n - 2] if n - 1 <= len(T975) else 1.96) * s / math.sqrt(n)
    return {"n": n, "pct": 100 * math.expm1(m), "lo": 100 * math.expm1(m - h), "hi": 100 * math.expm1(m + h)}


def analyse(rows: list[dict], baseline: str, tolerance: float) -> dict:
    done = [r for r in rows if r["rc"] == 0]
    typical = statistics.median(r["foreign_share"] for r in done) if done else 0.0
    for r in rows:
        r["valid"] = r["rc"] == 0 and r["foreign_share"] <= typical + FOREIGN_MARGIN
    ok = [r for r in rows if r["valid"]]
    by = {}
    for r in ok:
        by.setdefault(r["value"], {})[r["rep"]] = r
    out, best = {}, None
    for v, reps in by.items():
        med = {k: statistics.median(r[k] for r in reps.values()) for k in ("wall_s", "energy_j")}
        out[v] = {"runs": len(reps), "median_wall_s": med["wall_s"], "median_energy_j": med["energy_j"],
                  "median_watts": med["energy_j"] / med["wall_s"]}
        if v == baseline:
            continue
        common = sorted(set(reps) & set(by.get(baseline, {})))
        e = paired_change([by[baseline][k]["energy_j"] for k in common], [reps[k]["energy_j"] for k in common])
        t = paired_change([by[baseline][k]["wall_s"] for k in common], [reps[k]["wall_s"] for k in common])
        out[v].update({"energy_change_pct": e, "time_change_pct": t})
        if e["hi"] < 0 and t["hi"] <= 100 * tolerance and (best is None or e["pct"] < out[best]["energy_change_pct"]["pct"]):
            best = v
    return {"values": out, "recommended": best or baseline, "baseline": baseline, "tolerance": tolerance,
            "rule": "energy CI below 0 and time CI upper end <= tolerance; lowest mean energy wins"}


def _kv(text: str) -> dict:
    return {k: float(v) for k, v in (x.split("=") for x in text.split())}


class Tuner:
    def __init__(self, host: str, name: str, knob: str, values: list[str], reps: int, jobs: int | None,
                 files_per_job: int, funcs: int, settle_s: float, seed: int, log: str):
        self.host, self.name, self.knob, self.reps, self.seed = host, name, knob, reps, seed
        self.files_per_job, self.funcs, self.settle_s, self.log = files_per_job, funcs, settle_s, log
        self.values = values
        self.jobs = jobs
        self.baseline: str | None = None
        self.pushed = self.generated = False

    def sh(self, args: str, sudo: bool = False, timeout: float = 900) -> str:
        r = run_ssh(self.host, f"sh {HELPER} {args}", sudo=sudo, timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError(f"{self.name}: knob.sh {args.split()[0]} failed: {(r.stderr or r.stdout).strip()[-300:]}")
        return r.stdout.strip()

    def prepare(self) -> None:
        subprocess.run(["scp", "-q", PACKAGED, f"{self.host}:{HELPER}"], check=True, timeout=60)
        self.pushed = True
        if self.knob == "pl1":
            uw = int(self.sh("get pl1").split()[0])
            self.baseline = str(round(uw / 1e6, 1))
        elif self.knob == "epp":
            self.baseline = self.sh("get epp")
        else:
            self.baseline = "none"
        if self.baseline not in self.values:
            self.values = [self.baseline, *self.values]
        self.jobs = self.jobs or int(run_ssh(self.host, "nproc").stdout)
        self.generated = True  # also when gen fails halfway: clean removes what exists
        self.sh(f"gen {BUILD_DIR} {self.files_per_job * self.jobs} {self.funcs}")

    def apply(self, value: str) -> None:
        if self.knob == "scx":
            if value != "none":
                self.sh(f"scx start {value} {HOLD_S}", sudo=True)
            return
        v = str(int(float(value) * 1e6)) if self.knob == "pl1" else value
        b = str(int(float(self.baseline) * 1e6)) if self.knob == "pl1" else self.baseline
        self.sh(f"set {self.knob} {v} {b} {HOLD_S}", sudo=True)

    def unapply(self, value: str) -> None:
        if self.knob == "scx" and value != "none":
            self.sh("scx stop", sudo=True)

    def restore(self) -> None:
        if not self.pushed:
            return
        if self.knob == "scx":
            self.sh("scx stop", sudo=True)
        elif self.baseline is not None:
            b = str(int(float(self.baseline) * 1e6)) if self.knob == "pl1" else self.baseline
            self.sh(f"restore {self.knob} {b}", sudo=True)

    def wait_idle(self, limit_s: float = 900) -> None:
        end = time.time() + limit_s
        while True:
            busy, jobs = float(self.sh("idle")), slurm_jobs(self.name)
            if busy <= 0.10 and jobs == 0:
                return
            if time.time() > end:
                raise RuntimeError(f"{self.name} not idle for {limit_s:.0f} s (busy {busy:.0%}, {jobs} Slurm jobs)")
            time.sleep(30)

    def event_alarm(self) -> bool:
        r = run_ssh(self.host, "cat ~/.local/state/uengram-host/state.json", timeout=30)
        return r.returncode == 0 and bool(json.loads(r.stdout).get("alarm_event"))

    def run(self, rows: list[dict]) -> None:
        rng = random.Random(self.seed)
        self.wait_idle()
        self.sh(f"build {BUILD_DIR} {self.jobs}", sudo=True, timeout=3600)  # warm-up, discarded
        for rep in range(self.reps):
            order = list(self.values)
            rng.shuffle(order)
            for v in order:
                self.wait_idle()
                if self.event_alarm():
                    raise RuntimeError(f"{self.name}: uEngram error-counter alarm; stopped")
                self.apply(v)
                try:
                    time.sleep(self.settle_s)
                    t0 = time.time()
                    m = _kv(self.sh(f"build {BUILD_DIR} {self.jobs}", sudo=True, timeout=3600))
                finally:
                    self.unapply(v)
                foreign = max(0.0, m["busy_s"] - m["child_s"]) / (m["wall_s"] * m["ncpu"])
                row = {"node": self.name, "knob": self.knob, "value": v, "rep": rep, "start": t0,
                       "end": t0 + m["wall_s"], **m, "foreign_share": foreign}
                rows.append(row)
                with open(self.log, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"{time.strftime('%H:%M:%S')} rep {rep} {self.knob}={v}: {m['wall_s']:.1f} s "
                      f"{m['energy_j']:.0f} J ({m['energy_j'] / m['wall_s']:.1f} W) foreign {foreign:.1%}"
                      + ("" if m["rc"] == 0 else f" build failed (rc {m['rc']:.0f})"), flush=True)
                if m["throttle"] > 0:
                    raise RuntimeError(f"{self.name}: thermal throttling during the build; stopped")


def tune(host: str, name: str, knob: str, values: list[str], reps: int = 6, jobs: int | None = None,
         files_per_job: int = 16, funcs: int = 150, settle_s: float = 20.0, tolerance: float = 0.05,
         seed: int = 0, out_dir: str = "~/.local/state/uengram-host/tune",
         campaign_dir: str = "~/Backup/uengram-host/campaign", apply: bool = False) -> dict:
    if knob not in ("pl1", "epp", "scx"):
        raise ValueError(f"unknown knob {knob}")
    if campaign_active(os.path.expanduser(campaign_dir), time.time()):
        raise RuntimeError("the validation campaign is running; a knob experiment would contaminate it")
    out_dir = os.path.expanduser(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    t = Tuner(host, name, knob, values, reps, jobs, files_per_job, funcs, settle_s, seed,
              os.path.join(out_dir, f"{name}-{knob}-{stamp}.jsonl"))
    rows: list[dict] = []
    try:
        t.prepare()
        t.run(rows)
    finally:
        t.restore()
        if t.generated:
            run_ssh(host, f"sh {HELPER} clean {BUILD_DIR}", timeout=60)
    res = {"node": name, "knob": knob, "jobs": t.jobs, "build": f"{t.files_per_job * t.jobs} files x {funcs} functions",
           **analyse(rows, t.baseline, tolerance)}
    if apply and res["recommended"] != t.baseline:
        if knob == "scx":
            res["applied"] = "no: attach a scheduler persistently through Void's scx runit service"
        else:
            v = str(int(float(res["recommended"]) * 1e6)) if knob == "pl1" else res["recommended"]
            t.sh(f"restore {knob} {v}", sudo=True)  # set without a restore timer: lasts until reboot
            res["applied"] = f"{knob}={res['recommended']} until the next reboot"
    with open(os.path.join(out_dir, f"{name}-{knob}-{stamp}.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def format_result(res: dict) -> str:
    lines = [f"{res['node']} {res['knob']}: reference build {res['build']}, {res['jobs']} jobs; baseline {res['baseline']}"]
    for v, s in res["values"].items():
        line = f"  {v}: {s['runs']} runs, median {s['median_wall_s']:.1f} s, {s['median_energy_j']:.0f} J, {s['median_watts']:.1f} W"
        if "energy_change_pct" in s:
            e, t = s["energy_change_pct"], s["time_change_pct"]
            line += (f"; energy {e['pct']:+.1f} % [{e['lo']:+.1f}, {e['hi']:+.1f}],"
                     f" time {t['pct']:+.1f} % [{t['lo']:+.1f}, {t['hi']:+.1f}] (n={e['n']})")
        lines.append(line)
    lines.append(f"  recommended: {res['recommended']} ({res['rule']}, tolerance {100 * res['tolerance']:.0f} %)"
                 + (f"; applied {res['applied']}" if res.get("applied") else ""))
    return "\n".join(lines)
