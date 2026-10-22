"""Validation report for a uengram-host campaign (numpy).

Reads <dir>/campaign.json, faults.jsonl (ground truth written by the
campaign), models/, and nodes/<name>/{spool/, events.jsonl}. For every node it
replays the test phase through the deployed model and answers:

1. Calibration: how often does each channel fire without a fault, and are the
   conformal p-values calibrated (share of p <= a against a)? (a) On the clean
   recording week: commissioned offline on its first days, replayed on the
   rest, nothing injected. (b) On the fault week, deployed model, windows at
   least MARGIN windows away from any fault on the node. Alarms are serially
   dependent, so intervals come from a moving-block bootstrap whose blocks hold
   whole alarm episodes (episode_block), not iid. Also alarm episodes per
   week and step-change supersessions per day (validate_signals).
2. Detection: per fault type and strength, the share detected by any channel
   (Wilson interval), the delay, whether D2 attribution named the expected
   modality and whether a saturation flag named the expected resource.
3. Daemon consistency: the online events.jsonl against the replay of the same
   windows (the deployed path must equal the offline one).
4. Ablations and baselines (validate_variants): what each part contributes and
   whether simpler detectors would do as well.
5. Health (validate_signals): whether the distance to the frozen healthy
   reference ranks fault windows above normal ones.
"""
from __future__ import annotations

import copy
import json
import math
import os
import time

import numpy as np

from .collector import spool_files
from .features import read_columns, select, windows_from_columns
from .node import NodeMemory, Options
from .validate_criteria import wilson
from .validate_signals import alarm_episodes, health_ranking, health_table, step_changes_per_day

MARGIN = 2  # windows around a fault left out of the normal set
MIN_BLOCK = 4  # shortest bootstrap block, windows
EXPECT_MODALITY = {"cpu": "compute", "mem": "memory", "disk": "storage", "net": "network",
                   "netem": "network", "leak": "memory"}
EXPECT_SAT = {"cpu": "cpu_pressure", "mem": "mem_pressure", "disk": "io_pressure"}
EXPECT_EVENT = {"oom": "oom_kills"}
# A capped package under load runs slower than the same load uncapped; its
# clearest trace is the clock ratio. Loss shows as retransmits, a leak as used memory.
EXPECT_FEATURE = {"power": "freq_ratio", "netem": "tcp_retrans", "leak": "mem_used"}


# -- data ----------------------------------------------------------------
def load_columns(directory: str, name: str, lo: float = -math.inf, hi: float = math.inf):
    files = spool_files(os.path.join(directory, "nodes", name, "spool"), name)
    if not files:
        raise FileNotFoundError(f"no spool files for {name} under {directory}/nodes/{name}/spool")
    meta, cols = read_columns(files)
    return meta, between(cols, lo, hi)


def between(cols, lo: float, hi: float):
    return select(cols, (cols["t"] >= lo) & (cols["t"] < hi))


def commission_from_spool(directory: str, name: str, lo: float, hi: float, alpha: float,
                          options: Options = Options()) -> NodeMemory:
    meta, cols = load_columns(directory, name, lo, hi)
    return NodeMemory.commission(cols, meta, alpha=alpha, options=options)


def replay(node: NodeMemory, cols) -> list[dict]:
    """Feed columns through a copy of the node exactly as the daemon does after
    loading it (same thermal state and on-delay history; both reset themselves
    across a gap). The original stays as commissioned."""
    n = copy.deepcopy(node)
    return n._ingest(windows_from_columns(cols, n.K))


def load_faults(directory: str) -> list[dict]:
    path = os.path.join(directory, "faults.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def touches(f: dict, name: str) -> str | None:
    return "target" if f["node"] == name else "peer" if f.get("peer") == name else None


def near_fault(t0: float, faults: list[dict], name: str, K: int) -> bool:
    return any(touches(f, name) and t0 + K > f["start"] - MARGIN * K and t0 < f["end"] + MARGIN * K
               for f in faults)


def in_fault(t0: float, f: dict, K: int) -> bool:
    """Window [t0, t0+K) overlaps the fault, or is the window right after it
    (the on-delay alarms one window late)."""
    return t0 + K > f["start"] and t0 < f["end"] + K


def fired(r: dict) -> bool:
    return bool(r["alarm"] or r.get("saturation"))


def _loads(line: str) -> dict | None:
    try:
        return json.loads(line)
    except json.JSONDecodeError:  # a line cut by a crash or by rsync mid-write
        return None


# -- statistics -------------------------------------------------------------
def episode_block(x: np.ndarray, merge: int = 2) -> int:
    """Bootstrap block length that holds whole alarm episodes: twice the 95th
    percentile length of the runs of 1s (gaps of up to `merge` windows bridged),
    at least MIN_BLOCK."""
    ones = np.flatnonzero(x > 0)
    if len(ones) == 0:
        return MIN_BLOCK
    cut = np.flatnonzero(np.diff(ones) > merge + 1)
    lengths = np.r_[ones[cut], ones[-1]] - np.r_[ones[0], ones[cut + 1]] + 1
    return max(MIN_BLOCK, int(2 * np.ceil(np.quantile(lengths, 0.95))))


def block_ci(x, block: int | None = None, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean with a moving-block bootstrap 95 % interval (serially dependent x).
    Blocks are `block` long (default: episode_block) but at least 20 per series.
    A constant 0/1 series has no bootstrap spread; its bound is the rule of three
    over n/b blocks (a block holding any 1 holds at least one of its b windows)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    b = max(1, min(block or episode_block(x), n // 20))
    m = float(x.mean())
    if np.all(x == x[0]) and x[0] in (0.0, 1.0):
        r3 = min(1.0, 3.0 * b / n)
        return (m, 0.0, r3) if x[0] == 0.0 else (m, 1.0 - r3, 1.0)
    c = np.concatenate([[0.0], np.cumsum(x)])
    bsum = c[b:] - c[:-b]  # sum of every block of b consecutive windows
    k = int(math.ceil(n / b))
    starts = np.random.default_rng(seed).integers(0, n - b + 1, size=(n_boot, k))
    means = bsum[starts].sum(axis=1) / (k * b)
    return m, float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _fmt(ci) -> str:
    m, lo, hi = ci
    return "n/a" if not np.isfinite(m) else f"{m:.4f} [{lo:.4f}, {hi:.4f}]"


# -- sections ---------------------------------------------------------------
def calibration(recs: list[dict], faults: list[dict], name: str, K: int, alpha: float) -> dict:
    normal = [r for r in recs if not near_fault(r["t0"], faults, name, K)]
    p = np.array([r["p_latent"] for r in normal], dtype=np.float64)
    pf = p[np.isfinite(p)]
    ks = float(np.max(np.abs(np.sort(pf) - np.arange(1, len(pf) + 1) / len(pf)))) if len(pf) else float("nan")
    return {
        "normal_windows": len(normal),
        "latent": block_ci([r["alarm_latent"] for r in normal]),
        "event": block_ci([r["alarm_event"] for r in normal]),
        "saturation": block_ci([bool(r["saturation"]) for r in normal]),
        "any": block_ci([fired(r) for r in normal]),
        "p_le": {a: block_ci(np.where(np.isfinite(p), p <= a, np.nan)) for a in (alpha / 2, 0.05, 0.1)},
        "ks_d": ks,
        "episodes_per_week": alarm_episodes(normal, K, fired),
        "step_changes_per_day": step_changes_per_day(recs, K),
    }


def detection(recs_by_node: dict[str, list[dict]], faults: list[dict], K: int) -> list[dict]:
    rows = []
    for f in faults:
        if f.get("status") != "done":
            continue
        for name, recs in recs_by_node.items():
            role = touches(f, name)
            if role is None:
                continue
            rs = [r for r in recs if in_fault(r["t0"], f, K)]
            hit = next((r for r in rs if fired(r)), None)
            lat = next((r for r in rs if r["alarm_latent"]), None)
            mods = (lat or {}).get("modality_d2") or {}
            ftype = f["type"]
            rows.append({
                "id": f["id"], "type": ftype, "level": f["level"], "node": name, "role": role,
                "windows": len(rs), "detected": hit is not None,
                "latent": lat is not None, "event": any(r["alarm_event"] for r in rs),
                "saturation": sorted({k for r in rs for k in r["saturation"]}),
                "delay_s": (hit["t0"] + K - f["start"]) if hit else None,
                "top_modality": max(mods, key=mods.get) if mods else None,
                "attribution_ok": (max(mods, key=mods.get) == EXPECT_MODALITY[ftype]) if mods and ftype in
                EXPECT_MODALITY else None,
                "saturation_ok": (EXPECT_SAT[ftype] in {k for r in rs for k in r["saturation"]})
                if ftype in EXPECT_SAT else None,
                "event_ok": any(EXPECT_EVENT[ftype] in r["events"] for r in rs) if ftype in EXPECT_EVENT else None,
                "feature_ok": (EXPECT_FEATURE[ftype] in [n for n, _ in lat.get("top_features", [])])
                if lat and ftype in EXPECT_FEATURE else None,
            })
    return rows


def consistency(online: list[dict], recs: list[dict]) -> dict:
    by_t = {r["t0"]: r for r in recs}
    pairs = [(o, by_t[o["t0"]]) for o in online if o["t0"] in by_t]
    if not pairs:
        return {"matched": 0}
    same = [o["alarm"] == r["alarm"] and np.isclose(o["score"], r["score"], rtol=1e-6, equal_nan=True)
            for o, r in pairs]
    return {"matched": len(pairs), "online_windows": len(online), "identical": float(np.mean(same))}


# -- report -----------------------------------------------------------------
def _detection_table(rows: list[dict]) -> list[str]:
    out = ["| fault | level | role | n | detected [95 % CI] | latent | event | saturation flag ok | "
           "D2 modality ok | expected feature in top 3 | median delay s |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    keys = sorted({(r["type"], r["level"], r["role"]) for r in rows}, key=str)
    for ftype, level, role in keys:
        g = [r for r in rows if (r["type"], r["level"], r["role"]) == (ftype, level, role)]
        k = sum(r["detected"] for r in g)
        p, lo, hi = wilson(k, len(g))
        rate = lambda key: (f"{np.mean([r[key] for r in g if r[key] is not None]):.2f}"
                            if any(r[key] is not None for r in g) else "-")
        delays = [r["delay_s"] for r in g if r["delay_s"] is not None]
        delay = f"{np.median(delays):.0f}" if delays else "-"
        out.append(f"| {ftype} | {level} | {role} | {len(g)} | {p:.2f} [{lo:.2f}, {hi:.2f}] | "
                   f"{np.mean([r['latent'] for r in g]):.2f} | {np.mean([r['event'] for r in g]):.2f} | "
                   f"{rate('saturation_ok')} | {rate('attribution_ok')} | {rate('feature_ok')} | {delay} |")
    return out


def _calibration_table(cal: dict) -> list[str]:
    out = ["| node | normal windows | latent | event | saturation | any | p <= alpha/2 | p <= 0.05 | KS D | "
           "alarm episodes / week | step changes / day |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, c in cal.items():
        if "error" in c:
            out.append(f"| {name} | {c['error']} |" + " |" * 9)
            continue
        pa = list(c["p_le"].values())
        out.append(f"| {name} | {c['normal_windows']} | {_fmt(c['latent'])} | {_fmt(c['event'])} | "
                   f"{_fmt(c['saturation'])} | {_fmt(c['any'])} | {_fmt(pa[0])} | {_fmt(pa[1])} | "
                   f"{c['ks_d']:.3f} | {c['episodes_per_week']:.2f} | {c['step_changes_per_day']:.2f} |")
    return out


def clean_check(meta: dict, cols, camp: dict, K: int) -> dict:
    """False alarms with no fault anywhere: commission offline on the first
    days of the recording week, replay the rest of it."""
    try:
        node = NodeMemory.commission(between(cols, camp["start"], camp["clean_split"]), meta,
                                     window=K, alpha=camp["alpha"])
    except ValueError as e:
        return {"error": str(e)}
    recs = replay(node, between(cols, camp["clean_split"], camp["commission"]))
    return calibration(recs, [], "", K, camp["alpha"])


def write_report(directory: str, variants: tuple[str, ...] | None = None) -> str:
    from .validate_criteria import criteria, evaluate, verdict_table
    from .validate_variants import run_variants, variant_summary, variant_table
    with open(os.path.join(directory, "campaign.json")) as f:
        camp = json.load(f)
    faults, alpha = load_faults(directory), camp["alpha"]
    recs_by_node, cal, clean, cons, var_rows, K, skipped = {}, {}, {}, {}, {}, 32, {}
    for name in camp["nodes"]:
        dep = (camp.get("commissioned") or {}).get(name, {})
        model = os.path.join(directory, "models", f"model-{name}.pkl")
        if "error" in dep or not os.path.exists(model):
            skipped[name] = dep.get("error", "no model")
            continue
        node = NodeMemory.load(model)
        t_dep = dep.get("deployed", camp["commission"])
        meta, cols = load_columns(directory, name)
        test, comm = between(cols, t_dep, camp["end"]), between(cols, camp["start"], camp["commission"])
        clean[name] = clean_check(meta, cols, camp, node.K)
        del cols
        recs, K = replay(node, test), node.K
        recs_by_node[name] = recs
        cal[name] = calibration(recs, faults, name, node.K, alpha)
        ev_path = os.path.join(directory, "nodes", name, "events.jsonl")
        if os.path.exists(ev_path):
            with open(ev_path) as f:
                online = [r for r in map(_loads, f) if r and t_dep <= r["t0"] < camp["end"]]
            cons[name] = consistency(online, recs)
        var_rows[name] = run_variants(node, meta, comm, test, recs, alpha, variants)
    det = detection(recs_by_node, faults, K)
    summary = variant_summary(var_rows, faults, K)
    if "criteria" in camp:
        crit = camp["criteria"]
        registered = f"registered {time.strftime('%Y-%m-%d %H:%M', time.localtime(camp['criteria_registered']))}"
    else:  # campaigns planned before criteria were stored: evaluated, but not pre-registered
        crit, registered = criteria(alpha), "NOT pre-registered (campaign planned before criteria were stored)"
    verdicts = evaluate(crit, camp["levels"], clean, det, cons, summary)
    fmt = lambda t: time.strftime("%Y-%m-%d %H:%M", time.localtime(t))
    done = sum(f.get("status") == "done" for f in camp["schedule"])
    lines = [f"# uengram-host validation report ({fmt(time.time())})", "",
             f"Recording {fmt(camp['start'])}, commissioning cut {fmt(camp['commission'])}, test until "
             f"{fmt(camp['end'])}. Faults: {done} executed of {len(camp['schedule'])} scheduled. "
             f"alpha = {alpha} (latent alpha/2; event counters alpha/2, Bonferroni over the counters).", "",
             *([f"Nodes left out: {skipped}.", ""] if skipped else []),
             f"## 0. Pre-registered verdicts ({registered})", "", *verdict_table(verdicts), "",
             "## 1. Calibration", "",
             "Pass: the latent and event rates are each at most alpha/2 inside the CI (saturation "
             "flags have their own rank bound); the share of p <= a is close to a.", "",
             f"### 1a. Clean week: commissioned offline on {fmt(camp['start'])} to {fmt(camp['clean_split'])}, "
             f"replayed to {fmt(camp['commission'])}; nothing injected", "", *_calibration_table(clean), "",
             "### 1b. Fault week, deployed model, windows more than 2 windows from any fault on the node", "",
             *_calibration_table(cal)]
    lines += ["", "## 2. Detection of injected faults", "",
              "Pass: detection rises with strength; the saturation flag names the resource for "
              "cpu/mem/disk; OOM is caught by the event channel.", ""] + _detection_table(det)
    lines += ["", "## 3. Online daemon vs offline replay", "", "| node | matched windows | identical |", "|---|---|---|"]
    for name, c in cons.items():
        lines.append(f"| {name} | {c['matched']} | {c.get('identical', float('nan')):.3f} |")
    lines += ["", "## 4. Ablations and baselines", "",
              "Pass for a component: removing it lowers detection or raises false alarms.", ""]
    lines += variant_table(summary)
    health = health_ranking(recs_by_node, faults, K, near_fault, in_fault)
    lines += ["", "## 5. Health score as a ranking signal (reported, not alarmed)", "",
              "AUROC of fault windows against normal windows, per node, averaged over nodes. Health is the D2 "
              "to the frozen healthy reference (uEngram v3); the alarm score is the persistent novelty score.",
              ""] + health_table(health)
    path = os.path.join(directory, "report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(directory, "report.json"), "w") as f:
        json.dump({"verdicts": verdicts, "criteria": crit, "clean_week": clean, "calibration": cal,
                   "detection": det, "consistency": cons, "variants": summary, "health": health,
                   "skipped": skipped}, f, indent=1, default=str)
    return path
