"""Host pipeline on synthetic nodes: false alarms, detection, delay, attribution.

Per seed: commission on 4 h of synthetic ticks (uengram.host.synth), then a
2 h test stream with four injected faults at random onsets and durations:
memory thrash (memory), a memory-bound spin loop (compute), a cooling fault
(+12 C at the same power, power_thermal) and one OOM kill (count model).

  python -m bench.run_host_synth --seeds 0-19 --jobs 2 --out results/host_synth.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from bench.common import parse_seeds
from uengram.host.node import NodeMemory
from uengram.host.synth import make_ticks
from uengram.metrics import bootstrap_ci

K = 32
EXPECT = {"thrash": "memory", "spin": "compute", "hot": "power_thermal", "oom": "oom_kills"}


def _faults(rng, n_test: int) -> dict:
    slots = np.sort(rng.choice(np.arange(600, n_test - 900, 900), size=4, replace=False))
    names = rng.permutation(list(EXPECT))
    out = {}
    for name, start in zip(names, slots):
        start = int(start + rng.integers(0, 300))
        dur = 0 if name == "oom" else int(rng.integers(130, 400))
        out[str(name)] = (start, start + dur)
    return out


def run_seed(seed: int) -> dict:
    rng = np.random.default_rng(1000 + seed)
    t0 = time.time()
    ticks = make_ticks(4 * 3600, seed=seed)
    node = NodeMemory.commission(ticks, {"host": f"synth{seed}"})
    t_fit = time.time() - t0
    start = ticks[-1]["t"] + 1
    start += (K - start % K) % K
    n_test = 2 * 3600
    faults = _faults(rng, n_test)
    test = make_ticks(n_test, seed=10_000 + seed, t_start=start + 0.5, anomalies=faults)
    recs = list(node.feed_ticks(test))
    for r in recs:
        r["k"] = r["t0"] - test[0]["t"]
    row = {"seed": seed, "fit_s": t_fit, "threshold": node.alarm_threshold,
           "thermal": node.enc.fit_report["thermal"], "faults": faults}
    near = lambda r: any(lo - 2 * K <= r["k"] <= hi + 2 * K for lo, hi in faults.values())
    normal = [r for r in recs if not near(r)]
    row["false_alarm_rate"] = float(np.mean([r["alarm"] for r in normal]))
    for name, (lo, hi) in faults.items():
        rs = [r for r in recs if lo - K < r["k"] <= hi + K]
        key = "alarm_event" if name == "oom" else "alarm_latent"
        hit = next((r for r in rs if r[key]), None)
        row[f"{name}_detected"] = float(hit is not None)
        row[f"{name}_delay_s"] = float(hit["k"] + K - lo) if hit else float("nan")
        if hit is None:
            row[f"{name}_attribution_ok"] = float("nan")
        elif name == "oom":
            row[f"{name}_attribution_ok"] = float(EXPECT[name] in hit["events"])
        else:
            m = hit.get("modality_d2", {})
            row[f"{name}_attribution_ok"] = float(bool(m) and max(m, key=m.get) == EXPECT[name])
    return row


def summarize(rows: list[dict]) -> str:
    keys = ["false_alarm_rate"] + [f"{n}_{m}" for n in EXPECT for m in ("detected", "delay_s", "attribution_ok")]
    lines = [f"seeds: {len(rows)}", "", "| metric | mean [95% CI] |", "|---|---|"]
    for k in keys:
        m, lo, hi = bootstrap_ci([r[k] for r in rows])
        lines.append(f"| {k} | {m:.3g} [{lo:.3g}, {hi:.3g}] |")
    R = [r["thermal"].get("R_c_per_w", np.nan) for r in rows]
    lines.append(f"| thermal R (truth 0.8 C/W) | {np.nanmean(R):.3f} |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0-4")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--out", default="results/host_synth.jsonl")
    a = p.parse_args()
    seeds = parse_seeds(a.seeds)
    rows = []
    with ProcessPoolExecutor(a.jobs) as ex, open(a.out, "w") as f:
        for row in ex.map(run_seed, seeds):
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"seed {row['seed']}: false alarms {row['false_alarm_rate']:.3f}, "
                  f"detected {[row[f'{n}_detected'] for n in EXPECT]}, fit {row['fit_s']:.1f}s", flush=True)
            rows.append(row)
    print(summarize(rows))


if __name__ == "__main__":
    main()
