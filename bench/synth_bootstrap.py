"""Percentile-bootstrap intervals over seeds for the synthetic-stream results.

python -m bench.synth_bootstrap --out results/synth_bootstrap_ci.json

Resamples the 10 seeds with replacement (10,000 draws, fixed RNG seed) and reports the
95% interval of each method's mean, plus the paired d=16 -> d=36 change of the mean (the
two result files share seeds 0-9, so the same resampled seeds are used for both).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KEYS = ("event_recall", "write_reduction_pct", "ap")
DRAWS = 10_000


def _load(path: str, encoder: str) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row["encoder"] == encoder:
            out.setdefault(row["method"], {})[row["seed"]] = row["metrics"]
    return out


def _interval(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d16", default="results/synth_runs.jsonl")
    parser.add_argument("--d36", default="results/synth_d36_runs.jsonl")
    parser.add_argument("--out", default="results/synth_bootstrap_ci.json")
    parser.add_argument("--rng", type=int, default=20261029)
    args = parser.parse_args()

    runs = {"d16": _load(args.d16, "dsp"), "d36": _load(args.d36, "dsp_d36")}
    rng = np.random.default_rng(args.rng)
    out: dict = {"draws": DRAWS, "rng_seed": args.rng, "level": 0.95, "means": {}, "paired_d36_minus_d16": {}}
    for width, methods in runs.items():
        for method, by_seed in methods.items():
            seeds = sorted(by_seed)
            idx = rng.integers(0, len(seeds), size=(DRAWS, len(seeds)))
            entry = {"n_seeds": len(seeds)}
            for key in KEYS:
                vals = np.array([by_seed[s][key] for s in seeds], dtype=float)
                entry[key] = {"mean": float(vals.mean()), "ci95": _interval(vals[idx].mean(axis=1))}
            out["means"][f"{width}/{method}"] = entry
    for method in runs["d16"]:
        if method not in runs["d36"]:
            continue
        seeds = sorted(set(runs["d16"][method]) & set(runs["d36"][method]))
        idx = rng.integers(0, len(seeds), size=(DRAWS, len(seeds)))
        entry = {"n_seeds": len(seeds)}
        for key in KEYS:
            diff = np.array([runs["d36"][method][s][key] - runs["d16"][method][s][key] for s in seeds])
            entry[key] = {"mean": float(diff.mean()), "ci95": _interval(diff[idx].mean(axis=1))}
        out["paired_d36_minus_d16"][method] = entry
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True, allow_nan=False) + "\n")
    for name in ("d16/uengram", "d36/uengram"):
        e = out["means"][name]
        print(name, {k: (round(e[k]["mean"], 4), [round(x, 4) for x in e[k]["ci95"]]) for k in KEYS})
    e = out["paired_d36_minus_d16"]["uengram"]
    print("paired uengram", {k: (round(e[k]["mean"], 4), [round(x, 4) for x in e[k]["ci95"]]) for k in KEYS})


if __name__ == "__main__":
    main()
