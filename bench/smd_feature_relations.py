"""Per-machine relation between DSP feature-group signal-to-noise and the AUROC lost when the
group is dropped (SMD, 28 machines).

python -m bench.smd_feature_relations --out results/smd_feature_relations.json

Hypothesis fixed before the run: dropping a feature group costs more AUROC on machines where that
group has higher signal-to-noise. Descriptor (label-free, fit windows only): for each of the 38
channels, the robust variance of the feature across windows divided by the noise variance from
successive-window differences (MAD, divided by sqrt(2)); the machine value is the median over
non-constant channels of log10 of that ratio (band energies: mean over the 4 bands first).
Effect: AUROC(all features) - AUROC(group dropped), from results/smd_confirmation.json.
Statistic: Spearman rank correlation over machines, two-sided permutation p-value (10,000
permutations, fixed seed). Three groups are tested, so compare p with 0.05 / 3.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench.stream_smd import smd_stream
from uengram import encoders

GROUPS = {"mean": (0,), "log_std": (1,), "band_energy": (2, 3, 4, 5)}
DROP_KEY = {"mean": "uengram_no_mean", "log_std": "uengram_no_log_std", "band_energy": "uengram_no_band_energy"}


def _log_snr(F: np.ndarray) -> float:
    """Median over non-constant channels of log10(signal variance / noise variance)."""
    q75, q25 = np.percentile(F, [75, 25], axis=0)
    signal = ((q75 - q25) / 1.349) ** 2
    delta = np.diff(F, axis=0)
    mad = np.median(np.abs(delta - np.median(delta, axis=0)), axis=0)
    noise = (1.4826 * mad) ** 2 / 2.0
    ok = (signal > 1e-12) & (noise > 1e-12)
    return float(np.median(np.log10(signal[ok] / noise[ok]))) if ok.any() else float("nan")


def descriptors(machine: str) -> dict:
    stream = smd_stream(machine)
    F = encoders.band_features(stream.W[stream.fit]).reshape(int(stream.fit.sum()), -1, 6)
    return {name: float(np.nanmean([_log_snr(F[:, :, c]) for c in cols])) for name, cols in GROUPS.items()}


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    for v in np.unique(x):  # average ranks over ties
        ranks[x == v] = ranks[x == v].mean()
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


def permutation_p(x: np.ndarray, y: np.ndarray, n: int = 10_000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    observed = abs(spearman(x, y))
    hits = sum(abs(spearman(x, rng.permutation(y))) >= observed for _ in range(n))
    return (hits + 1) / (n + 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation", default="results/smd_confirmation.json")
    parser.add_argument("--out", default="results/smd_feature_relations.json")
    args = parser.parse_args()
    per_machine = json.loads(Path(args.confirmation).read_text())["per_machine"]
    rows = {m: {"descriptor": descriptors(m),
                "auroc_loss": {g: per_machine[m]["uengram_d16"]["auroc"] - per_machine[m][k]["auroc"]
                               for g, k in DROP_KEY.items()}} for m in per_machine}
    out = {"n_machines": len(rows), "per_machine": rows, "tests": {}}
    for g in GROUPS:
        x = np.array([r["descriptor"][g] for r in rows.values()])
        y = np.array([r["auroc_loss"][g] for r in rows.values()])
        ok = np.isfinite(x) & np.isfinite(y)
        out["tests"][g] = {"n": int(ok.sum()), "spearman": spearman(x[ok], y[ok]),
                           "permutation_p": permutation_p(x[ok], y[ok]),
                           "mean_auroc_loss": float(y[ok].mean()), "median_log_snr": float(np.median(x[ok]))}
        print(g, out["tests"][g], flush=True)
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
