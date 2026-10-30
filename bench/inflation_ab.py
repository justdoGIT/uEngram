"""A/B of the step-inflation minimum-support rule on CMAPSS (DSP encoders, uEngram).

Arm `strict` (min_pairs=0) keeps the jump-segmented estimate whenever it exists; arm `rule` (the
default MIN_INFLATION_PAIRS) falls back to the caller's groups when that estimate has too few warm
lag pairs. DSP encoders and the CMAPSS stream are deterministic, so one seed is the whole result.
    python -m bench.inflation_ab [--out results/inflation_min_support_ab.json]"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np

import bench.run_cmapss as rc
from uengram import cmapss
from uengram.engine import EngramConfig
from uengram.noise_fallback import MIN_INFLATION_PAIRS, soft_ewma_inflation

CASES = [(s, e, n) for s in cmapss.SUBSETS for e in ("dsp", "dsp_d24", "dsp_slope") for n in (False,)]
CASES += [("FD002", "dsp", True), ("FD004", "dsp", True)]
KEYS = ("eol_retention", "auroc", "health_auroc", "rul_rmse_health", "rul_rmse_health_age",
        "write_reduction_pct", "items_end")


def _arm(min_pairs: int):
    return lambda Z, groups=None: soft_ewma_inflation(Z, EngramConfig.step_lambda, groups, min_pairs=min_pairs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/inflation_min_support_ab.json")
    out_path = ap.parse_args().out
    warnings.simplefilter("ignore", RuntimeWarning)
    rows = []
    for subset, enc, norm in CASES:
        cmapss.fetch(subset=subset)
        n_units = int(np.unique(cmapss.load(f"data/cmapss/train_{subset}.txt")[:, 0]).size)
        for arm, min_pairs in (("strict", 0), ("rule", MIN_INFLATION_PAIRS)):
            rc.measure_step_inflation = _arm(min_pairs)
            r = rc.run_one((0, enc, ("uengram",), n_units, subset, norm))[0]
            rows.append({"subset": subset, "encoder": enc, "condition_norm": norm, "arm": arm,
                         "step_inflation_quality": r.get("step_inflation_quality", "exact"),
                         **{k: round(float(r["metrics"][k]), 4) for k in KEYS}})
            print(rows[-1], flush=True)
    with open(out_path, "w") as f:
        json.dump({"min_inflation_pairs": MIN_INFLATION_PAIRS, "seed": 0, "method": "uengram", "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
