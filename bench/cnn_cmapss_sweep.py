"""CNN encoder configurations on CMAPSS, uEngram method, one (subset, seed) per process.

    python -m bench.cnn_cmapss_sweep SUBSET raw|norm SEED NAME...
NAME is d<dim>e<epochs><avg|maxavg>[_b<ortho_beta>], e.g. d16e90maxavg_b0.05. One JSON line per NAME."""
from __future__ import annotations

import json
import re
import sys
import warnings

import numpy as np

import bench.run_cmapss as rc
from uengram import cmapss
from uengram.encoder_cnn import CNNEncoder

KEYS = ("eol_retention", "auroc", "health_auroc", "rul_rmse_health", "rul_rmse_health_age",
        "write_reduction_pct", "items_end")


def make_encoder(name: str, seed: int) -> CNNEncoder:
    m = re.fullmatch(r"d(\d+)e(\d+)(avg|maxavg)(?:_b([\d.]+))?", name)
    if m is None:
        raise SystemExit(f"bad configuration name {name!r}")
    return CNNEncoder(seed=seed, d=int(m[1]), epochs=int(m[2]), pool=m[3], ortho_beta=float(m[4] or 0.0))


def main() -> None:
    subset, norm, seed, names = sys.argv[1], sys.argv[2] == "norm", int(sys.argv[3]), sys.argv[4:]
    warnings.simplefilter("ignore", RuntimeWarning)
    rc.make_encoder = make_encoder
    cmapss.fetch(subset=subset)
    n_units = int(np.unique(cmapss.load(f"data/cmapss/train_{subset}.txt")[:, 0]).size)
    for name in names:
        r = rc.run_one((seed, name, ("uengram",), n_units, subset, norm))[0]
        print(json.dumps({"subset": subset, "condition_norm": norm, "config": name, "seed": seed,
                          "step_inflation_quality": r.get("step_inflation_quality", "exact"),
                          **{k: round(float(r["metrics"][k]), 4) for k in KEYS}}), flush=True)


if __name__ == "__main__":
    main()
