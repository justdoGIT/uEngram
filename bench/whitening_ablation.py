"""Noise-whitening ablation: learned noise rotation vs. identity (PCA kept).

python -m bench.whitening_ablation --out results/whitening_ablation.json

Replaces ``NoiseWhitener.fit`` with an identity rotation and compares against the normal
pipeline on (a) the synthetic stream (DSP d=16, uengram, budget 256) and (b) SKAB (seed 0,
frozen uengram alarm policy). Everything runs in this process: ``run_synth.main`` uses a
spawn Pool whose workers would ignore the patch.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np

import bench.run_synth as rs
from bench.alarm_eval import apply_policy
from bench.run_v2 import FROZEN
from bench.stream_skab import skab_stream
from bench.v2_scoring import score_stream
from uengram import encoders

_orig_fit = encoders.NoiseWhitener.fit
SYNTH_METRICS = ("event_recall", "write_reduction_pct", "ap")


def _identity_fit(self, z, noise_lag, groups):
    self.W = np.eye(np.asarray(z).shape[1])
    return self


def _synthetic(identity: bool, seeds: list[int]) -> dict:
    encoders.NoiseWhitener.fit = _identity_fit if identity else _orig_fit
    try:
        rows = [line for seed in seeds
                for line in rs.run_one((seed, "dsp", 20000, ["uengram"], [256], "whiten", 0.01, 4000))]
    finally:
        encoders.NoiseWhitener.fit = _orig_fit
    return {key: st.mean(row["metrics"][key] for row in rows) for key in SYNTH_METRICS}


def _skab(identity: bool) -> dict:
    encoders.NoiseWhitener.fit = _identity_fit if identity else _orig_fit
    try:
        stream = skab_stream(None, K=10)
        scores = score_stream(stream, "uengram", "dsp", 0)
        metrics, details, _ = apply_policy(stream, scores, FROZEN["uengram"])
    finally:
        encoders.NoiseWhitener.fit = _orig_fit
    return {"events_detected": float(metrics["events_detected"]), "n_events": len(stream.events),
            "false_alarm_episodes": int(details["false_alarm_episodes"]),
            "auroc": float(metrics["auroc"]), "ap": float(metrics["ap"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--out", default="results/whitening_ablation.json")
    args = parser.parse_args()
    lo, _, hi = args.seeds.partition("-")
    seeds = list(range(int(lo), int(hi) + 1)) if hi else [int(lo)]

    out = {"seeds": args.seeds, "synthetic": {}, "skab_seed0": {}}
    for identity, label in ((False, "whitening_on"), (True, "whitening_off")):
        out["synthetic"][label] = _synthetic(identity, seeds)
        out["skab_seed0"][label] = _skab(identity)
        print(label, out["synthetic"][label], out["skab_seed0"][label], flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
