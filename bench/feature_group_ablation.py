"""DSP feature-group ablation: drop one of {mean, log std, band energies} at a time.

python -m bench.feature_group_ablation --out results/feature_group_ablation.json

The DSP encoder builds 6 features per channel: mean (column 0), log std (1) and four log
band energies (2-5). ``encoders.band_features`` is patched to return only the kept columns;
everything else (robust scaling, PCA to d=16, noise whitening, alarm policy) is unchanged.
Runs on the synthetic stream, SKAB (seed 0, frozen uengram policy) and CMAPSS FD001/FD003
(uengram). Everything runs in this process: ``run_synth.main`` uses a spawn Pool whose
workers would ignore the patch. A configuration that raises is recorded as an error, not
dropped.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np

import bench.run_cmapss as rc
import bench.run_synth as rs
from bench.alarm_eval import apply_policy
from bench.run_v2 import FROZEN
from bench.stream_skab import skab_stream
from bench.v2_scoring import score_stream
from uengram import cmapss, encoders

_orig = encoders.band_features
GROUPS = {"all": (0, 1, 2, 3, 4, 5), "no_mean": (1, 2, 3, 4, 5), "no_log_std": (0, 2, 3, 4, 5),
          "no_band_energy": (0, 1)}
SYNTH_METRICS = ("event_recall", "write_reduction_pct", "ap")
CMAPSS_METRICS = ("health_auroc", "eol_retention", "rul_rmse", "write_reduction_pct")


def _patched(keep: tuple[int, ...]):
    def band_features(windows):
        full = _orig(windows)
        return full.reshape(len(full), -1, 6)[:, :, list(keep)].reshape(len(full), -1)
    return band_features


def _synthetic(seeds: list[int]) -> dict:
    rows = [line for seed in seeds
            for line in rs.run_one((seed, "dsp", 20000, ["uengram"], [256], "fg", 0.01, 4000))]
    return {key: st.mean(row["metrics"][key] for row in rows) for key in SYNTH_METRICS}


def _skab() -> dict:
    stream = skab_stream(None, K=10)
    scores = score_stream(stream, "uengram", "dsp", 0)
    metrics, details, _ = apply_policy(stream, scores, FROZEN["uengram"])
    return {"events_detected": float(metrics["events_detected"]), "n_events": len(stream.events),
            "false_alarm_episodes": int(details["false_alarm_episodes"]),
            "auroc": float(metrics["auroc"]), "ap": float(metrics["ap"])}


def _cmapss(subset: str) -> dict:
    cmapss.fetch(subset=subset)
    n_units = int(np.unique(cmapss.load(f"data/cmapss/train_{subset}.txt")[:, 0]).size)
    rows = rc.run_one((0, "dsp", ("uengram",), n_units, subset, False))
    return {key: float(rows[0]["metrics"][key]) for key in CMAPSS_METRICS}


def _guarded(fn, *args) -> dict:
    try:
        return fn(*args)
    except Exception as exc:  # recorded in the artifact; one failing configuration must not hide the rest
        return {"error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--out", default="results/feature_group_ablation.json")
    args = parser.parse_args()
    lo, _, hi = args.seeds.partition("-")
    seeds = list(range(int(lo), int(hi) + 1)) if hi else [int(lo)]

    out = {"seeds": args.seeds, "groups": {k: list(v) for k, v in GROUPS.items()}, "results": {}}
    try:
        for label, keep in GROUPS.items():
            encoders.band_features = _patched(keep)
            out["results"][label] = {
                "synthetic": _guarded(_synthetic, seeds), "skab_seed0": _guarded(_skab),
                "cmapss_fd001": _guarded(_cmapss, "FD001"), "cmapss_fd003": _guarded(_cmapss, "FD003")}
            print(label, json.dumps(out["results"][label]), flush=True)
    finally:
        encoders.band_features = _orig
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
