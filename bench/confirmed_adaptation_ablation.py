"""Operator-confirmed latent adaptation: frozen vs gated vs ungated, on benign drift and wear ramps.

python -m bench.confirmed_adaptation_ablation --out results/confirmed_adaptation_ablation.json

The stream is generated directly in an 8-dimensional unit-variance latent space and fed through
``SensorStateMemory`` (the live integration path) with an in-memory null event store. Per seed:
1500 healthy commissioning windows, a frozen healthy reference, 1000 healthy calibration windows
(conformal thresholds, alpha = 0.003, for the novelty and the health score), then 6000 test windows.

Scenarios
  drift     the healthy mean drifts linearly by 4 sigma over the test stream (benign: a sensor or
            ambient change the operator would confirm as normal), plus 5 slow wear ramps of 300
            windows reaching 8 sigma in an independent direction (never confirmed).
  no_drift  the same ramps with no benign drift (does adaptation cost anything when nothing drifts?).

Arms
  frozen    no adapter.
  gated     ConfirmedLatentAdapter; 25% of healthy windows carry operator_confirmed_normal.
  ungated   ConfirmedLatentAdapter; every window carries operator_confirmed_normal, ramps included
            (the adapter still refuses windows during an open novelty alarm). This is the
            poisoning control: what happens if confirmation is not a real gate.

Labels: a ramp window is a fault once the ramp reaches half its peak (4 sigma); the first half of
each ramp is excluded from both classes. Reported per arm over the 10 seeds: false-alarm rate on
healthy windows and recall on fault windows for each score (novelty, health), health AUROC, and
the number of the 5 ramps with at least one health alarm. Runs in a spawn pool; results do not
depend on --jobs.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from uengram import SensorStateMemory, SensorWindow
from uengram.confirmed_adaptation import ConfirmedLatentAdapter
from uengram.engine import Engram, EngramConfig
from uengram.metrics import auroc
from uengram.stats import conformal_threshold

D, N_COMMISSION, N_CALIBRATION, N_TEST = 8, 1500, 1000, 6000
DRIFT_SIGMA, RAMP_LEN, RAMP_PEAK, N_RAMPS = 4.0, 300, 8.0, 5
CONFIRM_FRACTION, ALPHA, MAX_OFFSET = 0.25, 0.003, 6.0
ARMS = ("frozen", "gated", "ungated")
SCENARIOS = ("drift", "no_drift")


class NullMemory:
    """EdgeMemory port that stores nothing; the benchmark reads decisions, not records."""

    def __init__(self):
        self.count = 0

    def remember(self, content, **kwargs):
        self.count += 1
        return SimpleNamespace(memory_id=self.count)

    def recall(self, **kwargs):
        return SimpleNamespace(items=())

    def forget(self, memory_id):
        return 0


def _unit(rng) -> np.ndarray:
    v = rng.normal(size=D)
    return v / np.linalg.norm(v)


def _window(step: int, features, confirmed: bool) -> SensorWindow:
    return SensorWindow(
        asset_id="asset", subsystem_id="sub", operating_mode="mode", step_index=step,
        observed_at_ms=1_800_000_000_000 + step * 1000, raw_idx=step,
        features=np.asarray(features, dtype=float), feature_schema="latent-v1",
        operator_confirmed_normal=confirmed,
    )


def _stream(rng, scenario: str):
    """Test latents, per-window healthy/fault/ignored class and ramp id."""
    u_drift, u_ramp = _unit(rng), _unit(rng)
    t = np.arange(N_TEST)
    mean = (DRIFT_SIGMA * t / N_TEST)[:, None] * u_drift if scenario == "drift" else np.zeros((N_TEST, D))
    starts = np.sort(rng.choice(np.arange(200, N_TEST - RAMP_LEN, RAMP_LEN + 100), N_RAMPS, replace=False))
    ramp = np.zeros(N_TEST)
    ramp_id = np.full(N_TEST, -1)
    for i, s in enumerate(starts):
        ramp[s:s + RAMP_LEN] = RAMP_PEAK * np.arange(RAMP_LEN) / RAMP_LEN
        ramp_id[s:s + RAMP_LEN] = i
    z = rng.normal(size=(N_TEST, D)) + mean + ramp[:, None] * u_ramp
    fault = ramp >= RAMP_PEAK / 2
    healthy = ramp_id == -1
    return z, healthy, fault, ramp_id


def _run(seed: int, scenario: str, arm: str) -> dict:
    rng = np.random.default_rng([seed, SCENARIOS.index(scenario)])
    adapter = None if arm == "frozen" else ConfirmedLatentAdapter(D, max_offset=MAX_OFFSET)
    engine = Engram(EngramConfig(B_sem=48, B_epi=16), D, np.ones(D), raw_window_bytes=D * 16 * 4)
    pipeline = SensorStateMemory(engine, NullMemory(), adapter=adapter)
    step = 0
    for z in rng.normal(size=(N_COMMISSION, D)):
        pipeline.ingest(_window(step, z, False))
        step += 1
    engine.set_reference(*engine.reference())
    novelty, health = [], []
    for z in rng.normal(size=(N_CALIBRATION, D)):
        d = pipeline.ingest(_window(step, z, False)).decision
        novelty.append(d.score)
        health.append(d.health)
        step += 1
    thr_n = conformal_threshold(np.array(novelty), ALPHA)
    thr_h = conformal_threshold(np.array(health), ALPHA)
    engine.alarm_tau = thr_n

    z, healthy, fault, ramp_id = _stream(rng, scenario)
    confirm = rng.random(N_TEST) < CONFIRM_FRACTION
    nov, hlt = np.full(N_TEST, np.nan), np.full(N_TEST, np.nan)
    for i in range(N_TEST):
        confirmed = bool(confirm[i] and healthy[i]) if arm == "gated" else arm == "ungated"
        d = pipeline.ingest(_window(step, z[i], confirmed)).decision
        nov[i], hlt[i] = d.score, d.health
        step += 1
    keep = healthy | fault
    ramps_hit = sum(bool((hlt[(ramp_id == r) & fault] > thr_h).any()) for r in range(N_RAMPS))
    return {
        "seed": seed, "scenario": scenario, "arm": arm,
        "novelty_false_alarm_rate": float((nov[healthy] > thr_n).mean()),
        "novelty_fault_recall": float((nov[fault] > thr_n).mean()),
        "health_false_alarm_rate": float((hlt[healthy] > thr_h).mean()),
        "health_fault_recall": float((hlt[fault] > thr_h).mean()),
        "health_auroc": float(auroc(hlt[keep], fault[keep])),
        "ramps_with_health_alarm": int(ramps_hit),
        "final_offset_norm": 0.0 if adapter is None else float(
            max((np.linalg.norm(v) for v in adapter.state.values()), default=0.0)),
    }


def _task(args):
    os.environ["OMP_NUM_THREADS"] = "1"
    return _run(*args)


def _summarize(rows: list[dict]) -> dict:
    out = {}
    for key in rows[0]:
        if key in ("seed", "scenario", "arm"):
            continue
        v = np.array([r[key] for r in rows], dtype=float)
        out[key] = {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--out", default="results/confirmed_adaptation_ablation.json")
    args = parser.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    tasks = [(s, sc, arm) for sc in SCENARIOS for arm in ARMS for s in range(args.seeds)]
    with mp.get_context("spawn").Pool(args.jobs) as pool:
        rows = pool.map(_task, tasks)
    summary = {sc: {arm: _summarize([r for r in rows if r["scenario"] == sc and r["arm"] == arm])
                    for arm in ARMS} for sc in SCENARIOS}
    out = {"seeds": args.seeds, "max_offset": MAX_OFFSET, "confirm_fraction": CONFIRM_FRACTION,
           "summary": summary, "runs": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    for sc in SCENARIOS:
        for arm in ARMS:
            s = summary[sc][arm]
            print(f"{sc:9s} {arm:8s} health FA {s['health_false_alarm_rate']['mean']:.3f} "
                  f"recall {s['health_fault_recall']['mean']:.3f} AUROC {s['health_auroc']['mean']:.3f} "
                  f"ramps {s['ramps_with_health_alarm']['mean']:.1f}/{N_RAMPS} | novelty FA "
                  f"{s['novelty_false_alarm_rate']['mean']:.3f} recall {s['novelty_fault_recall']['mean']:.3f}",
                  flush=True)


if __name__ == "__main__":
    main()
