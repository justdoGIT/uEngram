"""Shared harness helpers: store construction, streaming loop, JSONL output."""
from __future__ import annotations

import json
import time

import numpy as np

from uengram.baseline_reference import ReferenceBaseline
from uengram.baselines import FIFOStore, NaiveStore, ReservoirStore
from uengram.encoder_slope import DSPSlopeEncoder
from uengram.engine import Engram, EngramConfig
from uengram.stats import conformal_threshold, ewma_inflation

ALL_METHODS = ("naive", "fifo", "reservoir", "reference_labelfree", "reference_oracle",
               "uengram", "uengram_fixedvar", "uengram_nopromo", "uengram_frozen",
               "uengram_nostep")
CAL_WARMUP = 200
EPISODIC_SLOT_FRACTION = 0.75


def make_store(method: str, B: int, d: int, sigma0: np.ndarray, raw_bytes: int,
               seed: int, cfg_over: dict | None = None, step_inflation: np.ndarray | None = None):
    over = dict(cfg_over or {})
    if method == "naive":
        return NaiveStore(d, raw_bytes)
    if method == "fifo":
        return FIFOStore(B, d, raw_bytes)
    if method == "reservoir":
        return ReservoirStore(B, d, raw_bytes, seed=seed)
    if method.startswith("reference_"):
        return ReferenceBaseline(B, d, raw_bytes, severity_mode=method.split("_", 1)[1])
    if method.startswith("uengram"):
        variant = {"uengram": {}, "uengram_fixedvar": {"use_slot_var": False},
                   "uengram_nopromo": {"use_promotion": False},
                   "uengram_frozen": {"beta_min": 0.0},
                   "uengram_nostep": {"use_step_detect": False}}[method]
        b_epi = max(1, int(round(B * EPISODIC_SLOT_FRACTION)))
        cfg = EngramConfig(B_sem=B - b_epi, B_epi=b_epi, **{**over, **variant})
        return Engram(cfg, d, sigma0, raw_bytes, name=method, step_inflation=step_inflation)
    raise ValueError(f"unknown method {method}")


def encode_windows(enc, windows: np.ndarray, groups=None, sequential: bool = True) -> np.ndarray:
    """`enc.encode(...)`, passing `groups` through only to encoders that use
    segment info at encode time (currently `DSPSlopeEncoder`); other
    encoders' `encode()` does not accept it. `groups` must be the same ids
    the caller fit with (CMAPSS/MetroPT/SKAB's own segment/unit ids, or
    `None` for the synthetic benchmark's single implicit segment)."""
    if isinstance(enc, DSPSlopeEncoder):
        return enc.encode(windows, sequential=sequential, segment=groups)
    return enc.encode(windows, sequential=sequential)


def measure_step_inflation(Ztrain: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
    """EWMA variance inflation of the encoder's training latents at the engine's
    step-detector lambda; uEngram scales its step detector input by it."""
    return ewma_inflation(Ztrain, EngramConfig.step_lambda, groups)


def calibrate_alarm(method: str, B: int, d: int, sigma0: np.ndarray, raw_bytes: int,
                    seed: int, Zcal: np.ndarray, alpha: float, cfg_over: dict | None = None,
                    step_inflation: np.ndarray | None = None) -> float:
    """Conformal alarm threshold for a uEngram variant: run a fresh engine over a
    fault-free commissioning stream and take the split-conformal (1-alpha)
    quantile of its normal-state scores after a warm-up."""
    store = make_store(method, B, d, sigma0, raw_bytes, seed, cfg_over, step_inflation)
    scores = np.array([store.ingest(z, t, t).score for t, z in enumerate(Zcal)])
    return conformal_threshold(scores[CAL_WARMUP:], alpha)


def run_stream(store, Z: np.ndarray, record, oracle_sev: np.ndarray | None = None):
    """Feed Z through the store. `record(t, res)` receives each IngestResult after
    the ingest; harness bookkeeping lives there. Returns (scores, flags|None, lat_ns)."""
    T = Z.shape[0]
    scores = np.empty(T)
    flags: list = []
    lat = np.empty(T, dtype=np.int64)
    for t in range(T):
        kw = {"severity": float(oracle_sev[t])} if oracle_sev is not None else {}
        t0 = time.perf_counter_ns()
        res = store.ingest(Z[t], t, t, **kw)
        lat[t] = time.perf_counter_ns() - t0
        scores[t] = res.score
        flags.append(res.flagged)
        record(t, res)
    fl = None if flags[0] is None else np.array(flags, dtype=bool)
    return scores, fl, lat


def _json_default(value):
    """Convert NumPy values emitted by benchmark detail records."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.datetime64):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def dump(path: str, lines: list[dict], mode: str = "a") -> None:
    with open(path, mode) as f:
        for ln in lines:
            f.write(json.dumps(ln, default=_json_default) + "\n")


def parse_seeds(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]
