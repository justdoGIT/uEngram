"""Deterministic label-free candidate ranking with synthetic temporal faults."""
from __future__ import annotations

import numpy as np

FAULT_FAMILIES = ("level_shift", "drift", "impulse", "oscillation", "freeze")


def generate_synthetic_faults(selection_windows, seed: int = 0, strength: float = 3.0):
    """Return one corrupted copy per fault family without modifying input data."""
    x = np.asarray(selection_windows)
    if x.ndim != 3 or x.shape[0] == 0 or x.shape[2] < 4:
        raise ValueError("selection_windows must have shape [N,C,K], N > 0, K >= 4")
    if not np.isfinite(x).all() or strength <= 0:
        raise ValueError("windows must be finite and strength must be positive")
    rng = np.random.default_rng(seed)
    original = x.astype(np.float32, copy=False)
    result = {}
    n, channels, length = original.shape
    scale = np.maximum(original.std(axis=(0, 2)), 1e-6).astype(np.float32)
    offsets = rng.integers(0, max(1, length // 2), size=n)
    spans = np.maximum(1, length // 4)
    for family in FAULT_FAMILIES:
        damaged = original.copy()
        for row in range(n):
            channel = int(rng.integers(channels))
            offset = int(offsets[row])
            span = min(spans, length - offset)
            sl = slice(offset, offset + span)
            amplitude = strength * scale[channel]
            if family == "level_shift":
                damaged[row, channel, offset:] += amplitude
            elif family == "drift":
                damaged[row, channel, offset:] += np.linspace(0, amplitude, length - offset, dtype=np.float32)
            elif family == "impulse":
                damaged[row, channel, offset] += amplitude
            elif family == "oscillation":
                phase = rng.uniform(0, 2 * np.pi)
                damaged[row, channel, offset:] += amplitude * np.sin(
                    np.linspace(phase, phase + 8 * np.pi, length - offset, dtype=np.float32))
            else:  # freeze a local interval at its first observed value
                damaged[row, channel, sl] = damaged[row, channel, offset]
        result[family] = damaged
    return result


def _encode(encoder, windows):
    z = np.asarray(encoder.encode(windows, sequential=False), dtype=np.float64)
    if z.ndim != 2 or z.shape[0] != len(windows) or not np.isfinite(z).all():
        raise ValueError("candidate encode must return finite [N,D] values")
    return z


def rank_candidates(candidates, selection_windows, seed: int = 0, strength: float = 3.0,
                    quantile: float = 0.5):
    """Rank fitted encoders by their weakest synthetic-fault separation.

    ``candidates`` maps names to fitted objects implementing ``encode``. The
    supplied windows are assumed to be unlabeled normal selection data. No
    anomaly labels or external evaluation/test data are accepted.

    Returns records ordered best-first. Each record has ``name``, ``score``,
    ``worst_family_score``, and a ``fault_metrics`` mapping by fault family.
    Family values are the requested quantile of clean-to-corrupt latent
    displacement, normalized by median clean within-set displacement.
    """
    if not 0 < quantile <= 0.5:
        raise ValueError("quantile must be in (0, 0.5]")
    x = np.asarray(selection_windows)
    if x.ndim != 3 or len(x) < 2:
        raise ValueError("selection_windows must have shape [N,C,K] with N >= 2")
    if not candidates:
        raise ValueError("at least one candidate is required")
    faults = generate_synthetic_faults(x, seed=seed, strength=strength)
    records = []
    for name, encoder in candidates.items():
        clean = _encode(encoder, x)
        baseline = np.median(np.linalg.norm(clean[1:] - clean[:-1], axis=1))
        floor = max(float(baseline) * 1e-3, 1e-8)
        metrics = {}
        for family, corrupted in faults.items():
            altered = _encode(encoder, corrupted)
            metrics[family] = float(np.quantile(np.linalg.norm(altered - clean, axis=1), quantile) /
                                   max(baseline, floor))
        worst = min(metrics.values())
        records.append({"name": name, "score": float(worst), "worst_family_score": float(worst),
                        "fault_metrics": metrics})
    records.sort(key=lambda row: (-row["worst_family_score"], str(row["name"])))
    return records


def select_by_synthetic_challenges(encoders: dict[str, object], normal_windows: np.ndarray,
                                   *, alpha: float = 0.01) -> dict:
    """Choose a fitted encoder using unlabeled normal windows and synthetic faults.

    ``alpha`` is the lower-tail quantile used to require challenge sensitivity
    across at least that fraction of windows. The result contains the winning
    candidate, sorted ranking, and per-candidate metrics for every fault family.
    """
    if not 0 < alpha <= 0.5:
        raise ValueError("alpha must be in (0, 0.5]")
    records = rank_candidates(encoders, normal_windows, quantile=alpha)
    return {
        "winner": records[0]["name"],
        "ranking": [row["name"] for row in records],
        "per_candidate": {
            row["name"]: {"score": row["score"], "worst_family_score": row["worst_family_score"],
                          "fault_metrics": row["fault_metrics"]}
            for row in records
        },
    }
