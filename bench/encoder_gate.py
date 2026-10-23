"""Label-free diagnostics for fit/calibration latent distributions."""
from __future__ import annotations

import numpy as np


def gate_stats(encoder, W_fit, W_cal, groups_fit, cond_fit=None, cond_cal=None) -> dict:
    """Summarize member agreement, validation, latent noise, and range coverage."""
    z_fit = encoder.encode(W_fit, cond=cond_fit) if cond_fit is not None else encoder.encode(W_fit)
    z_cal = encoder.encode(W_cal, cond=cond_cal) if cond_cal is not None else encoder.encode(W_cal)
    cka = getattr(encoder, "member_cka", None)
    if cka is None or np.size(cka) < 2:
        mean_cka = min_cka = float("nan")
    else:
        cka = np.asarray(cka, dtype=np.float64)
        pairs = cka[np.triu_indices(len(cka), 1)]
        mean_cka, min_cka = float(np.mean(pairs)), float(np.min(pairs))
    spread = np.maximum(np.std(z_fit, axis=0), 1e-12)
    sigma = np.asarray(encoder.sigma0, dtype=np.float64)
    ratio = float(np.median(sigma) / np.median(spread))
    lo, hi = z_fit.min(axis=0), z_fit.max(axis=0)
    outside = (z_cal < lo) | (z_cal > hi)
    return {"mean_member_cka": mean_cka, "min_member_cka": min_cka,
            "validation_loss": float(getattr(encoder, "validation_loss", np.nan)),
            "median_noise_to_spread": ratio,
            "outside_fit_range_fraction": float(outside.mean())}


def accept_gate(stats: dict) -> bool:
    """Apply provisional development thresholds; these are not guarantees."""
    keys = ("validation_loss", "mean_member_cka", "min_member_cka",
            "median_noise_to_spread", "outside_fit_range_fraction")
    try:
        values = np.asarray([stats[key] for key in keys], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return False
    if not np.isfinite(values).all():
        return False
    _, mean_cka, min_cka, noise_ratio, outside = values
    return bool(mean_cka >= 0.98 and min_cka >= 0.95 and noise_ratio <= 1.0
                and outside <= 0.01)
