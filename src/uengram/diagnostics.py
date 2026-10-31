"""Pre-fit numerical conditioning and channel redundancy diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class ChannelDiagnosticReport:
    """Diagnostic report on telemetry channel conditioning and redundancy."""

    status: Literal[
        "well_conditioned",
        "moderately_conditioned",
        "ill_conditioned",
        "singular",
        "insufficient_valid_samples",
    ]
    n_samples: int
    valid_samples: int
    invalid_samples: int
    invalid_counts: dict[str, int]
    matrix_rank: int
    full_rank: int
    condition_number: float
    singular_values: tuple[float, ...]
    redundant_groups: tuple[tuple[str, ...], ...]
    colinear_pairs: tuple[tuple[str, str, float], ...]

    @property
    def is_safe(self) -> bool:
        """True when the channel set is safe for covariance inversion."""
        return self.status in ("well_conditioned", "moderately_conditioned")

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "is_safe": self.is_safe,
            "n_samples": self.n_samples,
            "valid_samples": self.valid_samples,
            "invalid_samples": self.invalid_samples,
            "invalid_counts": self.invalid_counts,
            "matrix_rank": self.matrix_rank,
            "full_rank": self.full_rank,
            "condition_number": self.condition_number,
            "singular_values": list(self.singular_values),
            "redundant_groups": [list(g) for g in self.redundant_groups],
            "colinear_pairs": [
                {"channel_a": a, "channel_b": b, "correlation": corr}
                for a, b, corr in self.colinear_pairs
            ],
        }


def _extract_2d_matrix(values) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        # For windows [N, C, L], use the window temporal mean per channel
        # to diagnose channel-level collinearity across windows.
        return x.mean(axis=2)
    raise ValueError("values must have shape (n_samples, n_channels) or (n_windows, n_channels, length)")


def diagnose_channels(
    values,
    channel_names: list[str] | tuple[str, ...] | None = None,
    sentinel_threshold: float = -90000.0,
) -> ChannelDiagnosticReport:
    """Diagnose matrix rank, conditioning, sentinels, and channel redundancy."""
    x = _extract_2d_matrix(values)
    n_samples, n_channels = x.shape
    if n_channels < 1:
        raise ValueError("need at least one channel to diagnose")

    names = (
        list(channel_names)
        if channel_names is not None
        else [f"ch_{i}" for i in range(n_channels)]
    )
    if len(names) != n_channels:
        raise ValueError(
            f"channel_names length ({len(names)}) does not match channel count ({n_channels})"
        )

    invalid_mask = ~np.isfinite(x) | (x <= sentinel_threshold)
    invalid_counts = {
        name: int(invalid_mask[:, i].sum()) for i, name in enumerate(names)
    }
    total_invalid = int(invalid_mask.sum())

    valid_rows = ~invalid_mask.any(axis=1)
    n_valid = int(valid_rows.sum())
    min_required = max(10, n_channels + 1)
    if n_valid < min_required:
        return ChannelDiagnosticReport(
            status="insufficient_valid_samples",
            n_samples=n_samples,
            valid_samples=n_valid,
            invalid_samples=total_invalid,
            invalid_counts=invalid_counts,
            matrix_rank=0,
            full_rank=n_channels,
            condition_number=float("inf"),
            singular_values=(),
            redundant_groups=(),
            colinear_pairs=(),
        )

    clean_x = x[valid_rows]
    cov = np.cov(clean_x, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[cov]])

    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, 0.0)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    max_ev = float(evals[0]) if len(evals) > 0 else 0.0
    min_ev = float(evals[-1]) if len(evals) > 0 else 0.0

    rcond_eps = np.finfo(np.float64).eps * max(max_ev, 1e-12) * n_channels
    effective_rank = int(np.sum(evals > rcond_eps))
    cond = float(max_ev / max(min_ev, 1e-18))

    redundant_groups: list[tuple[str, ...]] = []
    if effective_rank < n_channels:
        null_vecs = evecs[:, effective_rank:]
        for j in range(null_vecs.shape[1]):
            v = null_vecs[:, j]
            group = tuple(names[i] for i in np.where(np.abs(v) > 0.1)[0])
            if group and group not in redundant_groups:
                redundant_groups.append(group)

    perfect_pairs: list[tuple[str, str, float]] = []
    std = clean_x.std(axis=0)
    non_constant = np.where(std > 1e-9)[0]
    if len(non_constant) > 1:
        sub_x = clean_x[:, non_constant]
        corr = np.corrcoef(sub_x, rowvar=False)
        if corr.ndim == 2:
            for idx_a in range(len(non_constant)):
                for idx_b in range(idx_a + 1, len(non_constant)):
                    val = float(corr[idx_a, idx_b])
                    if abs(abs(val) - 1.0) < 1e-5:
                        orig_a = names[non_constant[idx_a]]
                        orig_b = names[non_constant[idx_b]]
                        perfect_pairs.append((orig_a, orig_b, val))

    status: Literal[
        "well_conditioned",
        "moderately_conditioned",
        "ill_conditioned",
        "singular",
    ]
    if effective_rank < n_channels or cond > 1e12:
        status = "singular" if effective_rank < n_channels else "ill_conditioned"
    elif cond > 1e6:
        status = "moderately_conditioned"
    else:
        status = "well_conditioned"

    return ChannelDiagnosticReport(
        status=status,
        n_samples=n_samples,
        valid_samples=n_valid,
        invalid_samples=total_invalid,
        invalid_counts=invalid_counts,
        matrix_rank=effective_rank,
        full_rank=n_channels,
        condition_number=cond,
        singular_values=tuple(float(v) for v in evals),
        redundant_groups=tuple(redundant_groups),
        colinear_pairs=tuple(perfect_pairs),
    )
