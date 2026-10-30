"""DSP encoder augmented with each channel's windowed-mean slope.

Tests docs/research-remediation-plan.md RC2's proposal: a rate of change,
not just a window's level, for faults that develop gradually (a leak, a
slowly rising water level) rather than producing a spectral or step
change the plain `DSPEncoder` already catches.

**Project-wide revalidation verdict (docs/research-skab-validation.md):**
this is a narrow, SKAB-specific fix, not a general-purpose replacement for
`DSPEncoder`. Measured against the frozen `dsp` encoder on every existing
benchmark with the same alarm policy: it is a severe regression on the
synthetic benchmark (event_recall 1.00 -> 0.23, seizure_recall 1.00 -> 0.00,
AP 0.995 -> 0.068, 5x more writes, all 10 seeds), and a mixed result on
CMAPSS and MetroPT-3 (worse novelty AUROC/AP, better on some RUL/false-alarm
metrics, worse on others). It must not be selected via `bench.run_synth
.make_encoder`/`"dsp_slope"` for any confirmed or development result;
it exists there only so the revalidation above could reuse the shared
harness. The only place it is a measured, documented improvement is the
three gradual, non-vibration SKAB faults it targets.

`fit`/`encode` take the same `groups`/`segment` positional slot every other
encoder in `bench.run_synth.make_encoder` already uses (`None` for the
synthetic benchmark, CMAPSS's per-unit ids, MetroPT's/SKAB's logging
segment ids): a slope is computed only against a same-segment predecessor,
never across a gap, unit, or fit/cal/test boundary. `segment=None` means a
single implicit segment (the synthetic benchmark's convention).
"""
from __future__ import annotations

import numpy as np

from .encoders import DSPEncoder, band_features, windowed_slopes


class DSPSlopeEncoder(DSPEncoder):
    def __init__(self, d: int = 16, lags: tuple[int, ...] = (3, 12)):
        super().__init__(d)
        self.lags = tuple(lags)

    def _slopes(self, windows, segment) -> np.ndarray:
        """band_features(windows) plus, for each lag, each channel's
        windowed mean minus its value `lag` windows earlier (zero when that
        earlier window is missing or in a different segment)."""
        F = band_features(windows)
        means = F[:, 0::6]  # one column per channel: band_features' layout is
        # [mean, log_std, b1..b4] per channel, so index 0 within each 6-block.
        return np.column_stack([F, windowed_slopes(means, segment, self.lags)])

    def fit(self, train_windows, noise_lag: int, groups=None) -> "DSPSlopeEncoder":
        F = self._slopes(train_windows, groups)
        self.med = np.median(F, axis=0)
        q75, q25 = np.percentile(F, [75, 25], axis=0)
        robust = (q75 - q25) / 1.349
        spread = F.std(axis=0)
        self.scale = np.where(robust > 1e-6, robust, np.maximum(spread, 1e-6))
        self._fit_projection((F - self.med) / self.scale, noise_lag, groups)
        return self

    def encode(self, windows, sequential: bool = True, segment=None) -> np.ndarray:
        """`sequential` is accepted for interface consistency with other
        encoders; this encoder's slopes are already segment-gated, not
        state-dependent. `segment` must be the *same* ids the caller fit
        with (or its test-time equivalent, e.g. CMAPSS unit ids) for a
        slope to mean anything; omitting it on data with real segments
        silently computes every slope as if there were only one segment."""
        F = (self._slopes(windows, segment) - self.med) / self.scale
        return self.noise.transform(self.whitener.transform(F))
