"""Optional Chronos-2 forecasting adapter for offline distillation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .teacher_transfer import TeacherSpec, WindowSchema


@dataclass
class Chronos2TeacherAdapter:
    """Turn Chronos-2 quantile forecasts into one feature row per window."""

    pipeline: Any
    spec: TeacherSpec
    prediction_length: int
    batch_size: int = 32

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_id: str = "amazon/chronos-2",
        channel_names: tuple[str, ...],
        sample_rate_hz: float,
        window_size: int,
        prediction_length: int = 8,
        batch_size: int = 32,
        revision: str | None = None,
        device: str = "cpu",
    ) -> "Chronos2TeacherAdapter":
        """Load a pinned Chronos-2 checkpoint with an explicit sensor contract."""
        if prediction_length <= 0 or batch_size <= 0:
            raise ValueError("prediction_length and batch_size must be positive")
        try:
            from chronos import Chronos2Pipeline
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Chronos-2 support requires chronos-forecasting"
            ) from exc
        kwargs = {"device_map": device}
        if revision is not None:
            kwargs["revision"] = revision
        pipeline = Chronos2Pipeline.from_pretrained(model_id, **kwargs)
        spec = TeacherSpec(
            model_name=model_id,
            model_version=revision or model_id,
            layer="forecast-quantiles-0.1-0.5-0.9+median",
            feature_dim=4 * prediction_length * len(channel_names),
            input_schema=WindowSchema(
                channel_names=tuple(channel_names),
                sample_rate_hz=sample_rate_hz,
                window_size=window_size,
            ),
            causal=True,
            alignment="right",
        )
        return cls(pipeline, spec, prediction_length, batch_size)

    def encode(self, windows: np.ndarray) -> np.ndarray:
        """Return quantile and median forecasts without using future observations."""
        values = np.asarray(windows, dtype=np.float32)
        self.spec.input_schema.validate(values)
        if len(values) == 0:
            return np.empty((0, self.spec.feature_dim), dtype=np.float64)
        quantiles, medians = self.pipeline.predict_quantiles(
            values,
            prediction_length=self.prediction_length,
            quantile_levels=[0.1, 0.5, 0.9],
            batch_size=self.batch_size,
        )
        q = np.stack([item.detach().cpu().numpy() for item in quantiles])
        median = np.stack([item.detach().cpu().numpy() for item in medians])
        if q.ndim != 4 or median.ndim != 3:
            raise ValueError("Chronos output has an unexpected shape")
        features = np.concatenate(
            (q.reshape(len(values), -1), median.reshape(len(values), -1)), axis=1
        )
        if features.shape != (len(values), self.spec.feature_dim):
            raise ValueError("Chronos output width does not match the feature contract")
        return features.astype(np.float64, copy=False)
