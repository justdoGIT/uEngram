"""Optional adapter for the pretrained MOMENT time-series foundation model.

The dependency is imported only when :meth:`from_pretrained` is called. This
keeps the core uEngram package usable on a small embedded target that does not
ship PyTorch or Hugging Face libraries.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from .teacher_transfer import TeacherSpec, WindowSchema


@dataclass
class MOMENTTeacherAdapter:
    """Run a frozen MOMENT embedding model on causal sensor windows.

    MOMENT was pretrained for 512-sample contexts. The ``edge`` policy
    left-pads short uEngram windows by repeating their first sample and keeps
    the newest samples when a window is longer. The ``linear`` policy resamples
    the full window to the native context, preserving its endpoints. Both are
    online-causal: the model never receives samples after the input window's
    right edge.
    """

    model: Any
    spec: TeacherSpec
    context_length: int
    device: str = "cpu"
    context_policy: str = "edge"
    include_instance_stats: bool = False
    base_feature_dim: int = 0

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_id: str = "AutonLab/MOMENT-1-large",
        channel_names: tuple[str, ...],
        sample_rate_hz: float,
        window_size: int,
        device: str = "cpu",
        local_files_only: bool = False,
        revision: str | None = None,
        context_policy: str = "edge",
        include_instance_stats: bool = False,
    ) -> "MOMENTTeacherAdapter":
        """Load a public MOMENT checkpoint and declare its feature contract."""
        if context_policy not in ("edge", "linear"):
            raise ValueError("context_policy must be 'edge' or 'linear'")
        try:
            import torch
            from momentfm import MOMENTPipeline
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "MOMENT support requires the optional momentfm environment; "
                "install momentfm==0.1.4 and its runtime dependencies"
            ) from exc

        load_kwargs = {"model_kwargs": {"task_name": "embedding"}}
        if revision is not None:
            load_kwargs["revision"] = revision
        if local_files_only:
            # PyTorchModelHubMixin 0.24 reads offline mode from the
            # environment rather than accepting local_files_only here.
            os.environ["HF_HUB_OFFLINE"] = "1"
        model = MOMENTPipeline.from_pretrained(model_id, **load_kwargs)
        model.init()
        model.to(device)
        model.eval()
        context_length = int(model.seq_len)
        base_feature_dim = int(model.config.d_model)
        schema = WindowSchema(
            channel_names=tuple(channel_names),
            sample_rate_hz=sample_rate_hz,
            window_size=window_size,
        )
        layer = "embedding-mean-patches"
        feature_dim = base_feature_dim
        if include_instance_stats:
            layer += "+instance-stats"
            feature_dim += 2 * len(channel_names)
        spec = TeacherSpec(
            model_name=model_id,
            model_version=revision or str(
                getattr(model.config, "_name_or_path", model_id)
            ),
            layer=layer,
            feature_dim=feature_dim,
            input_schema=schema,
            causal=True,
            alignment="right",
        )
        return cls(
            model=model,
            spec=spec,
            context_length=context_length,
            device=device,
            context_policy=context_policy,
            include_instance_stats=include_instance_stats,
            base_feature_dim=base_feature_dim,
        )

    def _prepare(self, windows: np.ndarray):
        import torch

        values = np.asarray(windows, dtype=np.float32)
        if self.context_policy not in ("edge", "linear"):
            raise ValueError("context_policy must be 'edge' or 'linear'")
        _, _, samples = values.shape
        if self.context_policy == "linear":
            source = np.linspace(0.0, 1.0, samples)
            target = np.linspace(0.0, 1.0, self.context_length)
            values = np.stack([
                np.stack([np.interp(target, source, row) for row in batch])
                for batch in values
            ]).astype(np.float32)
        elif samples < self.context_length:
            pad = np.repeat(values[:, :, :1], self.context_length - samples, axis=2)
            values = np.concatenate((pad, values), axis=2)
        elif samples > self.context_length:
            values = values[:, :, -self.context_length :]
        return torch.from_numpy(np.ascontiguousarray(values)).to(self.device)

    def encode(self, windows: np.ndarray) -> np.ndarray:
        """Return one dense embedding per input window, preserving row order."""
        import torch

        values = np.asarray(windows, dtype=np.float32)
        self.spec.input_schema.validate(values)
        x = self._prepare(values)
        mask = torch.ones((x.shape[0], self.context_length), device=x.device)
        with torch.inference_mode():
            output = self.model(x_enc=x, input_mask=mask)
            embeddings = output.embeddings
            if embeddings.ndim == 3:
                embeddings = embeddings.mean(dim=1)
            if embeddings.ndim != 2:
                raise ValueError(
                    "MOMENT embedding output must be (batch, features) or "
                    "(batch, patches, features)"
                )
        result = embeddings.detach().cpu().numpy()
        expected_base = self.base_feature_dim or result.shape[1]
        if result.shape[1] != expected_base:
            raise ValueError(
                "MOMENT embedding width does not match the declared feature contract"
            )
        if self.include_instance_stats:
            stats = np.concatenate(
                (values.mean(axis=2), values.std(axis=2)), axis=1
            )
            if result.shape[1] + stats.shape[1] != self.spec.feature_dim:
                raise ValueError(
                    "instance-statistics width does not match the declared feature contract"
                )
            result = np.concatenate((result, stats), axis=1)
        return result
