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

    MOMENT was pretrained for 512-sample contexts. Shorter uEngram windows are
    left-padded by repeating their first sample; longer windows retain the most
    recent 512 samples. The model therefore sees only the current window and
    never receives samples after its right edge. This is online-causal
    alignment, not a claim that the transformer's attention is causal inside a
    window.
    """

    model: Any
    spec: TeacherSpec
    context_length: int
    device: str = "cpu"

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
    ) -> "MOMENTTeacherAdapter":
        """Load a public MOMENT checkpoint and declare its feature contract."""
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
        feature_dim = int(model.config.d_model)
        schema = WindowSchema(
            channel_names=tuple(channel_names),
            sample_rate_hz=sample_rate_hz,
            window_size=window_size,
        )
        spec = TeacherSpec(
            model_name=model_id,
            model_version=revision or str(
                getattr(model.config, "_name_or_path", model_id)
            ),
            layer="embedding-mean-patches",
            feature_dim=feature_dim,
            input_schema=schema,
            causal=True,
            alignment="right",
        )
        return cls(model=model, spec=spec, context_length=context_length, device=device)

    def _prepare(self, windows: np.ndarray):
        import torch

        values = np.asarray(windows, dtype=np.float32)
        self.spec.input_schema.validate(values)
        _, _, samples = values.shape
        if samples < self.context_length:
            pad = np.repeat(values[:, :, :1], self.context_length - samples, axis=2)
            values = np.concatenate((pad, values), axis=2)
        elif samples > self.context_length:
            values = values[:, :, -self.context_length :]
        return torch.from_numpy(np.ascontiguousarray(values)).to(self.device)

    def encode(self, windows: np.ndarray) -> np.ndarray:
        """Return one dense embedding per input window, preserving row order."""
        import torch

        x = self._prepare(windows)
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
        return embeddings.detach().cpu().numpy()
