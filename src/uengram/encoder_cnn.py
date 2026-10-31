"""1D-CNN autoencoder encoder (torch imported lazily)."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .encoders import fit_projection
from .teacher_transfer import (
    TeacherFeatures,
    TeacherSpec,
    extract_multi_teacher_features,
    extract_teacher_features,
)


class CNNEncoder:
    def __init__(
        self,
        seed: int = 0,
        d: int = 16,
        epochs: int = 30,
        batch: int = 128,
        pool: str = "avg",
        ortho_beta: float = 0.0,
        representation: str = "bottleneck",
    ):
        if pool not in ("avg", "maxavg"):
            raise ValueError("pool must be 'avg' or 'maxavg'")
        if representation not in ("bottleneck", "middle", "multiscale"):
            raise ValueError(
                "representation must be 'bottleneck', 'middle', or 'multiscale'"
            )
        self.seed, self.d, self.epochs, self.batch = seed, d, epochs, batch
        self.pool, self.ortho_beta = pool, ortho_beta
        self.representation = representation

    def _build(self, C: int, K: int, teacher_dim: int | None = None):
        import torch.nn as nn

        self.conv = nn.Sequential(
            nn.Conv1d(C, 32, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.GELU(),
        )
        pooled_dim = 128 if self.pool == "maxavg" else 64
        middle_dim = 64 if self.pool == "maxavg" else 32
        self.fc = nn.Linear(
            pooled_dim, self.d, bias=self.ortho_beta == 0.0
        )
        if self.representation == "middle":
            self.readout_dim = middle_dim
        elif self.representation == "multiscale":
            self.readout_dim = middle_dim + pooled_dim
        else:
            self.readout_dim = self.d
        self.teacher_head = (
            nn.Linear(self.readout_dim, teacher_dim)
            if teacher_dim is not None
            else None
        )
        dec = nn.Sequential(
            nn.Linear(self.d, 128), nn.GELU(), nn.Linear(128, C * K)
        )
        return dec

    def _pool(self, h):
        import torch

        if self.pool == "maxavg":
            return torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=-1)
        return h.mean(dim=-1)

    def _feature_maps(self, xb):
        middle = self.conv[1](self.conv[0](xb))
        final = self.conv[3](self.conv[2](middle))
        return middle, final

    def _forward_enc(self, xb):
        _, final = self._feature_maps(xb)
        return self.fc(self._pool(final))

    def _readout(self, xb):
        import torch

        middle, final = self._feature_maps(xb)
        if self.representation == "middle":
            return self._pool(middle)
        if self.representation == "multiscale":
            return torch.cat([self._pool(middle), self._pool(final)], dim=-1)
        return self.fc(self._pool(final))

    def _std(self, windows):
        return (np.asarray(windows, dtype=np.float32) - self.ch_mean) / self.ch_std

    def fit(self, train_windows, noise_lag: int, groups: np.ndarray | None) -> "CNNEncoder":
        return self._fit(
            train_windows, noise_lag, groups, teacher_features=None
        )

    def fit_distilled(
        self,
        train_windows,
        teacher,
        teacher_spec: TeacherSpec,
        noise_lag: int,
        groups: np.ndarray | None,
        teacher_weight: float = 0.25,
        teacher_batch: int = 1024,
    ) -> "CNNEncoder":
        """Train the student with reconstruction and teacher-feature loss.

        The teacher is an offline adapter. Its declared representation must be
        causal and right-aligned; no future-looking teacher can enter this path.
        """
        if teacher_weight < 0 or not np.isfinite(teacher_weight):
            raise ValueError("teacher_weight must be finite and non-negative")
        features = extract_teacher_features(
            teacher, np.asarray(train_windows, dtype=np.float32),
            teacher_spec, batch=teacher_batch,
        )
        return self._fit(train_windows, noise_lag, groups, features, teacher_weight)

    def fit_distilled_multi(
        self,
        train_windows,
        teachers: Sequence[tuple[object, TeacherSpec]],
        noise_lag: int,
        groups: np.ndarray | None,
        teacher_weight: float = 0.25,
        teacher_batch: int = 1024,
    ) -> "CNNEncoder":
        """Distill equal-weight teacher feature blocks into one student head."""
        if teacher_weight < 0 or not np.isfinite(teacher_weight):
            raise ValueError("teacher_weight must be finite and non-negative")
        features = extract_multi_teacher_features(
            teachers, np.asarray(train_windows, dtype=np.float32), batch=teacher_batch
        )
        return self._fit(train_windows, noise_lag, groups, features, teacher_weight)

    def _fit(
        self,
        train_windows,
        noise_lag: int,
        groups: np.ndarray | None,
        teacher_features: TeacherFeatures | None,
        teacher_weight: float = 0.0,
    ) -> "CNNEncoder":
        import torch

        previous_threads = torch.get_num_threads()
        previous_determinism = torch.are_deterministic_algorithms_enabled()
        try:
            torch.manual_seed(self.seed)
            torch.set_num_threads(4)
            torch.use_deterministic_algorithms(True)
            x = np.asarray(train_windows, dtype=np.float32)
            N, C, K = x.shape
            if teacher_features is not None:
                if len(teacher_features.values) != N:
                    raise ValueError("teacher feature count must match train windows")
                self.teacher_spec = teacher_features.spec
                self.teacher_mean = teacher_features.values.mean(axis=0)
                self.teacher_std = np.maximum(
                    teacher_features.values.std(axis=0), 1e-6
                )
                teacher_target = (
                    teacher_features.values - self.teacher_mean
                ) / self.teacher_std
                teacher_dim = teacher_features.spec.feature_dim
            else:
                self.teacher_spec = None
                self.teacher_mean = None
                self.teacher_std = None
                teacher_target = None
                teacher_dim = None
            self.ch_mean = x.mean(axis=(0, 2), keepdims=True)[0].astype(np.float32)
            self.ch_std = np.maximum(
                x.std(axis=(0, 2), keepdims=True)[0], 1e-6
            ).astype(np.float32)
            xt = torch.from_numpy(self._std(x))
            dec = self._build(C, K, teacher_dim)
            params = [*self.conv.parameters(), *self.fc.parameters(), *dec.parameters()]
            if self.teacher_head is not None:
                params.extend(self.teacher_head.parameters())
            opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
            gen = torch.Generator().manual_seed(self.seed)
            self.conv.train(); self.fc.train(); dec.train()
            if self.teacher_head is not None:
                self.teacher_head.train()
            I = torch.eye(self.d) if self.ortho_beta > 0.0 else None
            target = (
                torch.from_numpy(teacher_target.astype(np.float32))
                if teacher_target is not None
                else None
            )
            for _ in range(self.epochs):
                perm = torch.randperm(N, generator=gen)
                for i in range(0, N, self.batch):
                    indices = perm[i : i + self.batch]
                    xb = xt[indices]
                    z = self._forward_enc(xb)
                    recon = torch.mean(
                        (dec(z).reshape(-1, C, K) - xb) ** 2
                    )
                    loss = recon
                    if self.ortho_beta > 0.0:
                        W = self.fc.weight
                        loss = loss + self.ortho_beta * torch.mean(
                            (W @ W.T - I) ** 2
                        )
                    if target is not None:
                        student = self.teacher_head(self._readout(xb))
                        if teacher_features is not None and teacher_features.block_slices:
                            teacher_loss = torch.stack(
                                [
                                    torch.mean(
                                        (student[:, start:end] - target[indices, start:end]) ** 2
                                    )
                                    for start, end in teacher_features.block_slices
                                ]
                            ).mean()
                        else:
                            teacher_loss = torch.mean(
                                (student - target[indices]) ** 2
                            )
                        loss = loss + teacher_weight * teacher_loss
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            self.conv.eval(); self.fc.eval()
            if self.teacher_head is not None:
                self.teacher_head.eval()
            self.whitener, self.noise, self.sigma0 = fit_projection(
                self._latent(x), self.d, noise_lag, groups
            )
            return self
        finally:
            torch.set_num_threads(previous_threads)
            torch.use_deterministic_algorithms(previous_determinism)

    def _latent(self, windows) -> np.ndarray:
        import torch

        xt = torch.from_numpy(self._std(windows))
        out = []
        with torch.no_grad():
            for i in range(0, xt.shape[0], 1024):
                out.append(self._readout(xt[i : i + 1024]).numpy())
        return np.concatenate(out, axis=0).astype(np.float64)
    def encode(self, windows, sequential: bool = True) -> np.ndarray:
        """`sequential` is accepted for protocol compatibility; windows are independent here."""
        return self.noise.transform(self.whitener.transform(self._latent(windows)))
