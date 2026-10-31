"""Streaming detector baselines: Half-Space Trees (River) and MemStream core.

MemStream is adapted from https://github.com/Stream-AD/MemStream (Apache-2.0;
license in MemStream-LICENSE.txt). Changes: shared-latent input, seed-local
initialization, CPU execution, independent calibration clones, and ingest API.
This is not the paper's full end-to-end feature pipeline.

Both expose `ingest(z, t, raw_idx)` returning an IngestResult with item_id -1
(no stored item) and flagged=None, so the harness streams them like stores.
"""
from __future__ import annotations

import copy

import numpy as np

from .store import IngestResult


class HSTDetector:
    """Streaming Half-Space Trees (Tan, Ting & Liu, IJCAI 2011), via River.

    Limits are fixed to the per-dimension min/max of the training latents.
    Each observation is scored before it is learned. The first window is a
    warmup period and should be excluded from calibration and evaluation.
    """

    name = "hst"
    in_ref_space = False

    def __init__(self, Ztrain: np.ndarray, seed: int, n_trees: int = 10, height: int = 8,
                 window_size: int = 250):
        from river.anomaly import HalfSpaceTrees

        Ztrain = np.asarray(Ztrain)
        lo, hi = Ztrain.min(axis=0), Ztrain.max(axis=0)
        limits = {j: (float(lo[j]), float(hi[j])) for j in range(Ztrain.shape[1])}
        self.warmup_count = window_size
        self.model = HalfSpaceTrees(n_trees=n_trees, height=height, window_size=window_size,
                                    limits=limits, seed=seed)

    def ingest(self, z, t: int, raw_idx: int, **_) -> IngestResult:
        x = {j: float(v) for j, v in enumerate(z)}
        score = float(self.model.score_one(x))
        self.model.learn_one(x)
        return IngestResult(-1, score, None, "score")


class MemStream:
    """Faithful MemStream core based on Bhatia et al. (WWW 2022) reference code.

    The denoising autoencoder (Linear(D, 2D) + tanh / Linear(2D, D)) is
    trained on the initial raw memory, using float32, sample standard
    deviation, and the reference 0.001 Gaussian input noise. The encoder is
    then frozen. Scores are minimum L1 distances to its encoded memory; an
    observation at or below beta replaces the next FIFO slot, and raw-memory
    normalization statistics are recomputed. This is the detector core, not
    the paper's full dataset/feature pipeline.
    """

    name = "memstream"
    in_ref_space = False

    def __init__(self, Zinit: np.ndarray, seed: int, memory_len: int = 256, epochs: int = 5000,
                 lr: float = 1e-2, beta: float = float("inf"), progress_callback=None):
        import torch
        import torch.nn as nn

        torch.set_num_threads(min(torch.get_num_threads(), 2))
        Zinit = np.asarray(Zinit)
        if Zinit.ndim != 2:
            raise ValueError(f"initial memory must be a 2D array, got shape {Zinit.shape}")
        if Zinit.shape[0] < memory_len:
            raise ValueError(f"need {memory_len} initial records, got {Zinit.shape[0]}")
        if Zinit.shape[1] == 0:
            raise ValueError("initial memory must have at least one feature")
        x = torch.as_tensor(Zinit[:memory_len], dtype=torch.float32).clone()
        d = x.shape[1]
        self.memory_len, self.beta, self.count = memory_len, beta, 0
        self.mem_data = x
        self._refresh_stats()
        # Do not perturb the caller's global RNG, but seed both parameter
        # initialization and reference denoising noise reproducibly.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = nn.Sequential(nn.Linear(d, 2 * d), nn.Tanh())
            decoder = nn.Linear(2 * d, d)
            opt = torch.optim.Adam(list(self.encoder.parameters()) + list(decoder.parameters()), lr=lr)
            new = self._normalise(x)
            for epoch in range(epochs):
                opt.zero_grad()
                out = decoder(self.encoder(new + 0.001 * torch.randn_like(new)))
                loss = torch.mean((out - new) ** 2)
                loss.backward()
                opt.step()
                if progress_callback is not None and (epoch + 1) % 250 == 0:
                    progress_callback(epoch + 1, epochs)
            with torch.no_grad():
                self.memory = self.encoder(new).detach()
        self.encoder.eval()
        self._feature_count = d

    @property
    def memory_bytes(self) -> int:
        """Exact bytes of the raw and encoded memory tensors (not Python overhead)."""
        return (self.mem_data.numel() * self.mem_data.element_size()
                + self.memory.numel() * self.memory.element_size())

    def _refresh_stats(self) -> None:
        self.mean, self.std = self.mem_data.mean(0), self.mem_data.std(0)

    def _normalise(self, x):
        new = (x - self.mean) / self.std
        new[:, self.std == 0] = 0
        return new

    def clone(self, beta: float) -> "MemStream":
        """An independent copy (trained encoder, initial memory) with update threshold beta."""
        other = copy.deepcopy(self)
        other.beta = beta
        return other

    def ingest(self, z, t: int, raw_idx: int, **_) -> IngestResult:
        import torch

        raw = np.asarray(z)
        if raw.shape != (self._feature_count,):
            raise ValueError(
                f"expected one raw record with shape ({self._feature_count},), got {raw.shape}"
            )
        with torch.no_grad():
            x = torch.as_tensor(raw, dtype=torch.float32).reshape(1, -1)
            enc = self.encoder(self._normalise(x))
            score = float(torch.norm(self.memory - enc, dim=1, p=1).min())
            if score <= self.beta:
                pos = self.count % self.memory_len
                self.memory[pos] = enc[0]
                self.mem_data[pos] = x[0]
                self._refresh_stats()
                self.count += 1
        return IngestResult(-1, score, None, "score")
