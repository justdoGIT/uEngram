"""1D-CNN autoencoder encoder (torch imported lazily)."""
from __future__ import annotations

import numpy as np

from .encoders import fit_projection


class CNNEncoder:
    def __init__(self, seed: int = 0, d: int = 16, epochs: int = 30, batch: int = 128):
        self.seed, self.d, self.epochs, self.batch = seed, d, epochs, batch

    def _build(self, C: int, K: int):
        import torch.nn as nn

        enc = nn.Sequential(
            nn.Conv1d(C, 32, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, self.d),
        )
        dec = nn.Sequential(nn.Linear(self.d, 128), nn.GELU(), nn.Linear(128, C * K))
        return enc, dec

    def _std(self, windows):
        return (np.asarray(windows, dtype=np.float32) - self.ch_mean) / self.ch_std

    def fit(self, train_windows, noise_lag: int, groups: np.ndarray | None) -> "CNNEncoder":
        import torch

        previous_threads = torch.get_num_threads()
        previous_determinism = torch.are_deterministic_algorithms_enabled()
        torch.manual_seed(self.seed)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        x = np.asarray(train_windows, dtype=np.float32)
        N, C, K = x.shape
        self.ch_mean = x.mean(axis=(0, 2), keepdims=True)[0].astype(np.float32)
        self.ch_std = np.maximum(x.std(axis=(0, 2), keepdims=True)[0], 1e-6).astype(np.float32)
        xt = torch.from_numpy(self._std(x))
        self.enc, dec = self._build(C, K)
        params = list(self.enc.parameters()) + list(dec.parameters())
        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
        gen = torch.Generator().manual_seed(self.seed)
        self.enc.train(); dec.train()
        for _ in range(self.epochs):
            perm = torch.randperm(N, generator=gen)
            for i in range(0, N, self.batch):
                xb = xt[perm[i : i + self.batch]]
                loss = torch.mean((dec(self.enc(xb)).reshape(-1, C, K) - xb) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.enc.eval()
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_determinism)
        self.whitener, self.noise, self.sigma0 = fit_projection(self._latent(x), self.d, noise_lag, groups)
        return self

    def _latent(self, windows) -> np.ndarray:
        import torch

        xt = torch.from_numpy(self._std(windows))
        out = []
        with torch.no_grad():
            for i in range(0, xt.shape[0], 1024):
                out.append(self.enc(xt[i : i + 1024]).numpy())
        return np.concatenate(out, axis=0).astype(np.float64)

    def encode(self, windows, sequential: bool = True) -> np.ndarray:
        """`sequential` is accepted for protocol compatibility; windows are independent here."""
        return self.noise.transform(self.whitener.transform(self._latent(windows)))
