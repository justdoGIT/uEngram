"""Task-aligned temporal CNN encoder (torch is imported only when fitting)."""
from __future__ import annotations

import numpy as np

from .encoders import band_features, fit_projection


class SSLTemporalEncoder:
    """Framewise temporal representation with explicit raw-level information path."""

    def __init__(self, seed: int = 0, d: int = 16, epochs: int = 30, batch: int = 128,
                 width: int = 32, mask_fraction: float = 0.15, corruption_fraction: float = 0.25,
                 learning_rate: float = 1e-3):
        if (not isinstance(d, (int, np.integer)) or isinstance(d, (bool, np.bool_))
                or d < 2 or d % 2):
            raise ValueError("d must be a positive even integer")
        if epochs < 1 or batch < 1 or width < 4:
            raise ValueError("epochs, batch, and width must be positive")
        if not 0 < mask_fraction < 1 or not 0 < corruption_fraction < 1:
            raise ValueError("mask_fraction and corruption_fraction must be in (0, 1)")
        self.seed, self.d, self.epochs, self.batch = seed, d, epochs, batch
        self.width, self.mask_fraction = width, mask_fraction
        self.corruption_fraction, self.learning_rate = corruption_fraction, learning_rate

    def _build(self, channels):
        import torch.nn as nn

        self.encoder = nn.Sequential(
            nn.Conv1d(channels, self.width, 5, padding=2), nn.GELU(),
            nn.Conv1d(self.width, self.width, 5, padding=2), nn.GELU(),
            nn.Conv1d(self.width, self.d, 1),
        )
        self.level = nn.Linear(channels, self.d, bias=False)
        self.decoder = nn.Conv1d(self.d, channels, 1)
        self.localizer = nn.Conv1d(self.d, 1, 1)

    def _frame(self, x):
        return self.encoder(x) + self.level(x.mean(dim=2)).unsqueeze(2)

    def _features(self, x):
        import torch
        h = self._frame(x)
        # Preserve raw per-channel level beside the learned temporal features.
        return torch.cat((h.mean(dim=2), x.mean(dim=2)), dim=1)
    def fit(self, train_windows, noise_lag: int, groups: np.ndarray | None) -> "SSLTemporalEncoder":
        import torch
        import torch.nn.functional as F

        x = np.asarray(train_windows, dtype=np.float32)
        if x.ndim != 3 or len(x) < 2:
            raise ValueError("train_windows must have shape [N,C,K] with N >= 2")
        n, channels, length = x.shape
        if length < 4 or not np.isfinite(x).all():
            raise ValueError("windows must be finite and contain at least four frames")
        previous_threads = torch.get_num_threads()
        previous_determinism = torch.are_deterministic_algorithms_enabled()
        previous_rng = torch.random.get_rng_state()
        try:
            torch.random.default_generator.manual_seed(self.seed)
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            self._build(channels)
            data = torch.from_numpy(x.copy())
            opt = torch.optim.AdamW(list(self.encoder.parameters()) + list(self.level.parameters()) +
                                    list(self.decoder.parameters()) + list(self.localizer.parameters()),
                                    lr=self.learning_rate, weight_decay=1e-4)
            gen = torch.Generator().manual_seed(self.seed)
            self.encoder.train(); self.level.train(); self.decoder.train(); self.localizer.train()
            for _ in range(self.epochs):
                order = torch.randperm(n, generator=gen)
                for start in range(0, n, self.batch):
                    xb = data[order[start:start + self.batch]]
                    b, _, k = xb.shape
                    # Mask contiguous temporal spans, reconstructing only masked values.
                    mask = torch.zeros((b, 1, k), dtype=xb.dtype)
                    span = max(1, int(k * self.mask_fraction))
                    for row in range(b):
                        offset = int(torch.randint(max(1, k - span + 1), (), generator=gen))
                        mask[row, :, offset:offset + span] = 1
                    masked = xb * (1 - mask)
                    h = self._frame(masked)
                    reconstruction = self.decoder(h)
                    recon_loss = (((reconstruction - xb) ** 2) * mask).sum() / (mask.sum() * channels + 1e-6)

                    # Synthetic localized faults train framewise corruption detection.
                    damaged = xb.clone()
                    labels = torch.zeros((b, 1, k), dtype=xb.dtype)
                    for row in range(b):
                        span = max(1, int(k * self.corruption_fraction))
                        offset = int(torch.randint(max(1, k - span + 1), (), generator=gen))
                        kind = int(torch.randint(4, (), generator=gen))
                        sl = slice(offset, offset + span)
                        labels[row, :, sl] = 1
                        if kind == 0:
                            damaged[row, :, sl] += torch.randn((channels, span), generator=gen) * 2.0
                        elif kind == 1:
                            damaged[row, :, sl] += 2.0
                        elif kind == 2:
                            damaged[row, :, sl] = damaged[row, :, offset:offset + 1]
                        else:
                            damaged[row, :, sl] *= -1
                    hd = self._frame(damaged)
                    loc_loss = F.binary_cross_entropy_with_logits(self.localizer(hd), labels)

                    # Compare only positions whose receptive fields stay inside
                    # both windows; zero padding is not circularly equivariant.
                    shift = int(torch.randint(1, max(2, k // 4 + 1), ()))
                    original_h = self._frame(xb)
                    shifted_h = self._frame(torch.roll(xb, shifts=shift, dims=2))
                    position = torch.arange(k)
                    source = (position - shift) % k
                    valid = (
                        (position >= 4) & (position < k - 4)
                        & (source >= 4) & (source < k - 4)
                    )
                    if bool(valid.any()):
                        equiv_loss = F.mse_loss(
                            shifted_h[:, :, valid], original_h[:, :, source[valid]]
                        )
                    else:
                        equiv_loss = original_h.new_zeros(())
                    loss = recon_loss + 0.5 * loc_loss + 0.25 * equiv_loss
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            self.encoder.eval(); self.level.eval(); self.decoder.eval(); self.localizer.eval()
        finally:
            torch.set_num_threads(previous_threads)
            torch.use_deterministic_algorithms(previous_determinism)
            torch.random.set_rng_state(previous_rng)
        self.whitener, self.noise, self.sigma0 = fit_projection(self._latent(x), self.d, noise_lag, groups)
        return self

    def _latent(self, windows):
        import torch
        x = torch.as_tensor(np.asarray(windows, dtype=np.float32))
        if x.ndim != 3 or x.shape[1] != self.level.in_features:
            raise ValueError("windows must have shape [N, fitted_channels, K]")
        outputs = []
        with torch.no_grad():
            for start in range(0, len(x), 512):
                outputs.append(self._features(x[start:start + 512]).cpu().numpy())
        channels = int(self.level.in_features)
        if not outputs:
            return np.empty((0, self.d + 7 * channels), dtype=np.float64)
        temporal = np.concatenate(outputs).astype(np.float64) * 0.125
        spectral = band_features(np.asarray(windows, dtype=np.float64))
        return np.concatenate((temporal, spectral), axis=1)

    def encode(self, windows, sequential: bool = True) -> np.ndarray:
        """Encode windows independently; sequential is accepted for protocol compatibility."""
        del sequential
        return self.noise.transform(self.whitener.transform(self._latent(windows)))
