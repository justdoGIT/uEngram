"""Phase-tolerant convolutional encoder with supervised-by-view self-supervision.

Two independently shifted/noisy views are pulled together by NT-Xent; no gain
or scale augmentation is used because absolute sensor levels carry fault
information. Auxiliary regression on per-channel quantiles, mean absolute
first difference and above-median occupancy preserves level and duty signals.
When CUDA is used, callers must set ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` before
CUDA initialises to make deterministic matrix operations available.
"""
from __future__ import annotations

import numpy as np

from .encoders import fit_projection


class CNN2Encoder:
    def __init__(self, seed: int = 0, d: int = 16, epochs: int = 24, batch: int = 128,
                 members: int = 5, patience: int = 5, device: str = "auto"):
        self.seed, self.d, self.epochs, self.batch = seed, d, epochs, batch
        self.members, self.patience, self.device = members, patience, device

    @staticmethod
    def _targets(x):
        import torch
        q = torch.quantile(x, torch.tensor([.1, .5, .9], device=x.device), dim=2).permute(1, 2, 0)
        diff = torch.mean(torch.abs(x[:, :, 1:] - x[:, :, :-1]), dim=2, keepdim=True)
        median = torch.sort(x, dim=2).values[:, :, (x.shape[2] - 1) // 2:(x.shape[2] - 1) // 2 + 1]
        occupancy = torch.mean((x > median).float(), dim=2, keepdim=True)
        return torch.cat((q, diff, occupancy), dim=2).flatten(1)

    def _build(self, channels, target_dim):
        import torch.nn as nn
        net = nn.Sequential(nn.Conv1d(channels, 24, 7, stride=2, padding=3), nn.GELU(),
                            nn.Conv1d(24, 48, 5, stride=2, padding=2), nn.GELU(),
                            nn.Conv1d(48, 64, 3, stride=1, padding=1), nn.GELU(),
                            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, self.d))
        head = nn.Sequential(nn.Linear(self.d, 64), nn.GELU(), nn.Linear(64, target_dim))
        return net, head

    def _prep(self, windows, cond):
        x = np.asarray(windows, dtype=np.float32)
        if cond is not None:
            c = np.asarray(cond, dtype=np.float32)
            if c.ndim != 3 or c.shape[0] != len(x) or c.shape[2] != x.shape[2]:
                raise ValueError("cond must have shape [N, channels, K] aligned with windows")
            x = np.concatenate((x, c), axis=1)
        return x

    def _fit_normalizer(self, x, split):
        """Fit default CNN2 normalization on the chronological fit prefix."""
        channels = x.shape[1]
        analog = x[:split, :self.analog_channels]
        self.center = np.median(analog, axis=(0, 2), keepdims=True).astype(np.float32)
        q25, q75 = np.percentile(analog, [25, 75], axis=(0, 2), keepdims=True)
        self.scale = np.maximum((q75 - q25) / 1.349, 1e-6).astype(np.float32)
        self.cond_center = (np.mean(x[:split, self.analog_channels:], axis=(0, 2), keepdims=True).astype(np.float32)
                            if channels > self.analog_channels else np.zeros((1, 0, 1), np.float32))
    def _scaled(self, x):
        analog = (x[:, :self.analog_channels] - self.center) / self.scale
        if self.analog_channels < x.shape[1]:
            states = x[:, self.analog_channels:] - self.cond_center
            return np.concatenate((analog, states), axis=1).astype(np.float32, copy=False)
        return analog.astype(np.float32, copy=False)

    @staticmethod
    def _view(x, shifts, jitter):
        import torch
        rolled = torch.roll(x, shifts=int(shifts), dims=2)
        return rolled + torch.randn_like(rolled) * jitter

    @staticmethod
    def _contrastive(a, b):
        import torch
        z = torch.nn.functional.normalize(torch.cat((a, b), dim=0), dim=1)
        logits = z @ z.T / 0.15
        n = len(a)
        logits.fill_diagonal_(-1e9)
        labels = torch.cat((torch.arange(n, 2*n, device=a.device), torch.arange(n, device=a.device)))
        return torch.nn.functional.cross_entropy(logits, labels)

    def fit(self, train_windows, noise_lag: int, groups: np.ndarray | None, cond=None):
        import time
        import torch
        started = time.perf_counter()
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device)
        x = self._prep(train_windows, cond)
        if len(x) < 4:
            raise ValueError("CNN2Encoder requires at least four fit windows")
        n, channels, k = x.shape
        self.analog_channels = np.asarray(train_windows).shape[1]
        # Hold out the final chronological 15% before estimating input scales.
        split = min(max(int(n * .85), 2), n - 1)
        self._fit_normalizer(x, split)

        xs = self._scaled(x)
        xt = torch.from_numpy(xs).to(self.device)
        train = xt[:split]
        val = xt[split:]
        member_nets = []
        latent_members = []
        val_losses = []
        member_times = []
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        rng_devices = list(range(torch.cuda.device_count())) if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=rng_devices):
            for m in range(self.members):
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                member_started = time.perf_counter()
                seed = self.seed + 104729 * m
                torch.manual_seed(seed)
                net, head = self._build(channels, self.analog_channels * 5)
                net, head = net.to(self.device), head.to(self.device)
                opt = torch.optim.AdamW(list(net.parameters()) + list(head.parameters()), lr=8e-4, weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=.5, patience=2)
                gen = torch.Generator().manual_seed(seed + 1)
                best, best_state, stale = float("inf"), None, 0
                for epoch in range(self.epochs):
                    net.train(); head.train()
                    order = torch.randperm(split, generator=gen)
                    for ix in order.split(self.batch):
                        xb = train[ix]
                        shifts = torch.randint(0, k, (2,), generator=gen)
                        a, b = self._view(xb, shifts[0].item(), .025), self._view(xb, shifts[1].item(), .025)
                        za, zb = net(a), net(b)
                        target = self._targets(xb[:, :self.analog_channels])
                        loss = self._contrastive(za, zb) + .35 * torch.nn.functional.smooth_l1_loss(
                            (head(za) + head(zb)) * .5, target)
                        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                    net.eval(); head.eval()
                    with torch.no_grad():
                        vv = val
                        za = net(self._view(vv, 0, .025)); zb = net(self._view(vv, max(1, k//3), .025))
                        vloss = (self._contrastive(za, zb) + .35 * torch.nn.functional.smooth_l1_loss(
                            (head(za)+head(zb))*.5, self._targets(vv[:, :self.analog_channels]))).item()
                    scheduler.step(vloss)
                    if vloss < best - 1e-5:
                        best, stale = vloss, 0
                        best_state = ({k: v.detach().clone() for k, v in net.state_dict().items()},
                                      {k: v.detach().clone() for k, v in head.state_dict().items()})
                    else:
                        stale += 1
                        if stale >= self.patience: break
                net.load_state_dict(best_state[0]); head.load_state_dict(best_state[1]); net.eval()
                member_nets.append(net)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                member_times.append(time.perf_counter() - member_started)
                val_losses.append(best)
                with torch.no_grad():
                    latent_members.append(net(xt).cpu().numpy().astype(np.float64))
        self.member_training_seconds = np.asarray(member_times, dtype=np.float64)
        self.validation_loss = float(np.mean(val_losses))
        self.member_cka = self._cka_matrix(latent_members)
        means = (self.member_cka.sum(axis=1) - 1) / max(1, self.members - 1)
        self.medoid = int(np.argmax(means))
        self.enc = member_nets[self.medoid]
        del member_nets
        self.whitener, self.noise, self.sigma0 = fit_projection(latent_members[self.medoid], self.d, noise_lag, groups)
        self.fit_seconds = time.perf_counter() - started
        return self

    @staticmethod
    def _cka_matrix(latents):
        def cka(a, b):
            a, b = a - a.mean(0), b - b.mean(0)
            cross = a.T @ b
            ga, gb = a.T @ a, b.T @ b
            return float((cross * cross).sum() / np.sqrt((ga * ga).sum() * (gb * gb).sum() + 1e-30))
        out = np.eye(len(latents), dtype=np.float64)
        for i in range(len(latents)):
            for j in range(i): out[i, j] = out[j, i] = cka(latents[i], latents[j])
        return out

    def _latent(self, windows, cond=None):
        import torch
        x = self._prep(windows, cond)
        if x.shape[1] != self.analog_channels + (self.cond_center.shape[1] if cond is not None else 0):
            raise ValueError("cond channel count differs from fit")
        xt = torch.from_numpy(self._scaled(x)).to(self.device)
        out = []
        with torch.no_grad():
            for i in range(0, len(xt), 1024):
                out.append(self.enc(xt[i:i+1024]).cpu().numpy())
        return np.concatenate(out).astype(np.float64)

    def encode(self, windows, sequential: bool = True, cond=None):
        return self.noise.transform(self.whitener.transform(self._latent(windows, cond)))
