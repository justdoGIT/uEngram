"""Fixed synthetic-workload benchmark for opt-in encoder research."""
from __future__ import annotations

import numpy as np

from bench.model_selection import rank_candidates
from uengram.encoder_cnn import CNNEncoder
from uengram.encoder_cnn_ssl import SSLTemporalEncoder
from uengram.encoder_multiscale import MultiScaleEncoder
from uengram.encoders import DSPEncoder


SEED = 20261025
WINDOWS = 192
SELECTION_WINDOWS = 48
CHANNELS = 3
LENGTH = 64
LATENT_D = 8
NEURAL_EPOCHS = 3


def normal_windows(n: int, seed: int):
    rng = np.random.default_rng(seed)
    time = np.arange(LENGTH, dtype=np.float64) / LENGTH
    states = rng.integers(0, 2, size=n, dtype=np.int64)
    windows = np.empty((n, CHANNELS, LENGTH), dtype=np.float32)
    for row, state in enumerate(states):
        for channel in range(CHANNELS):
            frequency = channel + 1
            phase = rng.uniform(-np.pi, np.pi)
            amplitude = 0.7 + 0.2 * rng.random()
            level = (state - 0.5) * (0.3 + channel * 0.1)
            signal = level + amplitude * np.sin(2 * np.pi * frequency * time + phase)
            signal += 0.15 * np.sin(2 * np.pi * (frequency + 1) * time)
            signal += rng.normal(0.0, 0.06, size=LENGTH)
            windows[row, channel] = signal
    return windows, states


def main() -> None:
    train, groups = normal_windows(WINDOWS, SEED)
    selection, _ = normal_windows(SELECTION_WINDOWS, SEED + 1)
    candidates = {
        "dsp": DSPEncoder(d=LATENT_D),
        "multiscale": MultiScaleEncoder(d=LATENT_D),
    }
    for name, constructor in (
        ("cnn", CNNEncoder), ("cnnssl", SSLTemporalEncoder),
    ):
        for seed in (0, 1):
            kwargs = {"seed": seed, "d": LATENT_D, "epochs": NEURAL_EPOCHS}
            if name == "cnnssl":
                kwargs.update(batch=64, width=16)
            candidates[f"{name}_seed{seed}"] = constructor(**kwargs)

    for name, encoder in candidates.items():
        print(f"fit_candidate={name}", flush=True)
        encoder.fit(train, noise_lag=1, groups=groups)
    print("stage=label_free_synthetic_selection", flush=True)
    ranking = rank_candidates(candidates, selection, seed=SEED + 2, strength=3.0, quantile=0.1)
    scores = {row["name"]: row["worst_family_score"] for row in ranking}
    for name in ("dsp", "multiscale"):
        print(f"METRIC {name}_min_family_separation={scores[name]:.12g}")
    for family in ("cnn", "cnnssl"):
        seed_scores = [scores[f"{family}_seed{seed}"] for seed in (0, 1)]
        print(f"METRIC {family}_min_seed_separation={min(seed_scores):.12g}")
        print(f"METRIC {family}_seed_spread={max(seed_scores) - min(seed_scores):.12g}")
    print(f"benchmark_winner={ranking[0]['name']}")


if __name__ == "__main__":
    main()
