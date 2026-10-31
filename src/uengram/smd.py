"""Server Machine Dataset (SMD, Su et al., KDD 2019): 28 server machines, 38 channels, 1-minute
sampling, a normal-only training file and a labelled test file per machine.

Source: https://github.com/NetManAIOps/OmniAnomaly (ServerMachineDataset/). The files carry no
timestamps; the documented 1-minute cadence is used to build a synthetic time axis (train then
test, contiguous), so event times are row offsets, not wall-clock times.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np

BASE = "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset"
MACHINES = tuple(f"machine-{g}-{i}" for g, n in ((1, 8), (2, 9), (3, 11)) for i in range(1, n + 1))
DEFAULT_ROOT = Path("data/smd")
CHANNELS = tuple(f"ch{i:02d}" for i in range(38))
EPOCH = np.datetime64("2000-01-01T00:00:00")


def _path(root: Path, kind: str, machine: str) -> Path:
    return Path(root) / kind / f"{machine}.txt"


def fetch(root: str | Path = DEFAULT_ROOT, machines: tuple[str, ...] = MACHINES) -> Path:
    """Download any missing train/test/test_label files (about 470 MB for all 28 machines)."""
    root = Path(root)
    for machine in machines:
        for kind in ("train", "test", "test_label"):
            dest = _path(root, kind, machine)
            if dest.exists() and dest.stat().st_size > 0:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_suffix(".part")
            urllib.request.urlretrieve(f"{BASE}/{kind}/{machine}.txt", part)
            os.replace(part, dest)
            print(f"[smd] {kind}/{machine}.txt {dest.stat().st_size // 1024} KB", flush=True)
    return root


def load(machine: str, root: str | Path = DEFAULT_ROOT):
    """Returns (train[Ntr,38], test[Nte,38], label[Nte] bool)."""
    train = np.loadtxt(_path(Path(root), "train", machine), delimiter=",", dtype=np.float32)
    test = np.loadtxt(_path(Path(root), "test", machine), delimiter=",", dtype=np.float32)
    label = np.loadtxt(_path(Path(root), "test_label", machine), dtype=np.int8).astype(bool)
    if train.shape[1] != len(CHANNELS) or test.shape[1] != len(CHANNELS) or len(label) != len(test):
        raise ValueError(f"{machine}: unexpected shapes {train.shape}, {test.shape}, {label.shape}")
    return train, test, label


def label_runs(label: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of True as inclusive (first_row, last_row) pairs."""
    padded = np.r_[False, label.astype(bool), False]
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [(int(a), int(b) - 1) for a, b in zip(edges[::2], edges[1::2])]
