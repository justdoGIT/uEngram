from __future__ import annotations

import numpy as np

from bench.capsule_warmup_ablation import warmup_ablation
from uengram import SensorWindow


def _drifting_windows(count=48):
    """Stable baseline for 24 windows, then a steady drift away from it."""
    rng = np.random.default_rng(0)
    rows = []
    for step in range(count):
        drift = max(0, step - 24) * 0.5
        rows.append(SensorWindow("a", "s", "steady", step, 1_900_000_000_000 + step * 1000, step,
                                 rng.normal(0, 0.1, 4) + np.array([drift, 0, 0, 0]), "v1"))
    return tuple(rows)


def test_replaying_the_whole_pre_restart_history_reproduces_the_reference_records():
    windows = _drifting_windows()
    restart = 32

    result = warmup_ablation(windows, restart, [0, restart])
    cold, full = result["arms"]

    # Full history is the uninterrupted run, so its post-restart records must equal the reference's.
    assert full["keys_equal_reference_after_restart"]
    assert full["records_after_restart"] == result["reference"]["records_after_restart"]
    assert full["first_flagged_step"] == result["reference"]["first_flagged_step"]
    # A cold restart re-baselines on the already-drifted data and does not reproduce them.
    assert not cold["keys_equal_reference_after_restart"]
