import numpy as np
import pytest

from uengram.role_architecture import (
    EmpiricalCalibrator,
    RoleBasedArchitecture,
    RoleScores,
    persistent_starts,
)


def _windows(seed=0, n=80, channels=4, length=16):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, channels, length)).astype(np.float32)


def test_role_scores_keep_level_dynamic_health_and_mode_separate():
    fit = _windows(1)
    model = RoleBasedArchitecture(d=8).fit(fit, noise_lag=1)
    scores = model.score(_windows(2, n=12))

    assert isinstance(scores, RoleScores)
    assert scores.level.shape == (12,)
    assert scores.dynamic.shape == (12,)
    assert scores.health.shape == (12,)
    assert scores.mode.shape == (12,)
    assert np.isfinite(scores.level).all()
    assert np.isfinite(scores.dynamic).all()
    assert np.isfinite(scores.health).all()
    assert not np.array_equal(scores.level, scores.dynamic)


def test_empirical_calibration_is_monotone_and_rejects_empty_input():
    calibrator = EmpiricalCalibrator().fit(np.arange(20, dtype=float))
    p = calibrator.p_values(np.array([0.0, 10.0, 19.0]))
    assert np.all(np.diff(p) <= 0)
    assert p[0] > p[-1]
    with pytest.raises(ValueError, match="finite calibration"):
        EmpiricalCalibrator().fit(np.array([np.nan, np.inf]))


def test_fusion_excludes_mode_and_requires_calibrated_evidence():
    model = RoleBasedArchitecture(d=8).fit(_windows(3), noise_lag=1)
    model.fit_calibration(_windows(4, n=40))
    p = model.p_values(model.score(_windows(5, n=6)))
    changed_mode = dict(p)
    changed_mode["mode"] = 1.0 - changed_mode["mode"]
    assert np.array_equal(
        model.fuse(p, alpha=0.2, policy="dynamic_health"),
        model.fuse(changed_mode, alpha=0.2, policy="dynamic_health"),
    )
    uncalibrated = RoleBasedArchitecture(d=8).fit(_windows(6))
    with pytest.raises(RuntimeError, match="fit_calibration"):
        uncalibrated.p_values(uncalibrated.score(_windows(7, n=6)))


def test_memory_route_quarantines_unconfirmed_normal_and_alarm_data():
    model = RoleBasedArchitecture(d=8)
    p = {"level": 0.9, "dynamic": 0.9, "health": 0.9, "mode": 0.9}
    assert model.memory_route(p, alpha=0.01, alarm_open=False, data_quality_ok=True) == "QUARANTINE"
    assert model.memory_route(p, alpha=0.01, alarm_open=False, data_quality_ok=True,
                              operator_confirmed_normal=True) == "SEMANTIC"
    assert model.memory_route(p | {"dynamic": 0.001}, alpha=0.01,
                              alarm_open=False, data_quality_ok=True) == "EPISODIC"
    assert model.memory_route(p, alpha=0.01, alarm_open=True, data_quality_ok=True) == "EPISODIC"
    assert model.memory_route(p, alpha=0.01, alarm_open=False, data_quality_ok=False) == "QUARANTINE"


def test_persistent_starts_collapse_overlapping_true_windows():
    flags = np.array([False, True, True, True, True, False, True, True, True])
    assert persistent_starts(flags, 3).tolist() == [False, True, False, False, False, False, True, False, False]
    with pytest.raises(ValueError, match="positive"):
        persistent_starts(flags, 0)
