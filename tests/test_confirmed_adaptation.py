import numpy as np
import pytest

from uengram.confirmed_adaptation import (
    AdaptiveConformalCalibrator,
    ConfirmedLatentAdapter,
)


def test_latent_adapter_requires_all_operator_and_safety_gates():
    adapter = ConfirmedLatentAdapter(d=2, max_offset=1.0, min_quarantine_count=2)
    latent = np.array([[4.0, -3.0], [6.0, -5.0]])
    original = latent.copy()
    assert adapter.update(
        latent, [0, 0], operator_confirmed_normal=False,
        alarm_closed=True, commissioned=True
    ) == 0
    assert adapter.update(
        latent, [0, 0], operator_confirmed_normal=True,
        alarm_closed=False, commissioned=True
    ) == 0
    assert adapter.update(
        latent, [0, 0], operator_confirmed_normal=True,
        alarm_closed=True, commissioned=False
    ) == 0
    assert adapter.state == {}
    np.testing.assert_array_equal(adapter.apply(latent, [0, 0]), latent)
    np.testing.assert_array_equal(latent, original)


def test_latent_adapter_updates_only_after_quarantine_and_bounds_offset():
    adapter = ConfirmedLatentAdapter(
        d=2, max_offset=1.0, min_quarantine_count=3
    )
    samples = np.array([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
    assert adapter.update(
        samples[:2], [4, 4], operator_confirmed_normal=True,
        alarm_closed=True, commissioned=True
    ) == 0
    assert 4 not in adapter.state
    assert adapter.update(
        samples[2:], [4], operator_confirmed_normal=True,
        alarm_closed=True, commissioned=True
    ) == 3
    np.testing.assert_allclose(adapter.state[4], [1.0, 0.0])
    corrected = adapter.apply(samples, [4, 4, 4])
    np.testing.assert_allclose(corrected, [[1.0, 0.0]] * 3)
    np.testing.assert_allclose(samples, [[2.0, 0.0]] * 3)


def test_latent_adapter_is_state_specific_and_rollback_restores_snapshot():
    adapter = ConfirmedLatentAdapter(
        d=2, max_offset=2.0, min_quarantine_count=2
    )
    adapter.update(
        [[1.0, 0.0], [1.0, 0.0]], [0, 0],
        operator_confirmed_normal=True, alarm_closed=True, commissioned=True
    )
    snapshot = adapter.snapshot()
    adapter.update(
        [[0.0, 1.0], [0.0, 1.0]], [1, 1],
        operator_confirmed_normal=True, alarm_closed=True, commissioned=True
    )
    assert set(adapter.state) == {0, 1}
    adapter.rollback(snapshot)
    assert set(adapter.state) == {0}
    np.testing.assert_allclose(adapter.state[0], [1.0, 0.0])


def test_adapter_shadow_disagreement_rolls_back_only_when_limit_is_exceeded():
    adapter = ConfirmedLatentAdapter(
        d=1, max_offset=1.0, min_quarantine_count=1
    )
    snapshot = adapter.snapshot()
    adapter.update(
        [[0.5]], [2], operator_confirmed_normal=True,
        alarm_closed=True, commissioned=True,
    )
    assert adapter.rollback_if_disagreement(
        snapshot, [1.0, 2.0], [1.05, 2.05],
        max_abs_difference=0.1,
    ) is False
    assert 2 in adapter.state
    assert adapter.rollback_if_disagreement(
        snapshot, [1.0, 2.0], [1.5, 2.5],
        max_abs_difference=0.1,
    ) is True
    assert adapter.state == {}


def test_adaptive_calibrator_uses_state_thresholds_and_global_fallback():
    calibrator = AdaptiveConformalCalibrator(
        calibration_scores=[1.0, 2.0, 3.0, 10.0],
        calibration_states=[0, 0, 0, 1], alpha=0.25,
        min_support=3, window_size=20,
    )
    assert calibrator.threshold(0) == 3.0
    # State 1 has insufficient support and therefore uses the global sample.
    assert calibrator.threshold(1) == 10.0
    assert calibrator.threshold(7) == 10.0


def test_adaptive_calibrator_rejects_unconfirmed_or_open_alarm_samples():
    calibrator = AdaptiveConformalCalibrator(
        [1.0, 2.0, 3.0], [0, 0, 0], alpha=0.25,
        min_support=2, window_size=10,
    )
    before_global = calibrator.global_scores
    before_state = calibrator.state
    assert calibrator.update(
        [100.0], [0], operator_confirmed_normal=False, alarm_closed=True,
        commissioned=True,
    ) == 0
    assert calibrator.update(
        [100.0], [0], operator_confirmed_normal=True, alarm_closed=False,
        commissioned=True,
    ) == 0
    assert calibrator.global_scores == before_global
    assert calibrator.state == before_state
    assert calibrator.update(
        [100.0], [0], operator_confirmed_normal=True, alarm_closed=True,
        commissioned=True,
    ) == 1
    assert calibrator.threshold(0) == 100.0


def test_adaptive_calibrator_requires_explicit_normal_calibration():
    with pytest.raises(ValueError, match="non-empty finite"):
        AdaptiveConformalCalibrator([], [], alpha=0.01)
    with pytest.raises(ValueError, match="integer"):
        AdaptiveConformalCalibrator([1.0], [0.5], alpha=0.01)
    with pytest.raises(ValueError, match="boolean"):
        AdaptiveConformalCalibrator([1.0], [0], alpha=0.01).update(
            [2.0], [0], operator_confirmed_normal=1, alarm_closed=True,
            commissioned=True,
        )


def test_latent_adapter_vector_commissioning_gates_mixed_states():
    adapter = ConfirmedLatentAdapter(
        d=1, max_offset=1.0, min_quarantine_count=2
    )
    changed = adapter.update(
        [[2.0], [3.0]], [0, 1], operator_confirmed_normal=True,
        alarm_closed=True, commissioned=[True, False],
    )
    assert changed == 0
    assert adapter.state == {}
    assert adapter.update(
        [[2.0]], [0], operator_confirmed_normal=True,
        alarm_closed=True, commissioned=True,
    ) == 2
    assert set(adapter.state) == {0}
    np.testing.assert_allclose(adapter.state[0], [1.0])


def test_adaptive_calibrator_prefers_supported_state_and_season_stratum():
    calibrator = AdaptiveConformalCalibrator(
        calibration_scores=[1, 2, 3, 20, 30, 40],
        calibration_states=[0, 0, 0, 0, 0, 0],
        calibration_seasons=["winter", "winter", "winter", "summer", "summer", "summer"],
        alpha=0.25, min_support=3, window_size=20,
    )
    assert calibrator.threshold(0, season="winter") == 3.0
    assert calibrator.threshold(0, season="summer") == 40.0
    # Unknown or undersupported season falls back to the supported state pool.
    assert calibrator.threshold(0, season="spring") == 40.0
    assert calibrator.threshold(1, season="winter") == 40.0


def test_adaptive_calibrator_updates_only_confirmed_state_season():
    calibrator = AdaptiveConformalCalibrator(
        calibration_scores=[1, 2, 3, 20, 30, 40],
        calibration_states=[0, 0, 0, 0, 0, 0],
        calibration_seasons=["winter", "winter", "winter", "summer", "summer", "summer"],
        alpha=0.25, min_support=3, window_size=20,
    )
    before = calibrator.threshold(0, season="winter")
    assert calibrator.update(
        [100], [0], seasons=["winter"],
        operator_confirmed_normal=False, alarm_closed=True, commissioned=True,
    ) == 0
    assert calibrator.threshold(0, season="winter") == before
    assert calibrator.update(
        [100], [0], seasons=["winter"],
        operator_confirmed_normal=True, alarm_closed=True, commissioned=True,
    ) == 1
    assert calibrator.threshold(0, season="winter") == 100.0


def test_calibrator_does_not_learn_from_uncommissioned_state_rows():
    calibrator = AdaptiveConformalCalibrator(
        [1.0, 2.0, 3.0], [0, 0, 0], alpha=0.25,
        min_support=2, window_size=10,
    )
    before = calibrator.global_scores
    assert calibrator.update(
        [100.0, 200.0], [0, 1],
        operator_confirmed_normal=True, alarm_closed=True,
        commissioned=[True, False],
    ) == 1
    assert calibrator.global_scores == (1.0, 2.0, 3.0, 100.0)
    assert calibrator.global_scores != before
    assert 1 not in calibrator.state

def test_calibrator_bounds_memory_across_state_and_season_churn():
    calibrator = AdaptiveConformalCalibrator(
        [1.0, 2.0, 3.0], [0, 0, 0], calibration_seasons=["a", "b", "c"],
        alpha=0.25, min_support=2, window_size=3,
    )
    assert calibrator.update(
        [4.0, 5.0, 6.0, 7.0], [1, 2, 3, 4],
        seasons=["d", "e", "f", "g"], operator_confirmed_normal=True,
        alarm_closed=True, commissioned=True,
    ) == 4
    assert calibrator.global_scores == (5.0, 6.0, 7.0)
    assert set(calibrator.state) == {2, 3, 4}
    assert len(calibrator.strata) == 3
