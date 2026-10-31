import types

import numpy as np
import pytest

from bench.alarm_eval import apply_policy
from bench.alarm_sensitivity import BASELINE
from bench.error_gated_alarm import Gate, Replay, _departs_from_calibration, frozen_scores_available, gate_alarm, latch
from bench.streams import load_stream

T0 = np.datetime64("2020-01-01T00:00:00", "s")


def _ends(n, step_min=10):
    return T0 + np.arange(n) * np.timedelta64(step_min * 60, "s")


def test_latch_same_window_and_hour_lookback():
    errors = np.array([[0], [1], [0], [0], [0], [0], [0], [0]], bool)
    valid = np.ones(8, bool)
    assert latch(errors, _ends(8), valid, 0.0)[:, 0].tolist() == [0, 1, 0, 0, 0, 0, 0, 0]
    # 10-minute windows: an error at window 1 still counts 30 min later (window 4), not 40.
    assert latch(errors, _ends(8), valid, 0.5)[:, 0].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]


def test_latch_ignores_ineligible_errors_and_never_marks_ineligible_windows():
    errors = np.array([[1], [0], [1], [0]], bool)
    valid = np.array([0, 1, 1, 0], bool)
    assert latch(errors, _ends(4), valid, 24.0)[:, 0].tolist() == [0, 0, 1, 0]


def test_gated_episode_opens_on_rising_edge_and_survives_logging_gap():
    active = np.array([1, 1, 1, 1, 1, 1], bool)
    condition = np.array([1, 1, 0, 1, 1, 1], bool)
    valid = np.array([1, 1, 1, 1, 0, 1], bool)
    gated, opened = gate_alarm(active, valid, condition)
    assert gated.tolist() == [1, 1, 0, 1, 0, 1]
    assert opened.tolist() == [1, 0, 0, 1, 0, 0]  # the gap at 4 neither closes nor reopens


def test_flag_healthy_value_comes_from_calibration_not_documented_polarity():
    stream = types.SimpleNamespace(cal=np.array([1, 1, 1, 0, 0, 0], bool), monitorable=np.ones(6, bool))
    active_low = np.array([[1, 1], [1, 1], [1, 1], [1, 1], [1, 0], [0, 0]], float)
    assert _departs_from_calibration(active_low, stream).tolist() == [0, 0, 0, 0, 1, 1]
    constant = np.zeros((6, 2))
    assert not _departs_from_calibration(constant, stream).any()


@pytest.fixture(scope="module")
def metropt3():
    if not frozen_scores_available("metropt3"):
        pytest.skip("frozen metropt3 scores not cached locally")
    return Replay(load_stream("metropt3"))


def test_always_true_gate_reproduces_the_actual_alarm(metropt3):
    reference, _, _ = apply_policy(metropt3.stream, metropt3.scores, BASELINE)
    ungated, _ = metropt3.evaluate(BASELINE, None)
    for key in ("events_detected", "false_alarm_per_week", "alarm_time_share_clean", "lead_hours",
                "notifications_per_week_all"):
        assert ungated[key] == pytest.approx(reference[key], nan_ok=True)


@pytest.mark.parametrize("gate", [Gate("dataset", 1, 0.0), Gate("dataset", 1, 24.0),
                                  Gate("calibrated", 2, 3.0)])
def test_error_gate_only_removes_alarm_time(metropt3, gate):
    policy, _ = metropt3.evaluate(BASELINE, None)
    gated, details = metropt3.evaluate(BASELINE, gate)
    assert gated["alarm_time_share_clean"] <= policy["alarm_time_share_clean"]
    assert gated["events_detected"] <= min(policy["events_detected"], details["error_ceiling_events"])


@pytest.mark.parametrize("lookback_h", [0.0, 3.0, 24.0])
def test_metropt3_may_29_failure_has_no_dataset_error_to_gate_on(metropt3, lookback_h):
    s, test = metropt3.stream, metropt3.test
    valid = metropt3.policy(BASELINE)[1]["valid"]
    condition = metropt3.condition(Gate("dataset", 1, lookback_h), valid)
    assert not (condition & (s.event_id()[test] == 1)).any()
