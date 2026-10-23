import numpy as np
import pytest

from bench.modality_alarm import (ModalityConfig, calibrate, channel_values, duty_only_prediction,
                                  evaluate, hold, modality_errors)
from bench.streams import COMPRESSOR_FEATURES, DIGITAL, Stream, load_stream
from uengram import metropt

BASE = np.datetime64("2020-01-01T00:00:00", "s")


def _stream(n=600, state=None, cal_windows=300):
    """Synthetic compressor stream: calibration prefix, then test; all states idle unless set."""
    start = BASE + np.arange(n) * np.timedelta64(600, "s")
    state = np.zeros(n, int) if state is None else np.asarray(state)
    features = np.random.default_rng(0).normal(5.0, 0.1, (n, len(COMPRESSOR_FEATURES)))
    cal = np.zeros(n, bool)
    cal[:cal_windows] = True
    W = np.full((n, len(metropt.ANALOG), 60), 5.0)
    cond = np.zeros((n, len(DIGITAL), 60))
    return Stream("synthetic", W, start, start + np.timedelta64(600, "s"), np.zeros(n, int), state,
                  np.ones(n, bool), np.zeros(n, bool), cal, ~cal, (), 0, features,
                  COMPRESSOR_FEATURES, cond=cond)


def test_hold_requires_consecutive_condition_for_k_windows():
    condition = np.array([1, 1, 0, 1, 1, 1, 1, 0, 0], bool)
    active, opened = hold(condition, np.ones(9, bool), k_hold=3, k_release=1)
    assert opened.tolist() == [0, 0, 0, 0, 0, 1, 0, 0, 0]  # the 2-window burst never opens
    assert active.tolist() == [0, 0, 0, 0, 0, 1, 1, 0, 0]  # releases on the first quiet window


def test_hold_release_delay_and_gap_semantics():
    condition = np.array([1, 1, 0, 1, 0, 0, 0], bool)
    active, _ = hold(condition, np.ones(7, bool), k_hold=2, k_release=2)
    assert active.tolist() == [0, 1, 1, 1, 1, 0, 0]  # one quiet window is not enough to clear
    # A logging gap resets the run (no opening across it) but does not release an open alarm.
    gap = np.array([1, 1, 1, 1, 1], bool)
    monitorable = np.array([1, 0, 1, 1, 1], bool)
    assert hold(gap, monitorable, 2, 1)[1].tolist() == [0, 0, 0, 1, 0]
    carried, _ = hold(np.array([1, 1, 1, 1], bool), np.array([1, 1, 0, 1], bool), 2, 1)
    assert carried.tolist() == [0, 1, 1, 1]
    with pytest.raises(ValueError):
        hold(gap, monitorable, 0, 1)


def test_too_small_reference_marks_channels_unjudgeable_not_healthy():
    stream = _stream(cal_windows=300)
    fine = calibrate(stream, ModalityConfig(alpha=0.01, per_state=False))  # 0.01/7 needs 699 windows
    coarse = calibrate(stream, ModalityConfig(alpha=0.1, per_state=False))  # 0.1/7 needs 69
    assert fine[-1]["judgeable"] == []
    assert len(coarse[-1]["judgeable"]) == 7


def test_unbaselined_state_is_not_judged_against_another_modes_range():
    state = np.zeros(600, int)
    state[300:] = 2  # sustained load appears only after calibration
    stream = _stream(state=state)
    stream.features[300:, COMPRESSOR_FEATURES.index("oil_temp_c")] = 80.0  # far outside idle range
    test = np.flatnonzero(stream.test)
    for policy, expect in (("ignore", False), ("global", True)):
        cfg = ModalityConfig(alpha=0.1, unbaselined=policy, duty=False)
        errors, names = modality_errors(stream, test, cfg, calibrate(stream, cfg))
        assert bool(errors[:, names.index("oil_temp_c")].any()) is expect


def test_loaded_conditioning_uses_only_loaded_samples():
    stream = _stream()
    tp3 = list(metropt.ANALOG).index("TP3")
    stream.W[:, tp3, :] = 9.0  # unloaded samples
    stream.W[0, tp3, :10] = 7.0  # loaded samples of window 0
    stream.cond[0, DIGITAL.index("DV_eletric"), :10] = 1
    stream.cond[1, DIGITAL.index("DV_eletric"), :2] = 1  # only 2 loaded samples
    values = channel_values(stream, ModalityConfig(conditioning="loaded", channels=("tp3_bar",)))
    assert values[0, 0] == 7.0
    assert np.isnan(values[1, 0]) and np.isnan(values[2, 0])


def test_requiring_two_modalities_suppresses_single_modality_runs():
    state = np.zeros(600, int)
    state[400:430] = 2  # 30-window sustained-load run, no physical channel out of range
    stream = _stream(state=state)
    one = evaluate(stream, ModalityConfig(alpha=0.1, k_hold=5, min_modalities=1))[1]
    two = evaluate(stream, ModalityConfig(alpha=0.1, k_hold=5, min_modalities=2))[1]
    assert sum(one["opening_modalities"]["clean"].values()) >= 1
    assert sum(two["opening_modalities"]["clean"].values()) == 0


@pytest.mark.parametrize("k_hold", [1, 3, 6, 30, 35, 50])
def test_duty_only_alarm_is_fully_determined_by_sustained_load_run_lengths(k_hold):
    stream = load_stream("metropt3")
    measured, _ = evaluate(stream, ModalityConfig(channels=(), k_hold=k_hold))
    predicted = duty_only_prediction(stream, k_hold)
    assert predicted["events_detected"] == measured["events_detected"]
    assert predicted["alarm_time_share_clean"] == pytest.approx(measured["alarm_time_share_clean"], abs=1e-12)
    assert predicted["lead_hours"] == pytest.approx(measured["lead_hours"], abs=1e-12)
