from __future__ import annotations

import numpy as np

from bench.cmapss_two_teacher_gate import _alarm_metrics


def test_alarm_metrics_requires_three_consecutive_flags() -> None:
    scores = np.array([0.0, 2.0, 2.0, 0.0, 2.0, 2.0], dtype=float)
    units = np.full(6, 41)
    rul = np.array([30.0, 20.0, 19.0, 18.0, 17.0, 16.0])

    result = _alarm_metrics(scores, units, rul, threshold=1.0)

    assert result["events_detected_before_failure"] == 0
    assert result["missed_events"] == 1
    assert result["pre_eol_alarm_episodes"] == 0


def test_alarm_metrics_separates_early_warning_from_false_alarm() -> None:
    scores = np.array([2.0, 2.0, 2.0, 0.0, 0.0], dtype=float)
    units = np.full(5, 41)
    rul = np.array([40.0, 39.0, 38.0, 37.0, 36.0])

    result = _alarm_metrics(scores, units, rul, threshold=1.0)

    assert result["events_detected_before_failure"] == 1
    assert result["event_recall_at_eol_horizon"] == 0.0
    assert result["pre_eol_alarm_episodes"] == 1
    assert result["warning_cycles_before_eol_horizon_mean"] == 20.0


def test_alarm_metrics_counts_sustained_alarm_at_horizon() -> None:
    scores = np.array([0.0, 0.0, 2.0, 2.0, 2.0], dtype=float)
    units = np.full(5, 41)
    rul = np.array([30.0, 22.0, 20.0, 19.0, 18.0])

    result = _alarm_metrics(scores, units, rul, threshold=1.0)

    assert result["events_detected_before_failure"] == 1
    assert result["pre_eol_alarm_episodes"] == 0
    assert result["event_recall_at_eol_horizon"] == 1.0
