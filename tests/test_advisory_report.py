import numpy as np
import pytest

from bench.advisory_report import notification_rate, verify_baseline


def test_notifications_include_baseline_alerts_and_merge_nearby_openings():
    end = np.array([0, 600, 7200, 7800, 14400], dtype="datetime64[s]")
    start = end - np.timedelta64(600, "s")
    valid = np.ones(5, bool)
    alerts = np.array([1, 0, 1, 0, 0], bool)
    opened = np.array([0, 1, 0, 0, 1], bool)
    # First alert + nearby opening are one notification episode, not two.
    assert notification_rate(start, end, valid, alerts, opened) == pytest.approx(3 * 604800 / 3000)


def test_notification_denominator_uses_observed_time_and_excludes_unscored_rows():
    end = np.array([600, 1200, 7200], dtype="datetime64[s]")
    start = end - np.timedelta64(600, "s")
    valid = np.array([1, 0, 1], bool)
    alerts = np.array([0, 1, 0], bool)
    opened = np.array([1, 0, 0], bool)
    assert notification_rate(start, end, valid, alerts, opened) == pytest.approx(604800 / 1200)


def test_reconstructed_baseline_must_match_notification_burden():
    with pytest.raises(ValueError, match="notifications_per_week_all"):
        verify_baseline({"notifications_per_week_all": 17.3}, {"notifications_per_week_all": 0.0})
