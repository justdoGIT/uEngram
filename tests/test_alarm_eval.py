import numpy as np

from bench.alarm_eval import apply_policy, evaluate_alarms
from bench.streams import Stream
from uengram.alarm import AlarmConfig


def test_unmonitorable_fault_windows_do_not_claim_detection():
    start = np.arange(4).astype("datetime64[h]").astype("datetime64[s]")
    end = start + np.timedelta64(1, "h")
    event_id = np.array([-1, 0, 0, -1])
    valid = np.array([True, False, False, True])
    active = np.array([True, True, True, True])
    opened = np.array([True, False, False, False])
    metrics, details = evaluate_alarms(
        start, end, event_id, np.array([True, False, False, True]), valid,
        [("1970-01-01T01", "1970-01-01T03")], active, opened,
        np.array([5.0, 5.0, 5.0, 0.0]), np.zeros(4))
    assert metrics["events_detected"] == 0
    assert details["events"][0]["alarm_windows"] == 0
    assert details["events"][0]["windows"] == 0
    assert details["eligible_windows"] == 2


def test_alarm_time_share_clean_is_duration_weighted_and_openings_are_episodes():
    start = np.array([0, 1, 2, 4]).astype("datetime64[h]").astype("datetime64[s]")
    end = np.array([1, 2, 4, 5]).astype("datetime64[h]").astype("datetime64[s]")
    valid = np.ones(4, bool)
    negative = np.ones(4, bool)
    active = np.array([True, True, False, True])
    opened = np.array([True, False, False, True])
    metrics, details = evaluate_alarms(
        start, end, np.full(4, -1), negative, valid, [], active, opened,
        np.zeros(4), np.zeros(4), alert_index=(0, 1))
    assert metrics["alarm_time_share_clean"] == 0.6
    assert details["clean_observed_seconds"] == 5 * 3600
    assert details["clean_alarm_seconds"] == 3 * 3600
    assert details["false_alarm_episodes"] == 2
    # The advisory and alarm at the first time are one operator event, not two.
    np.testing.assert_allclose(metrics["operator_events_per_week"], 3 * 168 / 5)


def test_retrospective_evaluation_mask_never_suppresses_runtime_persistence():
    n = 23
    t = (np.datetime64("2023-01-01") + np.arange(n) * np.timedelta64(10, "m")).astype("datetime64[s]")
    cal = np.arange(n) < 20
    s = Stream("case", np.ones((n, 1)), t - np.timedelta64(600, "s"), t,
               np.zeros(n, int), np.zeros(n, int), np.ones(n, bool), np.zeros(n, bool),
               cal, ~cal, ((str(t[21]), str(t[22])),), 4, np.ones((n, 1)), ("pressure",),
               point_labels=True, metric_mask=np.ones(n, bool))
    scores = {"novelty": np.r_[np.ones(20), 4., 4., 0.], "health": np.full(n, np.nan),
              "action": np.zeros(n, int)}
    cfg = AlarmConfig(alpha=.1, k_novelty=2, k_health=0, per_state=False, new_state_alert=False)
    _, before, _ = apply_policy(s, scores, cfg)
    s.metric_mask[20] = False  # A retrospective fault tag on the first exceedance.
    metrics, after, _ = apply_policy(s, scores, cfg)
    assert before["active"].tolist() == after["active"].tolist() == [False, True, True]
    assert metrics["events_detected"] == 1


def test_discarded_memory_arrival_remains_a_scorable_alarm_window():
    from bench.alarm_eval import ACTIONS

    n = 23
    t = (np.arange(n) * 600).astype("datetime64[s]")
    test = np.arange(n) >= 20
    stream = Stream(
        "discard",
        np.ones((n, 1)),
        t - np.timedelta64(600, "s"),
        t,
        np.zeros(n, int),
        np.zeros(n, int),
        np.ones(n, bool),
        np.ones(n, bool),
        ~test,
        test,
        (),
        4,
        np.ones((n, 1)),
        ("pressure",),
    )
    scores = {
        "novelty": np.r_[np.ones(20), 4.0, 4.0, 0.0],
        "health": np.full(n, np.nan),
        "action": np.full(n, ACTIONS.index("discard"), dtype=np.int8),
    }
    metrics, details, _ = apply_policy(
        stream,
        scores,
        AlarmConfig(
            alpha=0.1,
            k_novelty=2,
            k_health=0,
            per_state=False,
            new_state_alert=False,
        ),
    )
    assert details["active"].tolist() == [False, True, True]
    assert details["runtime_alarm_episodes"] == 1


def test_unavailable_alarm_calibration_is_an_error_not_an_all_quiet_detector():
    import pytest
    from bench.alarm_eval import CalibrationUnavailable
    for n, kn, kh in ((4, 1, 0), (300, 0, 1)):
        t = (np.arange(n) * 600).astype("datetime64[s]")
        test = np.arange(n) == n - 1
        s = Stream("case", np.ones((n, 1)), t - np.timedelta64(600, "s"), t,
                   np.zeros(n, int), np.zeros(n, int), np.ones(n, bool), np.zeros(n, bool),
                   ~test, test, (), 4, np.ones((n, 1)), ("pressure",))
        sc = {"novelty": np.ones(n), "health": np.full(n, np.nan), "action": np.zeros(n, int)}
        with pytest.raises(CalibrationUnavailable, match="no finite calibrated threshold"):
            apply_policy(s, sc, AlarmConfig(k_novelty=kn, k_health=kh))
