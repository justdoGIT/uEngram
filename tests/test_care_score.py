import numpy as np
import pytest

from bench import care_score as cs


def test_criticality_freezes_outside_normal_operation():
    pred = np.array([1, 1, 1, 0, 1, 0, 0, 0], bool)
    normal = np.array([1, 1, 0, 1, 1, 0, 1, 1], bool)
    # +1, +1, (frozen), -1, +1, (frozen), -1, -1
    assert cs.criticality(pred, normal).tolist() == [1, 2, 2, 1, 2, 2, 1, 0]


def test_weighted_score_rewards_early_detection():
    n = 40
    early, late = np.zeros(n, bool), np.zeros(n, bool)
    early[:5], late[-5:] = True, True
    assert cs.weighted_score(np.ones(n, bool)) == pytest.approx(1.0)
    assert cs.weighted_score(early) > 0.15 > cs.weighted_score(late)
    assert cs.weighted_score(np.zeros(n, bool)) == 0.0


def test_event_detection_counts_only_criticality_before_event_end():
    n = 200
    normal = np.ones(n, bool)
    in_event = np.zeros(n, bool)
    in_event[100:150] = True
    pred = np.zeros(n, bool)
    pred[150:] = True  # alarms only after the fault started: too late
    late = cs.evaluate_event(pred, normal, in_event, 149, "anomaly")
    assert not late["detected"] and late["max_criticality"] == 0
    pred = np.zeros(n, bool)
    pred[70:150] = True  # 80 consecutive detections up to the event end
    early = cs.evaluate_event(pred, normal, in_event, 149, "anomaly")
    assert early["detected"] and early["max_criticality"] == 80
    # Coverage: 50 true positives in the event, 30 early false positives.
    assert (early["tp"], early["fp"], early["fn"]) == (50, 30, 0)
    assert early["f_beta"] == pytest.approx(1.25 * 50 / (1.25 * 50 + 30))



def test_event_end_is_inclusive_for_criticality():
    # The 72nd qualifying prediction is exactly at end_pos; later rows do not
    # contribute to the criticality decision.
    predicted = np.zeros(75, dtype=bool)
    predicted[1:73] = True
    normal = np.ones(75, dtype=bool)
    in_event = np.zeros(75, dtype=bool)
    in_event[73:] = True

    at_end = cs.evaluate_event(predicted, normal, in_event, 72, "anomaly")
    after_end = cs.evaluate_event(predicted, normal, in_event, 71, "anomaly")

    assert at_end["max_criticality"] == 72 and at_end["detected"]
    assert after_end["max_criticality"] == 71 and not after_end["detected"]


def test_normal_event_false_alarms_are_penalized_only_on_normal_status():
    predicted = np.array([True, True, True, False, True, True])
    normal = np.array([True, False, False, True, True, False])
    in_event = np.zeros(len(predicted), dtype=bool)

    result = cs.evaluate_event(predicted, normal, in_event, 5, "normal")

    # Two of the three normal-operation points are false alarms. Predictions
    # during non-normal status are excluded from normal-event accuracy.
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (0, 2, 0, 1)
    assert result["accuracy"] == pytest.approx(1 / 3)
    assert np.isnan(result["f_beta"]) and np.isnan(result["weighted_score"])


def test_all_normal_and_all_anomaly_event_ground_truth_cases():
    all_normal = cs.evaluate_event(
        np.array([False, True, False]), np.ones(3, dtype=bool),
        np.zeros(3, dtype=bool), 2, "normal",
    )
    all_anomaly = cs.evaluate_event(
        np.array([True, True, False]), np.ones(3, dtype=bool),
        np.ones(3, dtype=bool), 2, "anomaly",
    )

    assert (all_normal["tp"], all_normal["fp"], all_normal["fn"], all_normal["tn"]) == (0, 1, 0, 2)
    assert all_normal["accuracy"] == pytest.approx(2 / 3)
    assert (all_anomaly["tp"], all_anomaly["fp"], all_anomaly["fn"], all_anomaly["tn"]) == (2, 0, 1, 0)
    assert all_anomaly["f_beta"] == pytest.approx(10 / 11)

def test_final_score_special_cases_and_weighting():
    a = {"label": "anomaly", "f_beta": 0.6, "weighted_score": 0.4, "detected": True}
    nrm = {"label": "normal", "accuracy": 0.9, "detected": False}
    out = cs.final_score([a, nrm])
    assert out["reliability"] == pytest.approx(1.0)
    assert out["care"] == pytest.approx((0.6 + 2 * 0.9 + 0.4 + 1.0) / 5)
    assert cs.final_score([dict(a, detected=False), nrm])["care"] == 0.0
    assert cs.final_score([a, dict(nrm, accuracy=0.4)])["care"] == pytest.approx(0.4)
    with pytest.raises(ValueError):
        cs.final_score([a])


def test_event_without_eligible_operating_rows_is_not_a_zero_accuracy_result():
    with pytest.raises(ValueError, match="no normal-status rows"):
        cs.evaluate_event(np.zeros(3, bool), np.zeros(3, bool), np.ones(3, bool), 2, "anomaly")
