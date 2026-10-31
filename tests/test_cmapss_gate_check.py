import json
from pathlib import Path

import pytest

from bench.cmapss_gate_check import evaluate_report

ARTIFACT = Path(__file__).parents[1] / "results" / "cmapss_two_teacher_gate.json"


def _report():
    return json.loads(ARTIFACT.read_text())


def test_recorded_baseline_meets_the_frozen_cmapss_gate():
    outcome = evaluate_report(_report(), model="baseline_cnn")

    assert outcome["passed"] is True
    assert outcome["metrics"]["event_recall_at_eol_horizon"] == pytest.approx(0.70)
    assert outcome["metrics"]["calibration_false_alarm_episodes"] == 0
    assert outcome["metrics"]["pre_eol_alarm_episodes"] == 7


def test_gate_accepts_inclusive_boundaries():
    report = {
        "protocol": {"eol_horizon_cycles": 20, "persistence_windows": 3},
        "results": {
            "candidate": {
                "event_recall_at_eol_horizon": 0.70,
                "calibration_false_alarm_episodes": 0,
                "pre_eol_alarm_episodes": 7,
            }
        },
    }

    assert evaluate_report(report, model="candidate")["passed"] is True


def test_gate_reports_each_failed_acceptance_condition():
    report = {
        "protocol": {"eol_horizon_cycles": 20, "persistence_windows": 3},
        "results": {
            "candidate": {
                "event_recall_at_eol_horizon": 0.69,
                "calibration_false_alarm_episodes": 1,
                "pre_eol_alarm_episodes": 8,
            }
        },
    }

    outcome = evaluate_report(report, model="candidate")

    assert outcome["passed"] is False
    assert outcome["failures"] == [
        "event_recall_at_eol_horizon < 0.7",
        "calibration_false_alarm_episodes != 0",
        "pre_eol_alarm_episodes > 7",
    ]
