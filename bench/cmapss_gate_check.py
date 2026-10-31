"""Validate the frozen CMAPSS lifecycle acceptance gate.

The gate is intentionally separate from model fitting. It checks a recorded
report without selecting a model from test labels or changing the protocol.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MIN_EVENT_RECALL_AT_EOL = 0.70
MAX_CALIBRATION_FALSE_ALARMS = 0
MAX_PRE_EOL_ALARMS = 7
EXPECTED_EOL_HORIZON = 20
EXPECTED_PERSISTENCE = 3


def evaluate_report(report: dict, model: str = "baseline_cnn") -> dict:
    """Return the acceptance checks for one recorded CMAPSS model result."""
    protocol = report.get("protocol", {})
    results = report.get("results", {})
    if model not in results:
        raise KeyError(f"model {model!r} is absent from report results")
    if protocol.get("eol_horizon_cycles") != EXPECTED_EOL_HORIZON:
        raise ValueError("report uses an unexpected EOL horizon")
    if protocol.get("persistence_windows") != EXPECTED_PERSISTENCE:
        raise ValueError("report uses an unexpected persistence length")

    metrics = results[model]
    checks = {
        "event_recall_at_eol_horizon": (
            float(metrics["event_recall_at_eol_horizon"]) >= MIN_EVENT_RECALL_AT_EOL,
            f"event_recall_at_eol_horizon < {MIN_EVENT_RECALL_AT_EOL:g}",
        ),
        "calibration_false_alarm_episodes": (
            int(metrics["calibration_false_alarm_episodes"]) == MAX_CALIBRATION_FALSE_ALARMS,
            "calibration_false_alarm_episodes != 0",
        ),
        "pre_eol_alarm_episodes": (
            int(metrics["pre_eol_alarm_episodes"]) <= MAX_PRE_EOL_ALARMS,
            f"pre_eol_alarm_episodes > {MAX_PRE_EOL_ALARMS}",
        ),
    }
    failures = [failure for passed, failure in checks.values() if not passed]
    return {
        "model": model,
        "passed": not failures,
        "failures": failures,
        "metrics": {name: metrics[name] for name in checks},
        "thresholds": {
            "min_event_recall_at_eol_horizon": MIN_EVENT_RECALL_AT_EOL,
            "max_calibration_false_alarm_episodes": MAX_CALIBRATION_FALSE_ALARMS,
            "max_pre_eol_alarm_episodes": MAX_PRE_EOL_ALARMS,
            "eol_horizon_cycles": EXPECTED_EOL_HORIZON,
            "persistence_windows": EXPECTED_PERSISTENCE,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in",
        dest="source",
        type=Path,
        default=Path("results/cmapss_two_teacher_gate.json"),
    )
    parser.add_argument("--model", default="baseline_cnn")
    args = parser.parse_args()

    outcome = evaluate_report(json.loads(args.source.read_text()), model=args.model)
    print(json.dumps(outcome, indent=2, sort_keys=True))
    return 0 if outcome["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
