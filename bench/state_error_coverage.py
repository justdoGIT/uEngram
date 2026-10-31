"""Calibrated-error coverage by operating state on the MetroPT-3 test split.

python -m bench.state_error_coverage --out results/state_error_coverage.json

Counts, per operating state, the test windows and how many have at least one calibrated
error crossing. Supports the audit of the error-gated alarm: a state with no calibration
baseline is never judged, so its calibrated errors never fire.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench.error_gated_alarm import calibrated_errors
from bench.streams import load_stream


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/state_error_coverage.json")
    args = parser.parse_args()

    stream = load_stream("metropt3")
    errors, names = calibrated_errors(stream)
    state = np.asarray(stream.state)
    any_error = np.asarray(errors).astype(bool).any(axis=1)
    rows = {}
    for s in sorted(int(v) for v in np.unique(state[stream.test])):
        mask = stream.test & (state == s)
        rows[str(s)] = {"test_windows": int(mask.sum()), "windows_with_calibrated_error": int(any_error[mask].sum())}
    out = {"stream": "metropt3", "error_channels": list(names), "by_state": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(rows)


if __name__ == "__main__":
    main()
