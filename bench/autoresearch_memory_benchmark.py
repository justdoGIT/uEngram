"""Stress-test rare-event memory retention at a fixed compact state budget.

The existing labelled sensor simulator is only a mechanics proxy for long-lived
simulation memories; this does not test a 3D renderer, an LLM, or AR/VR quality.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from bench.common import EPISODIC_SLOT_FRACTION
from bench.run_synth import run_one

SEEDS = (0, 1, 2, 3, 4)
BUDGET = 16
WINDOWS = 12_000
CALIBRATION_WINDOWS = 4_000
METHODS = ("uengram", "fifo", "reservoir")


def main() -> None:
    rows: dict[str, list[dict]] = defaultdict(list)
    for seed in SEEDS:
        result = run_one((seed, "dsp", WINDOWS, METHODS, (BUDGET,),
                          "autoresearch-memory", 0.01, CALIBRATION_WINDOWS))
        for item in result:
            rows[item["method"]].append(item["metrics"])

    print(f"configuration=simulated-sensor episode-memory budget={BUDGET} "
          f"episodic_fraction={EPISODIC_SLOT_FRACTION:.3f}")
    for method in METHODS:
        for key in ("event_recall", "seizure_recall", "type_recall",
                    "false_flag_rate_era1", "write_reduction_pct",
                    "items_end", "regime_acc_at1", "lat_p99_us"):
            values = np.asarray([row[key] for row in rows[method]], dtype=np.float64)
            finite = values[np.isfinite(values)]
            value = float(finite.mean()) if finite.size else float("nan")
            print(f"METRIC {method}_{key}={value:.12g}")
            if method == "uengram" and key in ("event_recall", "type_recall"):
                print(f"ASI {method}_{key}_sd={float(np.nanstd(values, ddof=1)):.12g}")


if __name__ == "__main__":
    main()
