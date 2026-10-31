"""Summarize the frozen cross-dataset uEngram gate.

This report intentionally separates the new MOMENT transfer holdout from the
older CMAPSS and SKAB protocol results. It is a release gate, not a claim that
one teacher wins every dataset.

Example:
    python -m bench.stronger_gate --out results/stronger_gate.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev


def _read_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def _summary(rows: list[dict], metric: str) -> dict:
    values = [float(row["metrics"][metric]) for row in rows]
    result = {"n": len(values), "mean": mean(values)}
    if len(values) > 1:
        deviation = stdev(values)
        result.update({
            "sd": deviation,
            "ci95_normal_approx": 1.96 * deviation / math.sqrt(len(values)),
        })
    else:
        result.update({"sd": 0.0, "ci95_normal_approx": 0.0})
    return result


def _cmapss_summary(path: str) -> dict:
    rows = _read_jsonl(path)
    selected = {}
    for encoder in ("dsp", "cnn"):
        for method in ("uengram", "uengram_frozen"):
            group = [
                row for row in rows
                if row["encoder"] == encoder and row["method"] == method
            ]
            selected[f"{encoder}:{method}"] = {
                metric: _summary(group, metric)
                for metric in ("auroc", "health_auroc", "rul_rmse_health", "eol_retention")
            }
    return {"protocol_rows": len(rows), "selected": selected}


def _tsb_summary(path: str) -> dict:
    data = json.loads(Path(path).read_text())
    variants = []
    for variant in data["variants"]:
        variants.append({
            "context_policy": variant["context_policy"],
            "include_instance_stats": variant["include_instance_stats"],
            "teacher_feature_dim": variant["teacher_feature_dim"],
            "overall": variant["scores"]["overall"],
            "macro_series_auroc": mean(
                item["auroc"] for item in variant["scores"]["per_series"].values()
            ),
            "macro_series_average_precision": mean(
                item["average_precision"]
                for item in variant["scores"]["per_series"].values()
            ),
        })
    return {
        "role": data["role"],
        "dataset": data["dataset"],
        "baseline": data["baseline"]["overall"],
        "variants": variants,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmapss", default="results/cmapss_runs.jsonl")
    parser.add_argument("--skab", default="results/skab_validation.json")
    parser.add_argument("--tsb", default="results/moment_tsb_ad_holdout.json")
    parser.add_argument("--out", default="results/stronger_gate.json")
    args = parser.parse_args()

    skab = json.loads(Path(args.skab).read_text())
    report = {
        "gate": "stronger-frozen-cross-dataset",
        "interpretation": (
            "This gate checks reproducible protocol coverage and reports dataset-specific "
            "metrics. It is not a universal accuracy or production-certification claim."
        ),
        "datasets": {
            "TSB-AD-U": {
                "role": "new MOMENT teacher-transfer holdout",
                "results": _tsb_summary(args.tsb),
            },
            "CMAPSS-FD001": {
                "role": "existing five-seed lifecycle and RUL gate",
                "results": _cmapss_summary(args.cmapss),
                "teacher_transfer": "not run under the MOMENT protocol",
            },
            "SKAB": {
                "role": "fresh frozen event-level validation",
                "results": {
                    "metrics": skab["metrics"],
                    "run_meta": skab["run_meta"],
                    "events": skab["n_experiments"],
                },
                "teacher_transfer": "not run under the MOMENT protocol",
            },
        },
        "checks": [
            "TSB has no test-label model selection in the recorded protocol.",
            "CMAPSS reports health discrimination, RUL error, and EOL retention together.",
            "SKAB reports event detection and false-alarm burden together.",
            "CMAPSS and SKAB teacher-transfer results remain an explicit follow-up gate.",
        ],
        "limitations": [
            "The datasets use different window sizes, labels, and operating assumptions.",
            "The TSB collection mixes domains and includes only six selected series.",
            "The normal-approximation intervals are descriptive with few seeds; bootstrap and asset-level confidence intervals are still needed.",
        ],
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("wrote", output)


if __name__ == "__main__":
    main()
