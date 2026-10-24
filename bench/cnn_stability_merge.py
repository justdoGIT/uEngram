#!/usr/bin/env python3
"""Merge fragments produced by ``bench.cnn_stability``."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def seed_summary(components: dict, variant: str, seeds: list[int]) -> dict:
    rows = [components[f"cnn_{variant}_seed{seed}"]["native"] for seed in seeds]
    fields = {
        "events_detected": [row["metrics"]["events_detected"] for row in rows],
        "auroc": [row["metrics"]["auroc"] for row in rows],
        "ap": [row["metrics"]["ap"] for row in rows],
        "false_alarm_episodes": [row["false_alarm_episodes"] for row in rows],
        "clean_time_share": [
            row["metrics"]["alarm_time_share_clean"] for row in rows
        ],
    }
    return {
        key: {
            "median": statistics.median(values),
            "minimum": min(values),
            "maximum": max(values),
            "values": values,
        }
        for key, values in fields.items()
    }


def merge(skab_primary: dict, skab_extra: dict, metropt3: dict) -> dict:
    skab = skab_primary["datasets"]["skab"]
    extra = skab_extra["datasets"]["skab"]
    metro = metropt3["datasets"]["metropt3"]
    for seed in range(5, 10):
        key = f"cnn_base_seed{seed}"
        if key in skab["components"]:
            raise ValueError(f"duplicate component: {key}")
        skab["components"][key] = extra["components"][key]
    skab["seed_groups"] = {
        "base": list(range(10)),
        "actcorr": list(range(5)),
        "swa": list(range(5)),
        "fusions": list(range(5)),
    }
    metro["seed_groups"] = {
        "base": list(range(3)),
        "actcorr": list(range(3)),
        "swa": list(range(3)),
        "fusions": list(range(3)),
    }
    result = {
        "study": skab_primary["study"],
        "protocol": {
            **skab_primary["protocol"],
            "variants": {
                "base": "CNN d=32, epochs=90, maxavg pooling, ortho_beta=0.05",
                "actcorr": (
                    "base plus beta=0.05 off-diagonal "
                    "activation-correlation penalty"
                ),
                "swa": (
                    "base plus arithmetic encoder-weight mean over epochs 60-90; "
                    "SWA-inspired rather than a canonical SWA learning-rate schedule"
                ),
            },
            "native_policy": "bench.run_v2.FROZEN['uengram']",
            "score_only_policy": (
                "same thresholds with every action forced to score and "
                "new_state_alert disabled; used only for comparable score fusion"
            ),
        },
        "datasets": {"skab": skab, "metropt3": metro},
        "derived_seed_summaries": {
            "skab": {
                "base_10_seeds": seed_summary(
                    skab["components"], "base", list(range(10))
                ),
                "actcorr_5_seeds": seed_summary(
                    skab["components"], "actcorr", list(range(5))
                ),
                "swa_5_seeds": seed_summary(
                    skab["components"], "swa", list(range(5))
                ),
            },
            "metropt3": {
                "base_3_seeds": seed_summary(
                    metro["components"], "base", list(range(3))
                ),
                "actcorr_3_seeds": seed_summary(
                    metro["components"], "actcorr", list(range(3))
                ),
                "swa_3_seeds": seed_summary(
                    metro["components"], "swa", list(range(3))
                ),
            },
        },
    }
    if not all(
        dataset["role"].startswith("consumed")
        for dataset in result["datasets"].values()
    ):
        raise ValueError("dataset role must mark these replays as consumed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skab-primary", type=Path, required=True)
    parser.add_argument("--skab-extra", type=Path, required=True)
    parser.add_argument("--metropt3", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = merge(
        load(args.skab_primary),
        load(args.skab_extra),
        load(args.metropt3),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
