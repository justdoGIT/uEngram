#!/usr/bin/env python3
"""Label-free screen for CNN depth, width, and training duration.

This is a development experiment. It injects synthetic fault families into
unlabelled normal windows and never uses benchmark test labels for selection.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bench.autoresearch_benchmark import normal_windows
from bench.model_selection import rank_candidates
from uengram.encoder_cnn import CNNEncoder


def _integers(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--dimensions", default="16,32")
    parser.add_argument("--epochs", default="30,90")
    parser.add_argument("--representations", default="bottleneck,middle,multiscale")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    seeds = _integers(args.seeds)
    dimensions = _integers(args.dimensions)
    epoch_counts = _integers(args.epochs)
    representations = [item for item in args.representations.split(",") if item]
    train, groups = normal_windows(384, 20261025)
    selection, _ = normal_windows(96, 20261026)
    candidates: dict[str, CNNEncoder] = {}
    fit_seconds: dict[str, float] = {}

    for representation in representations:
        for dimension in dimensions:
            for epochs in epoch_counts:
                for seed in seeds:
                    name = f"{representation}_d{dimension}_e{epochs}_s{seed}"
                    print(f"fit_candidate={name}", flush=True)
                    encoder = CNNEncoder(
                        seed=seed,
                        d=dimension,
                        epochs=epochs,
                        representation=representation,
                    )
                    started = time.perf_counter()
                    encoder.fit(train, noise_lag=1, groups=groups)
                    fit_seconds[name] = time.perf_counter() - started
                    candidates[name] = encoder

    ranking = rank_candidates(
        candidates,
        selection,
        seed=20261027,
        strength=3.0,
        quantile=0.1,
    )
    records = []
    for row in ranking:
        name = row["name"]
        prefix, seed_text = name.rsplit("_s", 1)
        representation, dimension_text, epoch_text = prefix.split("_")
        records.append(
            {
                **row,
                "representation": representation,
                "dimension": int(dimension_text[1:]),
                "epochs": int(epoch_text[1:]),
                "seed": int(seed_text),
                "fit_seconds": fit_seconds[name],
            }
        )

    aggregates = []
    for representation in representations:
        for dimension in dimensions:
            for epochs in epoch_counts:
                selected = [
                    row["worst_family_score"]
                    for row in records
                    if row["representation"] == representation
                    and row["dimension"] == dimension
                    and row["epochs"] == epochs
                ]
                aggregates.append(
                    {
                        "representation": representation,
                        "dimension": dimension,
                        "epochs": epochs,
                        "median": float(np.median(selected)),
                        "minimum": float(np.min(selected)),
                        "maximum": float(np.max(selected)),
                        "spread": float(np.max(selected) - np.min(selected)),
                    }
                )
    aggregates.sort(key=lambda row: (-row["minimum"], -row["median"]))
    result = {
        "role": "label-free development screen; not field confirmation",
        "selection": "10th-percentile clean-to-synthetic-fault displacement; worst of five fault families",
        "data": {"train_windows": len(train), "selection_windows": len(selection)},
        "aggregates": aggregates,
        "records": records,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for row in aggregates:
        print(
            "RESULT "
            f"rep={row['representation']} d={row['dimension']} epochs={row['epochs']} "
            f"median={row['median']:.6f} min={row['minimum']:.6f} "
            f"spread={row['spread']:.6f}",
            flush=True,
        )
    print(f"wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
