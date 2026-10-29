"""SKAB embedding-width sweep under the frozen validation protocol.

python -m bench.skab_d_sweep --out results/skab_d_sweep.json

Same stream, alarm policy and scoring path as ``bench.validate_skab``; only the
DSP encoder width varies. Single seed: the DSP encoder is deterministic and the
seed reaches only the uENGRAM store.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench.alarm_eval import apply_policy
from bench.run_v2 import FROZEN
from bench.stream_skab import skab_stream
from bench.v2_scoring import score_stream

ENCODERS = ("dsp_d16", "dsp_d24", "dsp_d32", "dsp_d48")
METRICS = ("alarm_time_share_clean", "alerts_per_week", "ap", "auroc", "events_detected",
           "false_alarm_per_week", "health_ap", "health_auroc", "lead_hours",
           "notifications_per_week_all", "operator_events_per_week")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--out", default="results/skab_d_sweep.json")
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--encoders", default=",".join(ENCODERS))
    parser.add_argument("--seeds", default="0", help="comma-separated; more than one writes a per-seed file")
    args = parser.parse_args()

    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    stream = skab_stream(args.root, K=args.K)
    names = stream.extra["experiment_names"]
    rows = []
    for seed in seeds:
        for enc in encoders:
            sc = score_stream(stream, "uengram", enc, seed, f"skab {enc} s{seed}")
            m, d, _ = apply_policy(stream, sc, FROZEN["uengram"])
            row = {
                "encoder": enc,
                "false_alarm_episodes": int(d["false_alarm_episodes"]),
                "detected_experiments": [names[r["event"]] for r in d["events"] if r["detected"]],
                "metrics": {k: float(m[k]) for k in METRICS if isinstance(m[k], (int, float, np.floating))},
            }
            if len(seeds) > 1:
                row["seed"] = seed
            rows.append(row)
            print(f"{enc} seed={seed}: detected={int(m['events_detected'])}/{len(stream.events)} "
                  f"fa_episodes={row['false_alarm_episodes']} ap={m['ap']:.4f}", flush=True)
    out = {"protocol": "docs/validation-skab-protocol.md", "K": args.K, "rows": rows}
    out["seed" if len(seeds) == 1 else "seeds"] = seeds[0] if len(seeds) == 1 else seeds
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
