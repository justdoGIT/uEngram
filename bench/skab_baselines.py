"""SKAB comparison of uENGRAM against the streaming baselines.

python -m bench.skab_baselines --seeds 0,1,2,3,4 --out results/skab_baselines.json

Same stream, DSP encoder, scoring path and per-method frozen alarm configuration
as the MetroPT-2 confirmation runs (``bench.run_v2.FROZEN``); only the dataset
differs. HST and MemStream are seeded, uENGRAM's DSP path is deterministic.
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

METHODS = ("uengram", "hst", "memstream")
METRICS = ("alarm_time_share_clean", "ap", "auroc", "events_detected", "false_alarm_per_week")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--out", default="results/skab_baselines.json")
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--encoder", default="dsp")
    args = parser.parse_args()

    stream = skab_stream(args.root, K=args.K)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rows = []
    for method in methods:
        for seed in seeds:
            sc = score_stream(stream, method, args.encoder, seed, f"skab {method} s{seed}")
            m, d, _ = apply_policy(stream, sc, FROZEN[method])
            rows.append({
                "method": method, "encoder": args.encoder, "seed": seed,
                "false_alarm_episodes": int(d["false_alarm_episodes"]),
                "n_events": len(stream.events),
                "metrics": {k: float(m[k]) for k in METRICS
                            if isinstance(m[k], (int, float, np.floating))},
            })
            print(f"{method} seed={seed}: detected={int(m['events_detected'])}/{len(stream.events)} "
                  f"fa_episodes={rows[-1]['false_alarm_episodes']} ap={m['ap']:.4f}", flush=True)
    out = {"protocol": "docs/validation-skab-protocol.md", "K": args.K, "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
