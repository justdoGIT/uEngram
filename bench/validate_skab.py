"""Run the frozen SKAB validation protocol (docs/validation-skab-protocol.md).

python -m bench.validate_skab --out results/skab_validation.json
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


def _per_experiment(stream, details) -> list[dict]:
    names = stream.extra["experiment_names"]
    out = []
    for rec in details["events"]:
        i = rec["event"]
        onset, offset = stream.events[i]
        out.append({
            "experiment": names[i], "detected": rec["detected"],
            "windows": rec["windows"], "alarm_windows": rec["alarm_windows"],
            "onset": onset, "offset": offset, "first_alarm": rec["first_alarm"],
            "lead_hours": rec["lead_hours"],
            "lead_seconds": None if rec["lead_hours"] is None else rec["lead_hours"] * 3600,
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--out", default="results/skab_validation.json")
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--shock-mult", type=float, default=0.0)
    args = parser.parse_args()

    stream = skab_stream(args.root, K=args.K)
    sc = score_stream(stream, "uengram", "dsp", 0, "skab")
    cfg = FROZEN["uengram"]
    if args.shock_mult > 0.0:
        from uengram.alarm import AlarmConfig
        cfg = AlarmConfig(**{**cfg.__dict__, "shock_mult": args.shock_mult})
    m, d, pol = apply_policy(stream, sc, cfg)
    per_experiment = _per_experiment(stream, d)
    out = {
        "protocol": "docs/validation-skab-protocol.md",
        "stream": "skab", "method": "uengram", "encoder": "dsp", "seed": 0,
        "K": args.K, "n_experiments": len(stream.events),
        "config": FROZEN["uengram"].__dict__,
        "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in m.items()},
        "details": {k: v for k, v in d.items() if k not in ("active", "opened", "valid",
                                                             "baseline_counts")},
        "per_experiment": per_experiment,
        "run_meta": {k: v for k, v in sc["meta"].items() if k != "encoding"},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True, allow_nan=False) + "\n")
    detected = int(m["events_detected"])
    print(f"events_detected={detected}/{len(stream.events)} "
          f"auroc={m['auroc']:.3f} ap={m['ap']:.3f} "
          f"false_alarm_per_week={m['false_alarm_per_week']:.2f} "
          f"alarm_time_share_clean={m['alarm_time_share_clean']:.4f}", flush=True)


if __name__ == "__main__":
    main()
