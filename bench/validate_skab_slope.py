"""SKAB slope-feature ablation (docs/research-skab-validation.md, RC2 test).

This is development on SKAB, not the frozen protocol in
docs/validation-skab-protocol.md and not a new confirmation: it reuses the
already-scored SKAB stream to test docs/research-remediation-plan.md RC2's
proposal (a rate-of-change feature, not just a window's level) for the
faults bench.validate_skab's frozen `dsp` encoder structurally cannot see.
The alarm policy (`bench.run_v2.FROZEN["uengram"]`) is unchanged; only the
encoder differs from the frozen SKAB result.

python -m bench.validate_skab_slope --out results/skab_slope_ablation.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench.alarm_eval import apply_policy
from bench.run_v2 import FROZEN
from bench.stream_skab import skab_stream
from bench.v2_scoring import score_embeddings
from uengram.encoder_slope import DSPSlopeEncoder


class _Enc:
    """Minimal stand-in so score_embeddings only needs `.sigma0`."""


def _score(stream, lags: tuple[int, ...], seed: int = 0) -> dict:
    enc = DSPSlopeEncoder(d=16, lags=lags).fit(stream.W[stream.fit], 1, stream.segment[stream.fit])
    Z = enc.encode(stream.W, segment=stream.segment)
    stub = _Enc()
    stub.sigma0 = enc.sigma0
    return score_embeddings(stream, "uengram", f"dsp_slope_{lags}", seed, stub, Z,
                            "skab_slope", record_encoding=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--out", default="results/skab_slope_ablation.json")
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--lags", default="3,12")
    args = parser.parse_args()

    lags = tuple(int(x) for x in args.lags.split(","))
    stream = skab_stream(args.root, K=args.K)
    sc = _score(stream, lags)
    m, d, pol = apply_policy(stream, sc, FROZEN["uengram"])

    names = stream.extra["experiment_names"]
    per_experiment = []
    for rec in d["events"]:
        i = rec["event"]
        onset, offset = stream.events[i]
        per_experiment.append({
            "experiment": names[i], "detected": rec["detected"],
            "windows": rec["windows"], "alarm_windows": rec["alarm_windows"],
            "onset": onset, "offset": offset, "first_alarm": rec["first_alarm"],
            "lead_hours": rec["lead_hours"],
        })

    out = {
        "note": "SKAB slope-feature ablation: development on SKAB, not a fresh "
                "confirmation. Compare against results/skab_validation.json "
                "(the frozen dsp encoder).",
        "stream": "skab", "method": "uengram", "encoder": "dsp_slope", "lags": list(lags),
        "seed": 0, "K": args.K, "n_experiments": len(stream.events),
        "config": FROZEN["uengram"].__dict__,
        "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in m.items()},
        "details": {k: v for k, v in d.items() if k not in ("active", "opened", "valid",
                                                             "baseline_counts")},
        "per_experiment": per_experiment,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True, allow_nan=False) + "\n")
    detected = int(m["events_detected"])
    print(f"lags={lags} events_detected={detected}/{len(stream.events)} "
          f"auroc={m['auroc']:.3f} ap={m['ap']:.3f} "
          f"alarm_time_share_clean={m['alarm_time_share_clean']*100:.2f}% "
          f"false_alarm_episodes={d['false_alarm_episodes']}", flush=True)


if __name__ == "__main__":
    main()
