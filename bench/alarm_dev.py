"""Alarm v2 development sweep on MetroPT-3 (docs/EXPLAINED.md section 17).

MetroPT-3 was consumed by protocol v1, so it is the development set: every
alarm-policy configuration in GRID is evaluated on it with labels. The
selection rule was fixed before the sweep ran:

1. keep configurations within the false-alarm budget: at most 1 false-alarm
   episode per clean week AND an alarm active in at most 2% of clean time
   (about 3.4 h a week; a few long standing alarms are also a nuisance);
2. take the most events detected;
3. ties: smallest k_novelty + k_health (earliest possible alarm), then fewest
   false-alarm episodes per week, then fewest optional components switched on
   (per_state, new_state_alert, ACI), then larger alpha.

Each (method, encoder) gets its own configuration under the same rule, so
baselines are tuned as much as uEngram. Seed 0 only; DSP uEngram is
deterministic. Usage:
python -m bench.alarm_dev --methods uengram:dsp,hst:dsp,memstream:dsp \
    --out ~/.local/state/uengram/alarm_dev.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

from bench.alarm_eval import CalibrationUnavailable, apply_policy
from bench.streams import metropt3_stream
from bench.v2_scoring import cached_scores
from uengram.alarm import AlarmConfig

GRID = {"alpha": (0.01, 0.003), "k_novelty": (0, 1, 2, 3, 4, 6, 9, 12),
        "k_health": (0, 3, 6, 12, 18), "per_state": (False, True),
        "new_state_alert": (False, True), "aci_gamma": (0.0, 0.005),
        "health_needs_baseline": (False, True)}
V1_RULE = AlarmConfig(k_novelty=1, k_health=0, per_state=False, new_state_alert=False)
SCORES = os.path.expanduser("~/.local/state/uengram/v2scores")
KEEP = ("events_detected", "false_alarm_per_week", "lead_hours", "alerts_per_week",
        "alarm_time_share_clean")


def configs():
    keys = list(GRID)
    for values in itertools.product(*(GRID[k] for k in keys)):
        cfg = AlarmConfig(**dict(zip(keys, values)))
        if cfg.k_novelty or cfg.k_health:
            yield cfg


def within_budget(m) -> bool:
    return m["false_alarm_per_week"] <= 1.0 and m["alarm_time_share_clean"] <= 0.02


def rank_key(row):
    c, m = row["config"], row["metrics"]
    extras = (int(c["per_state"]) + int(c["new_state_alert"]) + int(c["aci_gamma"] > 0)
              + int(c.get("health_needs_baseline", False)))
    return (-m["events_detected"], c["k_novelty"] + c["k_health"], m["false_alarm_per_week"],
            extras, -c["alpha"])


def select(rows):
    ok = [r for r in rows if within_budget(r["metrics"])]
    return min(ok, key=rank_key) if ok else None


def sweep(stream, sc, method, enc, seed, out, label):
    rows, t0, grid = [], time.monotonic(), list(configs())
    for j, cfg in enumerate(grid):
        try:
            m, d, _ = apply_policy(stream, sc, cfg)
            row = {"method": method, "encoder": enc, "seed": seed, "config": cfg.__dict__,
                   "metrics": {k: m[k] for k in KEEP}, "detected": [e["detected"] for e in d["events"]]}
        except CalibrationUnavailable as error:
            row = {"method": method, "encoder": enc, "seed": seed, "config": cfg.__dict__,
                   "status": "invalid_calibration", "error": str(error),
                   "metrics": {k: float("nan") for k in KEEP}, "detected": []}
        rows.append(row)
        out.write(json.dumps(row) + "\n")
        if (j + 1) % 100 == 0:
            print(f"{label} configs {j + 1}/{len(grid)} {time.monotonic() - t0:.0f}s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="uengram:dsp,hst:dsp,memstream:dsp")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.expanduser("~/.local/state/uengram/alarm_dev.jsonl"))
    a = ap.parse_args()
    stream = metropt3_stream()
    chosen = {}
    with open(a.out, "w") as out:
        for spec in a.methods.split(","):
            method, enc = spec.split(":")
            label = f"[alarm_dev] {method}/{enc} seed={a.seed}"
            print(f"{label} scoring (cached in {SCORES})", flush=True)
            sc = cached_scores(stream, method, enc, a.seed, SCORES, label)
            rows = sweep(stream, sc, method, enc, a.seed, out, label)
            v1 = apply_policy(stream, sc, V1_RULE)[0]
            best = chosen[spec] = select(rows)
            meets = sum(within_budget(r["metrics"]) for r in rows)
            print(f"{label} v1 rule: detected={v1['events_detected']:.0f}/4 "
                  f"FA/week={v1['false_alarm_per_week']:.2f}", flush=True)
            print(f"{label} {meets}/{len(rows)} configurations within budget; "
                  f"selected: {json.dumps(best)}", flush=True)
    print(json.dumps(chosen, indent=1))


if __name__ == "__main__":
    main()
