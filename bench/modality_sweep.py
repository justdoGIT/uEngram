"""Sensitivity sweep of the two-condition alarm (bench/modality_alarm.py).

Steps, in order, all development-only on consumed recordings:
  ofat      vary one ModalityConfig field at a time on MetroPT3
  pairwise  full grid over two fields on MetroPT3
  select    apply the rule below to every MetroPT3 row written so far
  replay    apply the single selected config, unchanged, to both MetroPT2 recordings
  relations closed-form duty-only alarm per stream, checked against the harness

SELECTION_RULE is fixed here before any MetroPT2 row is computed: among MetroPT3 rows with
all 4 events detected, alarm_time_share_clean <= 0.02 and false_alarm_per_week <= 1, take the
lowest alarm_time_share_clean, then the smaller lead_hours (lead_hours is first alarm minus
report start, so smaller means earlier), then the smaller k_hold. If no row qualifies,
FALLBACK_RULE (below) picks a budget-failing row that is replayed only to test transfer.
MetroPT2 replay is never used to choose or change anything.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import replace
from pathlib import Path

from bench.modality_alarm import (PHYSICAL_CHANNELS, PNEUMATIC_CHANNELS, ModalityConfig,
                                  duty_only_prediction, evaluate, sustained_runs)
from bench.streams import load_stream

# alpha=0.05 family-wise is the smallest grid level whose Bonferroni per-channel level
# (0.05/7) the 334-window state-1 and pooled loaded references can resolve (m >= 139).
BASELINE = ModalityConfig(alpha=0.05)
GRIDS = {
    "k_hold": (1, 2, 3, 6, 10, 15, 20, 25, 30, 35, 40, 50),
    "k_release": (1, 3, 6),
    "min_modalities": (1, 2, 3),
    "alpha": (0.2, 0.1, 0.05, 0.03, 0.01),
    "duty": (False, True),
    "unbaselined": ("ignore", "global"),
    "reference": ("cal", "fit+cal"),
    "correction": ("bonferroni", "none"),
    "calibration": ("conformal", "parametric"),
    "conditioning": ("state", "loaded"),
    "channels": (PHYSICAL_CHANNELS, PNEUMATIC_CHANNELS),
}
METRICS = ("events_detected", "false_alarm_per_week", "alarm_time_share_clean", "lead_hours",
           "window_sensitivity", "window_specificity", "window_precision",
           "notifications_per_week_all", "hold_minutes")
SELECTION_RULE = ("events_detected==4 AND alarm_time_share_clean<=0.02 AND "
                  "false_alarm_per_week<=1; then min alarm_time_share_clean, min lead_hours "
                  "(earliest first alarm), min k_hold")
OUT = Path("results")


def _clean(value):
    return None if isinstance(value, float) and not math.isfinite(value) else value


def row(stream, cfg: ModalityConfig, varied: dict) -> dict:
    m, d = evaluate(stream, cfg)
    n_events = len(stream.events)
    ok = (m["events_detected"] == n_events and m["alarm_time_share_clean"] <= 0.02
          and m["false_alarm_per_week"] <= 1.0)
    return {"stream": stream.name, "varied": varied, "config": cfg.__dict__ | {"channels": list(cfg.channels)},
            "metrics": {k: _clean(float(m[k])) for k in METRICS},
            "events": [{"event": e["event"], "detected": e["detected"], "lead_hours": e["lead_hours"],
                        "alarm_windows": e["alarm_windows"], "windows": e["windows"]}
                       for e in d["events"]],
            "opening_modalities": d["opening_modalities"],
            "reference_windows": d["reference_windows"],
            "judgeable_channels": d["judgeable_channels"],
            "per_channel_level": d["per_channel_level"],
            "clean_error_rate": d["clean_error_rate"],
            "event_error_rate": d["event_error_rate"],
            "satisfies_recall_and_budget": ok}


def _log(r: dict):
    m = r["metrics"]
    judgeable = sum(len(v) for v in r["judgeable_channels"].values())
    print(f"[{r['stream']}] {r['varied']} events={m['events_detected']} "
          f"fa/wk={m['false_alarm_per_week']:.3f} clean={m['alarm_time_share_clean']:.4f} "
          f"lead={m['lead_hours']} judgeable={judgeable} ok={r['satisfies_recall_and_budget']}", flush=True)


def ofat(stream) -> list[dict]:
    rows = [row(stream, BASELINE, {"param": None, "value": None})]
    _log(rows[-1])
    for name, grid in GRIDS.items():
        for value in grid:
            if value != getattr(BASELINE, name):
                rows.append(row(stream, replace(BASELINE, **{name: value}), {"param": name, "value": value}))
                _log(rows[-1])
    return rows


def pairwise(stream, a: str, b: str, fixed: dict) -> list[dict]:
    base, rows = replace(BASELINE, **fixed), []
    for va, vb in itertools.product(GRIDS[a], GRIDS[b]):
        rows.append(row(stream, replace(base, **{a: va, b: vb}), {a: va, b: vb, "fixed": fixed}))
        _log(rows[-1])
    return rows


def _tie_key(r: dict) -> tuple:
    """Deterministic, simplest-first ties: earlier alarm, shorter hold, OR over modalities,
    fewer judged channels, window-state conditioning, larger (cruder) alpha."""
    c, m = r["config"], r["metrics"]
    return (m["lead_hours"] if m["lead_hours"] is not None else 1e9, c["k_hold"],
            c["min_modalities"], len(c["channels"]), c["conditioning"] != "state", -c["alpha"])


def select(rows: list[dict]) -> tuple[dict | None, str]:
    """Predeclared before any MetroPT2 replay. Primary: SELECTION_RULE. Fallback when no row
    meets the budget (FALLBACK_RULE): the closest budget-failing row, reported as such."""
    rows = [r for r in rows if r["stream"] == "metropt3"]
    ok = [r for r in rows if r["satisfies_recall_and_budget"]]
    if ok:
        return min(ok, key=lambda r: (r["metrics"]["alarm_time_share_clean"], *_tie_key(r))), "primary"
    full = [r for r in rows if r["metrics"]["events_detected"] == 4.0
            and r["metrics"]["false_alarm_per_week"] <= 1.0]
    if not full:
        return None, "none"
    return min(full, key=lambda r: (r["metrics"]["alarm_time_share_clean"], *_tie_key(r))), "fallback"


FALLBACK_RULE = ("if no row meets SELECTION_RULE: among rows with events_detected==4 and "
                 "false_alarm_per_week<=1, min alarm_time_share_clean, then earliest alarm, "
                 "shorter hold, OR over modalities, fewer channels, state conditioning, larger alpha; "
                 "the result fails the clean-time budget and is replayed only to test transfer")


def _write(path: Path, rows: list[dict]):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _read(*paths: Path) -> list[dict]:
    return [json.loads(line) for p in paths if p.exists() for line in p.read_text().splitlines()]


def relations() -> dict:
    """Closed-form duty-only relations per stream, checked against the harness.

    MetroPT2 values are a posthoc structural diagnostic of why the frozen choice failed to
    transfer; they were computed after replay and are never used to select anything."""
    out = {}
    for name in ("metropt3", "metropt2_jan", "metropt2_apr"):
        stream = load_stream(name)
        runs = sustained_runs(stream)
        test = stream.test.nonzero()[0]
        event_id, clean = stream.event_id()[test], stream.clean_negative()[test]
        event_runs = {e: max((b - a + 1 for a, b in runs if (event_id[a:b + 1] == e).any()), default=0)
                      for e in range(len(stream.events))}
        clean_runs = sorted((b - a + 1 for a, b in runs if clean[a:b + 1].all()), reverse=True)
        rows = []
        for k in GRIDS["k_hold"]:
            measured, _ = evaluate(stream, replace(BASELINE, channels=(), k_hold=k))
            rows.append({"k_hold": k, "predicted": duty_only_prediction(stream, k),
                         "harness_duty_only": {key: _clean(float(measured[key])) for key in
                                               ("events_detected", "alarm_time_share_clean", "lead_hours")}})
        out[name] = {"role": "development" if name == "metropt3" else "posthoc diagnostic (not used for selection)",
                     "longest_event_run_windows": event_runs, "clean_runs_longest_first": clean_runs[:12],
                     "clean_run_count": len(clean_runs), "by_k_hold": rows}
        print(name, "event runs", event_runs, "longest clean runs", clean_runs[:6], flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("ofat", "pairwise", "select", "replay", "relations"))
    parser.add_argument("--pair", nargs=2, default=("k_hold", "min_modalities"))
    parser.add_argument("--fixed", nargs="*", default=())
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    fixed = {}
    for item in args.fixed:
        name, _, raw = item.partition("=")
        try:
            value = json.loads(raw.lower() if raw.lower() in ("true", "false") else raw)
        except json.JSONDecodeError:
            value = raw  # plain strings such as fit+cal or ignore
        fixed[name] = tuple(value) if isinstance(value, list) else value  # e.g. channels=[...]
    sweeps = sorted(OUT.glob("v5_modality_*_metropt3*.jsonl"))
    if args.mode == "ofat":
        _write(OUT / "v5_modality_ofat_metropt3.jsonl", ofat(load_stream("metropt3")))
    elif args.mode == "pairwise":
        rows = pairwise(load_stream("metropt3"), *args.pair, fixed)
        _write(OUT / f"v5_modality_pairwise{args.tag}_metropt3.jsonl", rows)
    elif args.mode == "select":
        rows = _read(*sweeps)
        chosen, how = select(rows)
        doc = {"selection_rule": SELECTION_RULE, "fallback_rule": FALLBACK_RULE, "used": how,
               "candidate_files": [str(p) for p in sweeps], "candidates": len(rows),
               "budget_satisfying_candidates": sum(r["satisfies_recall_and_budget"] for r in rows),
               "selected": chosen}
        (OUT / "v5_modality_selection.json").write_text(json.dumps(doc, indent=2) + "\n")
        print(how, json.dumps(chosen["config"] if chosen else None))
    elif args.mode == "relations":
        (OUT / "v5_modality_relations.json").write_text(json.dumps(relations(), indent=2) + "\n")
    else:
        doc = json.loads((OUT / "v5_modality_selection.json").read_text())
        if doc["selected"] is None:
            raise SystemExit("no MetroPT3 configuration met any selection rule; nothing to replay")
        cfg_dict = dict(doc["selected"]["config"])
        cfg = ModalityConfig(**(cfg_dict | {"channels": tuple(cfg_dict["channels"])}))
        rows = [row(load_stream(name), cfg, {"replay_of": "v5_modality_selection.json", "rule": doc["used"]})
                for name in ("metropt2_jan", "metropt2_apr")]
        for r in rows:
            _log(r)
        _write(OUT / "v5_modality_replay_metropt2.jsonl", rows)


if __name__ == "__main__":
    main()
