"""Factorial study: every alarm component at every level of the others (v8).

python -m bench.alarm_interaction {sweep,select,effects,replay,trace}

The alarm measure is uEngram's (frozen seed-0 DSP scores, AlarmPolicy). Each component
below is varied while every other one takes each of its levels; DEFAULT is the actual
frozen v2 alarm.

  alpha          conformal setpoint level                      0.003 | 0.01 | 0.03
  channels       novelty only, or novelty and trusted health    novelty | both
  unbaselined    a state without its own reference: GLOBAL setpoint, unknown, or
                 commissioned online from its first windows     global | unknown | commission
  veto           new memory state with normal health: counts as normal, or unknown
                                                                normal | hold
  persistence    on/off-delay filter (uengram.persistence)      counter | consecutive | m_of_n | ewma
  k              filter length in windows                       3 | 6
  corroboration  required error, m = 1 within L = 6 h (bench.alarm_definition)
                                                                none | dataset | physical
  stale          unknown inputs may hold a filter for at most this many windows (0 = no limit)
                                                                0 | 36

Rules, fixed before any row was computed:
  SELECTION_RULE  rows meeting events_detected == 4, alarm_time_share_clean <= 0.02 and
                  false_alarm_per_week <= 1; then min clean share, most events warned
                  before their report, earliest first alarm, fewest components changed
                  from DEFAULT, design order
  FALLBACK_RULE   if none: rows meeting both alarm budgets, max events_detected, then the
                  same tie order
MetroPT2 (already consumed) is replayed only as a post-hoc diagnostic.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import multiprocessing as mp
from dataclasses import replace
from pathlib import Path

import numpy as np

from bench.alarm_definition import Corroboration, Definition, event_trace
from bench.alarm_eval import CalibrationUnavailable
from bench.alarm_sensitivity import BASELINE
from bench.streams import load_stream
from uengram.alarm import AlarmLogic

LEVELS = {"alpha": (0.003, 0.01, 0.03), "channels": ("novelty", "both"),
          "unbaselined": ("global", "unknown", "commission"), "veto": ("normal", "hold"),
          "persistence": ("counter", "consecutive", "m_of_n", "ewma"), "k": (3, 6),
          "corroboration": ("none", "dataset", "physical"), "stale": (0, 36)}
DEFAULT = {"alpha": 0.003, "channels": "novelty", "unbaselined": "global", "veto": "normal",
           "persistence": "counter", "k": 3, "corroboration": "none", "stale": 0}
DESIGN = [dict(zip(LEVELS, combo)) for combo in itertools.product(*LEVELS.values())]
METRICS = ("events_detected", "false_alarm_per_week", "alarm_time_share_clean",
           "events_warned_before_report", "hours_before_report", "events_warned_2h_before_lps")
EFFECT_METRICS = ("events_detected", "alarm_time_share_clean", "false_alarm_per_week",
                  "events_warned_before_report")
OUT = Path("results")
_DEFS: dict[str, Definition] = {}


def build(levels: dict):
    """(AlarmConfig, AlarmLogic, Corroboration or None) for one design point."""
    both = levels["channels"] == "both"
    cfg = replace(BASELINE, alpha=levels["alpha"], k_novelty=levels["k"],
                  k_health=levels["k"] if both else 0, health_needs_baseline=not both)
    logic = AlarmLogic(persistence=levels["persistence"], veto=levels["veto"],
                       unbaselined=levels["unbaselined"], stale_windows=levels["stale"])
    c = None if levels["corroboration"] == "none" else Corroboration(levels["corroboration"], 1, 6.0)
    return cfg, logic, c


def _definition(name: str) -> Definition:
    if name not in _DEFS:
        _DEFS[name] = Definition(load_stream(name))
    return _DEFS[name]


def _num(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def row(name: str, index: int, levels: dict) -> dict:
    d = _definition(name)
    cfg, logic, c = build(levels)
    base = {"stream": name, "index": index, "levels": levels,
            "changed": sum(levels[k] != DEFAULT[k] for k in LEVELS)}
    try:
        m, det, _ = d.evaluate(cfg, c, logic)
    except CalibrationUnavailable as exc:
        return base | {"metrics": None, "error": str(exc), "satisfies_recall_and_budget": False}
    metrics = {k: _num(m[k]) for k in METRICS if k in m}
    metrics["first_alarm_minus_report_h"] = _num(m["lead_hours"])  # positive = after the report
    ok = (m["events_detected"] == len(d.stream.events) and m["alarm_time_share_clean"] <= 0.02
          and m["false_alarm_per_week"] <= 1.0)
    return base | {"metrics": metrics, "satisfies_recall_and_budget": ok,
                   "events": [{"event": e["event"], "detected": e["detected"],
                               "first_alarm_minus_report_h": e["lead_hours"]} for e in det["events"]]}


def _metropt3_row(args):
    return row("metropt3", *args)


def sweep(jobs: int) -> list[dict]:
    with mp.get_context("spawn").Pool(jobs) as pool:
        rows = []
        for r in pool.imap(_metropt3_row, list(enumerate(DESIGN)), chunksize=8):
            rows.append(r)
            if len(rows) % 144 == 0:
                print(f"[sweep] {len(rows)}/{len(DESIGN)} rows", flush=True)
    return rows


def _tie(r: dict) -> tuple:
    m, first = r["metrics"], r["metrics"]["first_alarm_minus_report_h"]
    return (m["alarm_time_share_clean"], -(m["events_warned_before_report"] or 0),
            first if first is not None else 1e9, r["changed"], r["index"])


def within_budgets(r: dict) -> bool:
    m = r["metrics"]
    return m is not None and m["alarm_time_share_clean"] <= 0.02 and m["false_alarm_per_week"] <= 1.0


def select(rows: list[dict]) -> tuple[dict | None, str]:
    primary = [r for r in rows if r["satisfies_recall_and_budget"]]
    if primary:
        return min(primary, key=_tie), "primary"
    budget = [r for r in rows if within_budgets(r)]
    if not budget:
        return None, "none"
    return min(budget, key=lambda r: (-r["metrics"]["events_detected"],) + _tie(r)), "fallback"


def effects(rows: list[dict]) -> dict:
    """For each component X and level x != DEFAULT[X]: paired change (x minus default) of
    each metric with all other components held fixed; its median overall and its median
    at every level of every other component Y (how X behaves as Y is tweaked)."""
    by = {tuple(r["levels"][k] for k in LEVELS): r["metrics"] for r in rows if r["metrics"]}
    names = list(LEVELS)
    out = {}
    for xi, x in enumerate(names):
        for level in LEVELS[x]:
            if level == DEFAULT[x]:
                continue
            pairs = []
            for key, m in by.items():
                if key[xi] != DEFAULT[x]:
                    continue
                alt = key[:xi] + (level,) + key[xi + 1:]
                if alt in by:
                    pairs.append((key, {k: (by[alt][k] or 0.0) - (m[k] or 0.0) for k in EFFECT_METRICS}))
            medians = {k: float(np.median([p[1][k] for p in pairs])) for k in EFFECT_METRICS}
            entry = {"pairs": len(pairs), "median": medians, "by_other": {}}
            for yi, y in enumerate(names):
                if y == x:
                    continue
                entry["by_other"][y] = {str(lv): {k: float(np.median([p[1][k] for p in pairs if p[0][yi] == lv]))
                                                 for k in EFFECT_METRICS} for lv in LEVELS[y]}
            out[f"{x}={level}"] = entry
    return out


def _write(path: Path, rows: list[dict]):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("sweep", "select", "effects", "replay", "trace"))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    sweep_path = OUT / "v8_interaction_sweep_metropt3.jsonl"
    selection_path = OUT / "v8_interaction_selection.json"
    if args.mode == "sweep":
        _write(sweep_path, sweep(args.jobs))
    elif args.mode == "select":
        rows = _read(sweep_path)
        chosen, how = select(rows)
        budget = [r for r in rows if within_budgets(r)]
        early = max(budget, key=lambda r: (r["metrics"]["events_warned_before_report"] or 0,
                                           r["metrics"]["events_detected"], -r["metrics"]["alarm_time_share_clean"]),
                    default=None)
        doc = {"design_points": len(DESIGN), "rows_meeting_budget": sum(r["satisfies_recall_and_budget"] for r in rows),
               "rows_within_both_budgets": len(budget), "used": how, "selected": chosen,
               "most_early_warnings_within_budgets": early,
               "default_row": next(r for r in rows if r["levels"] == DEFAULT)}
        selection_path.write_text(json.dumps(doc, indent=2) + "\n")
        print(how, None if chosen is None else (chosen["levels"], chosen["metrics"]))
    elif args.mode == "effects":
        (OUT / "v8_interaction_effects.json").write_text(json.dumps(effects(_read(sweep_path)), indent=2) + "\n")
    elif args.mode == "replay":
        doc, rows = json.loads(selection_path.read_text()), []
        for name in ("metropt2_jan", "metropt2_apr"):
            for label in ("selected", "default_row"):
                rows.append(row(name, doc[label]["index"], doc[label]["levels"]) | {"replay_of": label})
                m = rows[-1]["metrics"]
                print(f"[{name}] {label} events={m['events_detected']:.0f} fa={m['false_alarm_per_week']:.2f} "
                      f"clean={m['alarm_time_share_clean']:.4f} warned={m['events_warned_before_report']} "
                      f"2h_before_lps={m.get('events_warned_2h_before_lps')}", flush=True)
        _write(OUT / "v8_interaction_replay_metropt2.jsonl", rows)
    else:
        doc, out = json.loads(selection_path.read_text()), []
        for name in ("metropt3", "metropt2_jan", "metropt2_apr"):
            for label in ("selected", "default_row"):
                cfg, logic, c = build(doc[label]["levels"])
                out.append({"stream": name, "which": label, "levels": doc[label]["levels"],
                            "events": event_trace(_definition(name), cfg, c, logic)})
        (OUT / "v8_interaction_trace.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
