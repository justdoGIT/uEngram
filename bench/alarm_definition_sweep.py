"""Sweep, select, replay and explain the v7 alarm definition (bench/alarm_definition.py).

Modes, in order (frozen seed-0 DSP uEngram scores, no rescoring):
  sweep      every ALARM_CONFIG x (uncorroborated + every CORROBORATION) on MetroPT3
  select     the rules below, separately per source family, on the MetroPT3 rows
  replay     each selected alarm, unchanged, on both MetroPT2 recordings
  relations  exact checks on the MetroPT3 rows
  trace      when each condition was first met around every event, selected alarms

ALARM_CONFIGS is the v6 list: the actual alarm definition (frozen v2 AlarmConfig) first,
then every AlarmConfig the v4 study explored.

Rules, fixed before any corroborated row was computed:
  SELECTION_RULE  corroborated rows with events_detected == 4,
                  alarm_time_share_clean <= 0.02 and false_alarm_per_week <= 1; then min
                  clean share, earliest first alarm, shorter lookback, smaller m, earlier
                  ALARM_CONFIG
  FALLBACK_RULE   if none: corroborated rows meeting both alarm budgets, max
                  events_detected, then the same tie order; replayed only to test transfer
MetroPT2 is never used to choose or change anything.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
from dataclasses import asdict
from pathlib import Path

from bench.alarm_definition import Corroboration, Definition, event_trace
from bench.alarm_eval import CalibrationUnavailable
from bench.error_gated_sweep import ALARM_CONFIGS
from bench.streams import load_stream

OUT = Path("results")
CORROBORATIONS = tuple(Corroboration(f, m, L) for f, ms in (("dataset", (1, 2)), ("calibrated", (1, 2, 3)))
                       for m in ms for L in (0, 1, 3, 6, 12, 24))
METRICS = ("events_detected", "false_alarm_per_week", "alarm_time_share_clean", "events_warned_before_report",
           "hours_before_report", "notifications_per_week_all", "events_warned_before_lps",
           "hours_before_lps", "events_warned_2h_before_lps")
SELECTION_RULE = ("corroborated rows, events_detected==4 AND alarm_time_share_clean<=0.02 AND "
                  "false_alarm_per_week<=1; min clean share, earliest first alarm, shorter lookback, "
                  "smaller m, earlier ALARM_CONFIG")
FALLBACK_RULE = ("corroborated rows with alarm_time_share_clean<=0.02 AND false_alarm_per_week<=1; "
                 "max events_detected, then the SELECTION_RULE tie order; replayed only to test transfer")
_DEFINITIONS: dict[str, Definition] = {}


def _num(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def _definition(name: str) -> Definition:
    if name not in _DEFINITIONS:
        _DEFINITIONS[name] = Definition(load_stream(name))
    return _DEFINITIONS[name]


def row(definition: Definition, index: int, c: Corroboration | None) -> dict:
    s = definition.stream
    base = {"stream": s.name, "alarm_index": index, "alarm_config": asdict(ALARM_CONFIGS[index]),
            "corroboration": None if c is None else asdict(c)}
    try:
        m, d, _ = definition.evaluate(ALARM_CONFIGS[index], c)
    except CalibrationUnavailable as exc:
        return base | {"calibration_unavailable": str(exc), "metrics": None, "satisfies_recall_and_budget": False}
    metrics = {k: _num(m[k]) for k in METRICS if k in m}
    metrics["first_alarm_minus_report_h"] = _num(m["lead_hours"])  # mean; positive = after the report
    events = [{"event": e["event"], "detected": e["detected"], "first_alarm_minus_report_h": e["lead_hours"],
               "alarm_windows": e["alarm_windows"], "windows": e["windows"],
               "hours_before_lps": e.get("hours_before_lps")} for e in d["events"]]
    ok = (m["events_detected"] == len(s.events) and m["alarm_time_share_clean"] <= 0.02
          and m["false_alarm_per_week"] <= 1.0)
    return base | {"calibration_unavailable": None, "metrics": metrics, "events": events,
                   "vote_share_clean": d.get("vote_share_clean"),
                   "corroboration_ceiling_events": d.get("corroboration_ceiling_events"),
                   "satisfies_recall_and_budget": ok}


def _config_rows(index: int) -> list[dict]:
    definition = _definition("metropt3")
    rows = [row(definition, index, None)]
    if rows[0]["metrics"] is not None:
        rows += [row(definition, index, c) for c in CORROBORATIONS]
    return rows


def sweep(jobs: int) -> list[dict]:
    rows = []
    with mp.get_context("spawn").Pool(jobs) as pool:
        for index, block in enumerate(pool.imap(_config_rows, range(len(ALARM_CONFIGS)))):
            rows += block
            ref = block[0]["metrics"]
            if ref is None:
                print(f"[sweep] config {index}: calibration unavailable", flush=True)
                continue
            best = {f: max((r for r in block[1:] if r["corroboration"]["family"] == f),
                           key=lambda r: (r["metrics"]["events_detected"], -r["metrics"]["alarm_time_share_clean"]))
                    for f in ("dataset", "calibrated")}
            print(f"[sweep] config {index}: policy {ref['events_detected']:.0f} ev "
                  f"clean={ref['alarm_time_share_clean']:.4f} fa={ref['false_alarm_per_week']:.2f} | " +
                  " | ".join(f"{Corroboration(**b['corroboration']).label()} "
                             f"{b['metrics']['events_detected']:.0f} ev "
                             f"clean={b['metrics']['alarm_time_share_clean']:.4f} "
                             f"fa={b['metrics']['false_alarm_per_week']:.2f}"
                             for b in best.values()), flush=True)
    return rows


def _tie(r: dict) -> tuple:
    first = r["metrics"]["first_alarm_minus_report_h"]
    return (r["metrics"]["alarm_time_share_clean"], first if first is not None else 1e9,
            r["corroboration"]["lookback_h"], r["corroboration"]["min_errors"], r["alarm_index"])


def select(rows: list[dict], family: str) -> tuple[dict | None, str]:
    cand = [r for r in rows if r["metrics"] is not None and r["corroboration"] is not None
            and r["corroboration"]["family"] == family]
    primary = [r for r in cand if r["satisfies_recall_and_budget"]]
    if primary:
        return min(primary, key=_tie), "primary"
    budget = [r for r in cand if r["metrics"]["alarm_time_share_clean"] <= 0.02
              and r["metrics"]["false_alarm_per_week"] <= 1.0]
    if not budget:
        return None, "none"
    return min(budget, key=lambda r: (-r["metrics"]["events_detected"],) + _tie(r)), "fallback"


def relations(rows: list[dict]) -> dict:
    """Exact checks. Dataset votes are two-valued, so corroboration can only remove alarm
    time; calibrated votes can be unknown, which holds counters and can extend an alarm."""
    tol = 1e-12
    ref = {r["alarm_index"]: r["metrics"] for r in rows if r["corroboration"] is None and r["metrics"]}
    v6 = {r["alarm_index"]: r["metrics"] for r in map(json.loads, (OUT / "v6_error_gated_sweep_metropt3.jsonl")
                                                      .read_text().splitlines()) if r["gate"] is None and r["metrics"]}
    keys = ("events_detected", "false_alarm_per_week", "alarm_time_share_clean")
    out = {"uncorroborated_rows_differing_from_v6_ungated": sum(
        any(abs(ref[i][k] - v6[i][k]) > tol for k in keys) for i in ref)}
    cor = [r for r in rows if r["corroboration"] is not None and r["metrics"]]
    for family in ("dataset", "calibrated"):
        fam = [r for r in cor if r["corroboration"]["family"] == family]
        out[family] = {
            "rows": len(fam),
            "clean_share_above_policy": sum(r["metrics"]["alarm_time_share_clean"] >
                                            ref[r["alarm_index"]]["alarm_time_share_clean"] + tol for r in fam),
            "events_above_policy": sum(r["metrics"]["events_detected"] > ref[r["alarm_index"]]["events_detected"]
                                       for r in fam),
            "events_above_corroboration_ceiling": sum(r["metrics"]["events_detected"] >
                                                      r["corroboration_ceiling_events"] for r in fam),
            "rows_meeting_budget": sum(r["satisfies_recall_and_budget"] for r in fam),
            "max_events_within_both_budgets": max((r["metrics"]["events_detected"] for r in fam
                                                   if r["metrics"]["alarm_time_share_clean"] <= 0.02
                                                   and r["metrics"]["false_alarm_per_week"] <= 1.0), default=None),
            "corroboration_ceiling_by_rule": {Corroboration(**r["corroboration"]).label():
                                              r["corroboration_ceiling_events"] for r in fam if r["alarm_index"] == 0}}
        by = {(r["alarm_index"], r["corroboration"]["min_errors"], r["corroboration"]["lookback_h"]):
              r["metrics"]["alarm_time_share_clean"] for r in fam}
        out[family]["clean_share_lower_at_longer_lookback"] = sum(
            by[(i, m, b)] + tol < by[(i, m, a)] for (i, m, a) in by for (j, n, b) in by if (j, n) == (i, m) and b > a)
        out[family]["clean_share_higher_at_larger_m"] = sum(
            by[(i, m + 1, L)] > by[(i, m, L)] + tol for (i, m, L) in by if (i, m + 1, L) in by)
    return out


def _write(path: Path, rows: list[dict]):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _selected(doc: dict):
    for family, entry in doc["families"].items():
        if entry["selected"] is not None:
            yield family, entry, entry["selected"]["alarm_index"], Corroboration(**entry["selected"]["corroboration"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("sweep", "select", "replay", "relations", "trace"))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    sweep_path = OUT / "v7_alarm_definition_sweep_metropt3.jsonl"
    selection_path = OUT / "v7_alarm_definition_selection.json"
    if args.mode == "sweep":
        _write(sweep_path, sweep(args.jobs))
    elif args.mode == "select":
        rows = _read(sweep_path)
        doc = {"selection_rule": SELECTION_RULE, "fallback_rule": FALLBACK_RULE, "families": {}}
        for family in ("dataset", "calibrated"):
            chosen, how = select(rows, family)
            doc["families"][family] = {"used": how, "selected": chosen}
            print(family, how, None if chosen is None else
                  (Corroboration(**chosen["corroboration"]).label(), chosen["alarm_index"], chosen["metrics"]))
        selection_path.write_text(json.dumps(doc, indent=2) + "\n")
    elif args.mode == "replay":
        doc, rows = json.loads(selection_path.read_text()), []
        for name in ("metropt2_jan", "metropt2_apr"):
            definition = _definition(name)
            for family, entry, index, c in _selected(doc):
                for cc in (c, None):  # the selected alarm, then the same AlarmConfig uncorroborated
                    rows.append(row(definition, index, cc) | {"replay_of": family, "rule": entry["used"]})
                    m = rows[-1]["metrics"]
                    print(f"[{name}] {family} {'uncorroborated' if cc is None else cc.label()} "
                          f"events={m['events_detected']:.0f} fa={m['false_alarm_per_week']:.2f} "
                          f"clean={m['alarm_time_share_clean']:.4f} "
                          f"2h_before_lps={m.get('events_warned_2h_before_lps')}",
                          flush=True)
        _write(OUT / "v7_alarm_definition_replay_metropt2.jsonl", rows)
    elif args.mode == "relations":
        rel = relations(_read(sweep_path))
        (OUT / "v7_alarm_definition_relations.json").write_text(json.dumps(rel, indent=2) + "\n")
        print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "corroboration_ceiling_by_rule"}
                              if isinstance(v, dict) else v) for k, v in rel.items()}))
    else:
        doc, out = json.loads(selection_path.read_text()), []
        for name in ("metropt3", "metropt2_jan", "metropt2_apr"):
            definition = _definition(name)
            for family, entry, index, c in _selected(doc):
                out.append({"stream": name, "family": family, "alarm_index": index, "corroboration": asdict(c),
                            "events": event_trace(definition, ALARM_CONFIGS[index], c)})
        (OUT / "v7_alarm_definition_trace.json").write_text(json.dumps(out, indent=2) + "\n")
        for r in out:
            print(r["stream"], r["family"], [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in e.items()}
                                             for e in r["events"]])


if __name__ == "__main__":
    main()
