"""Sweep, select and replay the error-gated AlarmPolicy alarm (bench/error_gated_alarm.py).

Modes, in order (all on frozen seed-0 DSP uEngram scores, no rescoring):
  sweep      every ALARM_CONFIG x (ungated + every GATE) on MetroPT3
  select     apply the rules below, separately per error family, to the MetroPT3 rows
  replay     apply each selected config, unchanged, to both MetroPT2 recordings
  relations  exact checks: identity, AND-monotonicity, error ceilings, nesting in L and m

ALARM_CONFIGS starts with the actual alarm definition (the frozen v2 AlarmConfig, v4
BASELINE), then every AlarmConfig the v4 study explored. Only gated rows (m >= 1) are
selection candidates; ungated rows are the reference.

Rules, fixed before any gated row was computed:
  SELECTION_RULE  events_detected == 4, alarm_time_share_clean <= 0.02 and
                  false_alarm_per_week <= 1; then min clean share, earliest first alarm
                  (smaller lead_hours), shorter lookback, smaller m, earlier ALARM_CONFIG
  FALLBACK_RULE   if no row meets it: rows meeting both alarm budgets
                  (clean share <= 0.02, false alarms <= 1/week), max events_detected, then
                  the same tie order. Replayed only to test transfer.
MetroPT2 is never used to choose or change anything.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

from bench.alarm_eval import CalibrationUnavailable
from bench.alarm_sensitivity import BASELINE, OFAT_GRIDS
from bench.error_gated_alarm import Gate, Replay
from bench.streams import load_stream

OUT = Path("results")
FAMILIES = ("dataset", "calibrated")
GATES = tuple(Gate(f, m, L) for f in FAMILIES for m in (1, 2, 3) for L in (0, 1, 3, 6, 12, 24))
METRICS = ("events_detected", "false_alarm_per_week", "alarm_time_share_clean", "lead_hours",
           "notifications_per_week_all", "alerts_per_week", "events_warned_before_lps",
           "hours_before_lps")
SELECTION_RULE = ("gated rows, events_detected==4 AND alarm_time_share_clean<=0.02 AND "
                  "false_alarm_per_week<=1; min clean share, earliest first alarm, "
                  "shorter lookback, smaller m, earlier ALARM_CONFIG")
FALLBACK_RULE = ("gated rows with alarm_time_share_clean<=0.02 AND false_alarm_per_week<=1; "
                 "max events_detected, then the SELECTION_RULE tie order; replayed only to "
                 "test transfer")


def _alarm_configs() -> list:
    out = [BASELINE]
    out += [replace(BASELINE, **{name: v}) for name, grid in OFAT_GRIDS.items() for v in grid]
    out += [replace(BASELINE, health_needs_baseline=False, k_health=kh, new_state_alert=nsa)
            for nsa in (True, False) for kh in (1, 3, 6, 10)]
    out += [replace(BASELINE, health_needs_baseline=False, k_health=kh, k_novelty=kn, alpha=a)
            for kh in (1, 3, 6) for kn in (3, 4, 6) for a in (0.003, 0.01)]
    return list(dict.fromkeys(out))


ALARM_CONFIGS = _alarm_configs()


def _clean(v):
    return None if isinstance(v, float) and math.isnan(v) else v


def row(replay: Replay, index: int, gate: Gate | None) -> dict:
    cfg = ALARM_CONFIGS[index]
    base = {"stream": replay.stream.name, "alarm_index": index, "alarm_config": asdict(cfg),
            "gate": None if gate is None else asdict(gate)}
    try:
        m, d = replay.evaluate(cfg, gate)
    except CalibrationUnavailable as exc:
        return base | {"calibration_unavailable": str(exc), "metrics": None,
                       "satisfies_recall_and_budget": False}
    metrics = {k: _clean(float(m[k])) for k in METRICS if k in m}
    ok = (m["events_detected"] == len(replay.stream.events) and m["alarm_time_share_clean"] <= 0.02
          and m["false_alarm_per_week"] <= 1.0)
    events = [{k: e.get(k) for k in ("event", "detected", "lead_hours", "alarm_windows", "windows",
                                       "hours_before_lps")} for e in d["events"]]
    return base | {"calibration_unavailable": None, "metrics": metrics, "events": events,
                   "policy_events_detected": d["policy_events_detected"],
                   "error_ceiling_events": d["error_ceiling_events"],
                   "condition_share_clean": d["condition_share_clean"],
                   "satisfies_recall_and_budget": ok}


def sweep(replay: Replay) -> list[dict]:
    rows = []
    for i in range(len(ALARM_CONFIGS)):
        rows.append(row(replay, i, None))
        if rows[-1]["metrics"] is None:
            print(f"[sweep] config {i}: calibration unavailable", flush=True)
            continue
        rows += [row(replay, i, g) for g in GATES]
        best = max((r for r in rows[-len(GATES):] if r["gate"]["family"] == "dataset"),
                   key=lambda r: (r["metrics"]["events_detected"], -r["metrics"]["alarm_time_share_clean"]))
        m0, mb = rows[-len(GATES) - 1]["metrics"], best["metrics"]
        print(f"[sweep] config {i}: policy {m0['events_detected']:.0f} ev "
              f"clean={m0['alarm_time_share_clean']:.4f} fa={m0['false_alarm_per_week']:.2f} | "
              f"best dataset gate {Gate(**best['gate']).label()} {mb['events_detected']:.0f} ev "
              f"clean={mb['alarm_time_share_clean']:.4f} fa={mb['false_alarm_per_week']:.2f}", flush=True)
    return rows


def _tie(r: dict) -> tuple:
    lead = r["metrics"]["lead_hours"]
    return (r["metrics"]["alarm_time_share_clean"], lead if lead is not None else 1e9,
            r["gate"]["lookback_h"], r["gate"]["min_errors"], r["alarm_index"])


def select(rows: list[dict], family: str) -> tuple[dict | None, str]:
    cand = [r for r in rows if r["metrics"] is not None and r["gate"] is not None
            and r["gate"]["family"] == family]
    primary = [r for r in cand if r["satisfies_recall_and_budget"]]
    if primary:
        return min(primary, key=_tie), "primary"
    budget = [r for r in cand if r["metrics"]["alarm_time_share_clean"] <= 0.02
              and r["metrics"]["false_alarm_per_week"] <= 1.0]
    if not budget:
        return None, "none"
    return min(budget, key=lambda r: (-r["metrics"]["events_detected"],) + _tie(r)), "fallback"


def relations(rows: list[dict]) -> dict:
    """Exact properties checked on every MetroPT3 row."""
    ref = {r["alarm_index"]: r for r in rows if r["gate"] is None and r["metrics"] is not None}
    gated = [r for r in rows if r["gate"] is not None and r["metrics"] is not None]
    tol = 1e-12
    out = {"gated_rows": len(gated),
           "clean_share_above_policy": sum(r["metrics"]["alarm_time_share_clean"] >
                                           ref[r["alarm_index"]]["metrics"]["alarm_time_share_clean"] + tol
                                           for r in gated),
           "events_above_policy": sum(r["metrics"]["events_detected"] > r["policy_events_detected"]
                                      for r in gated),
           "events_above_error_ceiling": sum(r["metrics"]["events_detected"] > r["error_ceiling_events"]
                                             for r in gated),
           "false_alarms_above_policy": sum(r["metrics"]["false_alarm_per_week"] >
                                            ref[r["alarm_index"]]["metrics"]["false_alarm_per_week"] + tol
                                            for r in gated)}
    by = {(r["alarm_index"], r["gate"]["family"], r["gate"]["min_errors"], r["gate"]["lookback_h"]): r
          for r in gated}
    lookbacks = sorted({k[3] for k in by})

    def share(key):
        return by[key]["metrics"]["alarm_time_share_clean"]

    out["clean_share_decreasing_in_lookback"] = sum(
        share((i, f, m, b)) + tol < share((i, f, m, a))
        for (i, f, m, a) in by for b in lookbacks if b > a and (i, f, m, b) in by)
    out["clean_share_increasing_in_m"] = sum(
        share((i, f, m + 1, L)) > share((i, f, m, L)) + tol
        for (i, f, m, L) in by if (i, f, m + 1, L) in by)
    out["error_ceiling_by_gate"] = {Gate(**r["gate"]).label(): r["error_ceiling_events"]
                                    for r in gated if r["alarm_index"] == 0}
    return out


def _write(path: Path, rows: list[dict]):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("sweep", "select", "replay", "relations"))
    args = parser.parse_args()
    sweep_path = OUT / "v6_error_gated_sweep_metropt3.jsonl"
    selection_path = OUT / "v6_error_gated_selection.json"
    if args.mode == "sweep":
        _write(sweep_path, sweep(Replay(load_stream("metropt3"))))
    elif args.mode == "select":
        rows = _read(sweep_path)
        doc = {"selection_rule": SELECTION_RULE, "fallback_rule": FALLBACK_RULE,
               "alarm_configs": len(ALARM_CONFIGS), "gates": len(GATES), "families": {}}
        for family in FAMILIES:
            chosen, how = select(rows, family)
            doc["families"][family] = {"used": how, "selected": chosen,
                                       "primary_candidates": sum(r["satisfies_recall_and_budget"] for r in rows
                                                                 if r["gate"] and r["gate"]["family"] == family)}
            print(family, how, None if chosen is None else
                  (Gate(**chosen["gate"]).label(), chosen["alarm_index"], chosen["metrics"]))
        selection_path.write_text(json.dumps(doc, indent=2) + "\n")
    elif args.mode == "replay":
        doc = json.loads(selection_path.read_text())
        rows = []
        for name in ("metropt2_jan", "metropt2_apr"):
            replay = Replay(load_stream(name))
            for family, entry in doc["families"].items():
                if entry["selected"] is None:
                    continue
                index, gate = entry["selected"]["alarm_index"], Gate(**entry["selected"]["gate"])
                for g in (gate, None):  # the selected gated alarm, then its ungated policy
                    rows.append(row(replay, index, g) | {"replay_of": family, "rule": entry["used"]})
                    m = rows[-1]["metrics"]
                    print(f"[{name}] {family} {'ungated' if g is None else g.label()} "
                          f"events={m['events_detected']:.0f} fa={m['false_alarm_per_week']:.2f} "
                          f"clean={m['alarm_time_share_clean']:.4f} lead={m['lead_hours']} "
                          f"before_lps={m.get('events_warned_before_lps')} ceiling={rows[-1]['error_ceiling_events']}",
                          flush=True)
        _write(OUT / "v6_error_gated_replay_metropt2.jsonl", rows)
    else:
        rel = relations(_read(sweep_path))
        (OUT / "v6_error_gated_relations.json").write_text(json.dumps(rel, indent=2) + "\n")
        print(json.dumps({k: v for k, v in rel.items() if k != "error_ceiling_by_gate"}))


if __name__ == "__main__":
    main()
