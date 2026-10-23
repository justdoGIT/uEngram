"""Protocol v2 sections of docs/RESULTS.md (called by bench.report).

MetroPT2: one table per stream (method x policy, mean over seeds) and the
pre-registered criteria for uEngram. CARE: the CARE score and sub-scores per
(method, policy) with 95% cluster-bootstrap CIs resampling assets within farms,
the paired uEngram v2 - v1 difference, pooled
false-alarm burden on the normal datasets and event detection.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

from bench.care_score import final_score

N_BOOT, SEED = 2000, 12345
MP2_COLS = ("events_detected", "false_alarm_per_week", "alarm_time_share_clean", "alerts_per_week",
            "operator_events_per_week", "notifications_per_week_all", "lead_hours",
            "events_warned_before_lps", "hours_before_lps",
            "events_retained", "auroc", "health_auroc")


def _load(path):
    return [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []


def _fmt(x, pct=False):
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{100 * x:.2f}%" if pct else (f"{x:.0f}" if abs(x) >= 1000 else f"{x:.3g}")


def metropt2_section(rows) -> list[str]:
    out = []
    for stream in sorted({r["stream"] for r in rows}):
        grp = defaultdict(list)
        for r in rows:
            if r["stream"] == stream and r["policy"] != "secondary":
                grp[(r["method"], r["encoder"], r["policy"])].append(r["metrics"])
        n_events = max((len(r["details"]["events"]) for r in rows if r["stream"] == stream), default=0)
        out += [f"### `{stream}` ({n_events} reported events)", "",
                "| method | encoder | policy | n | " + " | ".join(MP2_COLS) + " |",
                "|" + "---|" * (len(MP2_COLS) + 4)]
        for (meth, enc, pol), ms in sorted(grp.items()):
            cells = []
            for c in MP2_COLS:
                v = [m.get(c, np.nan) for m in ms]
                v = [x for x in v if x is not None and np.isfinite(x)]
                cells.append(_fmt(np.mean(v) if v else np.nan, pct=c == "alarm_time_share_clean")
                             + (f" ({_fmt(min(v))}-{_fmt(max(v))})" if len(v) > 1 and np.ptp(v) > 0 else ""))
            out.append(f"| {meth} | {enc} | {pol} | {len(ms)} | " + " | ".join(cells) + " |")
        out.append("")
        for r in rows:
            if r["stream"] == stream and r["method"] == "uengram" and r["policy"] == "v2" and r["seed"] == 0:
                m = r["metrics"]
                q1 = m["events_detected"] >= n_events
                q2 = m["false_alarm_per_week"] <= 1 and m["alarm_time_share_clean"] <= 0.02
                q3 = m.get("events_retained", 0) >= n_events
                out += [f"uEngram ({r['encoder']}) v2 criteria: Q1 all events detected: "
                        f"{'PASS' if q1 else 'FAIL'}; Q2 false-alarm budget: {'PASS' if q2 else 'FAIL'}; "
                        f"Q3 all events in live provenance: {'PASS' if q3 else 'FAIL'}", ""]
                known = [e.get("hours_before_lps") for e in r["details"]["events"]
                         if "hours_before_lps" in e]
                if known:
                    q4 = all(v is not None and v >= 2 for v in known)
                    out += [f"Q4 at least 2 h before every published LPS warning: "
                            f"{'PASS' if q4 else 'FAIL'} ({len(known)} warnings).", ""]
        for r in rows:
            if r["stream"] == stream and r["method"] == "leak_duty":
                m = r["metrics"]
                out += [f"Secondary (parked leak duty): detected {_fmt(m['events_detected'])}/{n_events}, "
                        f"{_fmt(m['false_alarm_per_week'])} false-alarm episodes per clean **parked** week, "
                        f"{r['details']['test_parked_windows']} parked test windows.", ""]
    return out


def _care_events(rows, method, encoder, policy):
    sel = [r for r in rows if (r["method"], r["encoder"], r["policy"]) == (method, encoder, policy)]
    return {r["stream"]: r for r in sel}


def _boot(streams, events, fn, rng):
    blocks = defaultdict(lambda: defaultdict(list))
    for s in streams:
        meta = events[s]["details"]["stream_metadata"]
        blocks[meta["farm"]][meta["asset"]].append(s)
    vals = []
    for _ in range(N_BOOT * 10):
        pick = []
        for assets in blocks.values():
            keys = sorted(assets)
            for asset in rng.choice(keys, len(keys)):
                pick.extend(assets[asset])
        labels = {events[s]["details"]["care_event"]["label"] for s in pick}
        if labels == {"anomaly", "normal"}:
            vals.append(fn(pick))
        if len(vals) == N_BOOT:
            return np.percentile(vals, [2.5, 97.5])
    raise ValueError("insufficient class-complete cluster bootstrap replicates")


def care_section(rows) -> list[str]:
    keys = sorted({(r["method"], r["encoder"], r["policy"]) for r in rows})
    out = ["| method | encoder | policy | datasets | CARE [95% CI] | coverage | accuracy | earliness | "
           "reliability | anomaly events alarmed | criticality detections | normal datasets flagged | "
           "false alarms / normal week | clean time in alarm (normal) | notifications / all observed week |",
           "|" + "---|" * 15]
    by_key = {k: _care_events(rows, *k) for k in keys}
    for k in keys:
        ev = by_key[k]
        streams = sorted(ev)
        score = lambda ss: final_score([ev[s]["details"]["care_event"] for s in ss])["care"]
        fs = final_score([ev[s]["details"]["care_event"] for s in streams])
        lo, hi = _boot(streams, ev, score, np.random.default_rng(SEED))
        anom = [s for s in streams if ev[s]["details"]["care_event"]["label"] == "anomaly"]
        norm = [s for s in streams if s not in anom]
        alarmed = sum(ev[s]["metrics"]["events_detected"] for s in anom)
        fa = sum(ev[s]["details"]["false_alarm_episodes"] for s in norm)
        weeks = sum(ev[s]["details"]["clean_observed_hours"] for s in norm) / 168
        share = np.average([ev[s]["metrics"]["alarm_time_share_clean"] for s in norm],
                           weights=[ev[s]["details"]["clean_observed_hours"] for s in norm])
        all_hours = sum(ev[s]["details"]["runtime_observed_hours"] for s in streams)
        notification_rate = sum(ev[s]["metrics"]["notifications_per_week_all"] *
                                ev[s]["details"]["runtime_observed_hours"] for s in streams) / all_hours
        out.append(f"| {k[0]} | {k[1]} | {k[2]} | {len(streams)} | {fs['care']:.3f} [{lo:.3f}, {hi:.3f}] | "
                   f"{fs['coverage']:.3f} | {fs['accuracy']:.3f} | {fs['earliness']:.3f} | "
                   f"{fs['reliability']:.3f} | {alarmed:.0f}/{len(anom)} | "
                   f"{fs['events_detected_by_criticality']}/{len(anom)} | {fs['normal_events_flagged']}/{len(norm)} | "
                   f"{fa / weeks:.2f} | {100 * share:.2f}% | {notification_rate:.2f} |")
    out.append("")
    for meth, enc in sorted({(m, e) for m, e, _ in keys}):
        v1, v2 = by_key.get((meth, enc, "v1_rule")), by_key.get((meth, enc, "v2"))
        if not v1 or not v2:
            continue
        streams = sorted(set(v1) & set(v2))
        diff = lambda ss: (final_score([v2[s]["details"]["care_event"] for s in ss])["care"]
                           - final_score([v1[s]["details"]["care_event"] for s in ss])["care"])
        lo, hi = _boot(streams, v2, diff, np.random.default_rng(SEED))
        star = "*" if lo > 0 or hi < 0 else ""
        out.append(f"- {meth} ({enc}) CARE v2 - v1 rule: {diff(streams):+.3f} [{lo:+.3f}, {hi:+.3f}]{star}")
    out.append("")
    return out


def build_v2(runs_dir: str) -> list[str]:
    mp2 = _load(f"{runs_dir}/v2_metropt2.jsonl")
    care = _load(f"{runs_dir}/v2_care.jsonl")
    failures = [r for r in mp2 + care if r.get("status") == "invalid_calibration"]
    blocked = {(r["method"], r["encoder"]) for r in care if r.get("status") == "invalid_calibration"}
    mp2 = [r for r in mp2 if r.get("status") != "invalid_calibration"]
    care = [r for r in care if r.get("status") != "invalid_calibration"
            and (r["method"], r["encoder"]) not in blocked]
    out = ["## Protocol v2: alarm policy confirmation", "",
           "Protocol and pass criteria in `docs/EXPLAINED.md` section 17. Each method runs one continuous "
           "memory per stream; `v1_rule` alarms on every window above the alpha = 0.01 calibration "
           "threshold, `v2` is the method's alarm configuration frozen on MetroPT-3. False-alarm episodes "
           "merge openings less than 1 h apart; clean time in alarm is the duration-weighted share of "
           "eligible clean exposure. Cells are means over seeds (range when seeds differ).", ""]
    if failures:
        out += [f"**Invalid-calibration runs: {len(failures)}.** No zero-alarm fallback was used. "
                "A CARE method with any missing dataset has no aggregate score or CI.", ""]
        out += [f"- `{r['stream']}` {r['method']}/{r['encoder']} seed {r['seed']}: {r['error']}"
                for r in failures[:10]]
        out += [""]
    out += ["### MetroPT2", ""] + (metropt2_section(mp2) if mp2 else ["not run", ""])
    out += ["### CARE to Compare", "",
            "CARE score as published (Fraunhofer IEE reference implementation, criticality threshold 72); "
            "CIs resample assets within farms (all event datasets of an asset stay together); "
            "class-incomplete bootstrap draws are rejected. `*` marks an exploratory paired CI excluding 0; "
            "no multiplicity correction or inference to unseen farms.", ""]
    out += care_section(care) if care else ["not run", ""]
    return out
