#!/usr/bin/env python3
"""Generate the SVG figures for the uEngram research paper.

Needs the "docs" extra (markdown, pygments, pillow): `uv sync --extra docs`.
Usage: .venv/bin/python scripts/generate_paper_figures.py

Numbers in the data figures come from committed result files:
  TSB-AD-U holdout comparison   results/moment_tsb_ad_holdout.json
  Domain-specialist comparison  results/osmetrics_specialist_gate.json, results/role_skab_gate.json,
                                results/skab_sensor_fusion_gate.json
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from paper_figures_lib import (
    BODY_SIZE, INK, SERIES, Canvas, hbar_panel, label_gap,
)

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "research-paper" / "figures"
MARGIN = 16


def fig_rag() -> Canvas:
    spec = {
        "t0": ("Chunks", ("documents / chat", "split into chunks")),
        "t1": ("Embed", ("vector", "+ metadata")),
        "t2": ("Retrieve", ("related facts", "and episodes")),
        "t3": ("LLM answer", ("grounded in evidence;", "memory merges / ages facts")),
        "s0": ("Sensor windows", ("aligned channels", "raw stream retained")),
        "s1": ("Features", ("deterministic vector", "+ time / asset ID")),
        "s2": ("Retrieve", ("similar episodes", "+ current baseline")),
        "s3": ("Diagnostic view", ("why now, what changed,", "which raw slice proves it")),
    }
    place = {f"{r}{i}": (i, row) for row, r in enumerate("ts") for i in range(4)}
    tones = {k: ("slate" if k[0] == "t" else "blue") for k in spec}
    c = Canvas(0, 0, "rag")
    b, right, bottom = c.grid(spec, place, tones, MARGIN, 34, [30] * 3, [54])
    c.heading(MARGIN, 22, "Text RAG + memory layer")
    c.heading(MARGIN, b["s0"].y - 12, "Sensor-state memory (proposed DB capability)")
    for r in "ts":
        for i in range(3):
            c.arrow([b[f"{r}{i}"].right_mid(), b[f"{r}{i + 1}"].left_mid()])
    c.text(MARGIN, bottom + 24, "Sensor baselines update only under a safe admission rule; "
           "the sensor-state DBMS is proposed, not implemented.", 10, italic=True)
    c.width, c.height = right + MARGIN, bottom + 34
    return c


def fig_architecture() -> Canvas:
    spec = {
        "win": ("Sensor window", ("C channels", "× K samples")),
        "enc": ("Encoder", ("DSP · CNN · roles", "PCA + noise whitening")),
        "slots": ("uEngram slots", ("semantic", "candidate · episodic")),
        "db": ("Event store", ("event records", "raw-window pins")),
        "op": ("Operator", ("optional LLM", "after retrieval")),
        "diag": ("Pre-fit diagnostics", ("rank · condition", "sentinel rates")),
        "fit": ("Fit encoder", ("PCA + MAD", "noise covariance N")),
        "teach": ("Teacher distillation", ("optional, offline", "→ 46 KB student")),
    }
    place = {"win": (0, 0), "enc": (1, 0), "slots": (2, 0), "db": (3, 0), "op": (4, 0),
             "diag": (0, 1), "fit": (1, 1), "teach": (2, 1)}
    tones = {k: ("blue" if place[k][1] == 0 else "violet") for k in spec}
    c = Canvas(0, 0, "arch")
    b, right, bottom = c.grid(spec, place, tones, MARGIN, 34, [26] * 4,
                              [74], dashed=("diag", "fit", "teach"))
    c.heading(MARGIN, 22, "Edge path · per window, deterministic")
    c.heading(MARGIN, b["diag"].y - 10, "Commissioning · offline", color="#6d28d9")
    for a, z in (("win", "enc"), ("enc", "slots"), ("slots", "db"), ("db", "op")):
        c.arrow([b[a].right_mid(), b[z].left_mid()])
    c.arrow([b["diag"].right_mid(), b["fit"].left_mid()])
    c.arrow([b["teach"].left_mid(), b["fit"].right_mid()], dash="4 3")
    c.arrow([b["fit"].top_mid(), b["enc"].bottom_mid()], label="frozen parameters",
            label_at=(b["enc"].cx, b["enc"].bottom + 24))
    c.width, c.height = right + MARGIN, bottom + 16
    return c


def fig_slots() -> Canvas:
    spec = {
        "z": ("Whitened window z", ()),
        "gate": ("Gate test", ("d² ≤ τ for a", "stored slot?")),
        "sem": ("Semantic slot", ("EW mean / variance", "α = max(1/(n+1), β_min)")),
        "sup": ("Superseded", ("old baseline kept", "for history")),
        "cand": ("Candidate", ("probation;", "stays replaceable")),
        "epi": ("Episodic", ("rare excursion", "+ raw pointer")),
        "adm": ("Competitive admission", ("full bank: admit only if", "surprisal > weakest episode")),
        "drop": ("Discard", ("no slot written",)),
    }
    place = {"z": (0, 1), "gate": (1, 1), "sem": (2, 0), "cand": (2, 1), "adm": (2, 2),
             "sup": (3, 0), "epi": (3, 1), "drop": (3, 2)}
    tones = {"sem": "green", "cand": "amber", "epi": "red", "sup": "slate", "adm": "white",
             "drop": "slate", "z": "blue", "gate": "white"}
    c = Canvas(0, 0, "slots")
    col_gaps = [36, label_gap("other gate hit"),
                label_gap("MEWMA step", "not confirmed", "reject")]
    b, right, bottom = c.grid(spec, place, tones, MARGIN, 26, col_gaps, [52, 52])
    c.arrow([b["z"].right_mid(), b["gate"].left_mid()])
    gx = b["gate"].cx
    c.arrow([b["gate"].top_mid(), (gx, b["sem"].cy), b["sem"].left_mid()],
            label="semantic gate", label_at=((gx + b["sem"].x) / 2, b["sem"].cy))
    c.arrow([b["gate"].right_mid(), b["cand"].left_mid()], label="other gate hit")
    c.arrow([b["gate"].bottom_mid(), (gx, b["adm"].cy), b["adm"].left_mid()],
            label="no gate", label_at=((gx + b["adm"].x) / 2, b["adm"].cy))
    c.arrow([b["sem"].right_mid(), b["sup"].left_mid()], label="MEWMA step")
    c.arrow([b["cand"].top_mid(), b["sem"].bottom_mid()], label="5 hits in 8 steps")
    c.arrow([b["cand"].right_mid(), b["epi"].left_mid()], label="not confirmed")
    c.arrow([b["adm"].top_mid(), b["cand"].bottom_mid()], label="admit")
    c.arrow([b["adm"].right_mid(), b["drop"].left_mid()], label="reject")
    lane = (b["sem"].bottom + b["cand"].y) / 2
    hook = b["sem"].right - 22
    c.arrow([b["epi"].top_mid(), (b["epi"].cx, lane), (hook, lane), (hook, b["sem"].bottom)],
            label="20 hits in 30 steps", label_at=((b["epi"].cx + hook) / 2, lane))
    c.text(MARGIN, bottom + 24, "Default parameters: promotion m = 5 hits in n = 8 steps, "
           "re-confirmation m = 20 hits in n = 30 steps, minimum adaptation rate β_min = 0.01.", 10,
           italic=True)
    c.width, c.height = right + MARGIN, bottom + 34
    return c


def fig_persistence() -> Canvas:
    spec = {
        "win": ("Sensor window", ("validate schema", "step must increase")),
        "eng": ("Engine update", ("encode + lifecycle",)),
        "gate": ("Persist gate", ("action, alarm edge,", "or lifecycle event?")),
        "skip": ("Receipt only", ("no storage write",)),
        "pin": ("Pin raw slice", ("before the write",)),
        "rem": ("Remember event", ("embedded event store", "idempotency key")),
        "ok": ("Receipt", ("record id, persisted flag",)),
        "fail": ("Unpin + raise", ("error carries", "the pending record")),
        "retry": ("Retry pending", ("retry storage;", "never re-ingest")),
        "recall": ("Recall (read path)", ("engine state", "+ keyword history")),
    }
    place = {"win": (0, 0), "eng": (1, 0), "gate": (2, 0), "skip": (3, 0), "pin": (2, 1),
             "rem": (2, 2), "ok": (3, 2), "fail": (2, 3), "retry": (3, 3), "recall": (0, 2)}
    tones = {"win": "blue", "eng": "blue", "gate": "white", "skip": "slate", "pin": "amber",
             "rem": "green", "ok": "green", "fail": "red", "retry": "slate", "recall": "slate"}
    c = Canvas(0, 0, "persist")
    b, right, bottom = c.grid(spec, place, tones, MARGIN, 16, [30, 30, label_gap("no", "ok")],
                              [34, 34, 40])
    for a, z in (("win", "eng"), ("eng", "gate"), ("pin", "rem"), ("fail", "retry")):
        if place[a][1] == place[z][1]:
            c.arrow([b[a].right_mid(), b[z].left_mid()])
        else:
            c.arrow([b[a].bottom_mid(), b[z].top_mid()])
    c.arrow([b["gate"].right_mid(), b["skip"].left_mid()], label="no")
    c.arrow([b["gate"].bottom_mid(), b["pin"].top_mid()], label="yes")
    c.arrow([b["rem"].right_mid(), b["ok"].left_mid()], label="ok")
    c.arrow([b["rem"].bottom_mid(), b["fail"].top_mid()], label="write fails")
    edge_x = b["retry"].right + 14
    c.arrow([b["retry"].right_mid(), (edge_x, b["retry"].cy), (edge_x, b["pin"].cy),
             b["pin"].right_mid()], dash="4 3")
    c.width, c.height = edge_x + MARGIN, bottom + 16
    return c


def fig_tsb() -> Canvas:
    data = json.loads((ROOT / "results" / "moment_tsb_ad_holdout.json").read_text())
    models = [("Ordinary CNN", data["baseline"]["overall"], data["baseline"]["per_series"])]
    names = {"edge": "MOMENT, edge padding", "linear": "MOMENT, resampled + stats"}
    for v in data["variants"]:
        models.append((names[v["context_policy"]], v["scores"]["overall"], v["scores"]["per_series"]))

    def metric(entry, key, equal):
        per = entry[2]
        return statistics.mean(s[key] for s in per.values()) if equal else entry[1][key]

    c = Canvas(720, 250, "tsb")
    for m, (name, color) in enumerate(zip([m[0] for m in models], SERIES[:3])):
        lx = MARGIN + m * 200
        c.rect(lx, 10, 12, 12, color, rx=2)
        c.text(lx + 18, 20, name, 10.5, color=INK)
    for p, (key, title) in enumerate((("auroc", "AUROC"), ("average_precision", "Average precision"))):
        px = MARGIN + p * 356
        c.heading(px, 50, title)
        bar_x, bar_w = px + 96, 230
        for g, (equal, label) in enumerate(((False, "Pooled"), (True, "Equal-series"))):
            gy = 66 + g * 88
            c.text(px, gy + 28, label, BODY_SIZE, True, INK)
            for m, (entry, color) in enumerate(zip(models, SERIES[:3])):
                v = metric(entry, key, equal)
                y = gy + m * 22
                c.rect(bar_x, y + 4, bar_w * v, 15, color, rx=3)
                c.text(bar_x + bar_w * v + 5, y + 16, f"{v:.4f}", 10, True, INK)
    c.text(MARGIN, 240, "Pooled metrics weight long series; equal-series averages give each of "
           "the six series one vote.", 10, italic=True)
    return c


def fig_specialists() -> Canvas:
    os_gate = json.loads((ROOT / "results" / "osmetrics_specialist_gate.json").read_text())
    skab = json.loads((ROOT / "results" / "role_skab_gate.json").read_text())["policies"]
    fusion = json.loads((ROOT / "results" / "skab_sensor_fusion_gate.json").read_text())["results"]
    cnn_hits = max(r["events_detected"] for r in fusion.values())
    cnn_total = skab["dynamic_health"]["events_total"]
    cnn_fa = [r["false_alarms_per_hour"] for r in fusion.values()]
    dsp = os_gate["baseline_dsp_d16"]
    role = os_gate["baseline_all_channel_role"]["dynamic_health"]
    spec = os_gate["domain_specialists"]["policies"]["or_dynamic_health"]

    def os_row(label, r, color):
        n, total = r["events_detected_strict"], r["events_total"]
        return (label, n / total, f"{100 * n / total:.1f}% ({n}/{total})",
                f"{r['false_alarm_episodes_per_week']:.1f} false alarms / week", color)

    def skab_row(label, r, color):
        n, total = r["events_detected"], r["events_total"]
        return (label, n / total, f"{100 * n / total:.1f}% ({n}/{total})",
                f"{r['false_alarms_per_hour']:.2f} false alarms / hour", color)

    pct = lambda v: f"{100 * v:.0f}%"  # noqa: E731
    panels = [
        ("(A) OS-Metrics, 5 channels: strict event recall", [
            os_row("DSP baseline (d = 16)", dsp, SERIES[0]),
            os_row("All-channel role encoder (dynamic–health)", role, SERIES[1]),
            os_row("Domain specialists (OR of dynamic–health)", spec, SERIES[2]),
        ]),
        ("(B) SKAB, 8 channels: strict in-interval recall", [
            ("8-channel CNN (all three readouts)", cnn_hits / cnn_total,
             f"{100 * cnn_hits / cnn_total:.1f}% ({cnn_hits}/{cnn_total})",
             f"{min(cnn_fa):.2f}–{max(cnn_fa):.2f} false alarms / hour", SERIES[0]),
            skab_row("Role separation (dynamic–health)", skab["dynamic_health"], SERIES[1]),
            skab_row("Role separation (at least 2 branches)", skab["at_least_2"], SERIES[2]),
        ]),
    ]
    all_rows = [row for _, rows in panels for row in rows]
    c = Canvas(720, 0, "specialists")
    y = 18
    for title, rows in panels:
        y = hbar_panel(c, MARGIN, y, 688, title, rows, 1.0, pct, label_rows=all_rows) + 26
    c.height = y - 14
    return c


FIGURES_OUT = {
    "rag-sensor-memory.svg": fig_rag,
    "system-architecture.svg": fig_architecture,
    "slot-state-machine.svg": fig_slots,
    "sensor-persistence-pipeline.svg": fig_persistence,
    "tsb-holdout-comparison.svg": fig_tsb,
    "domain-specialist-comparison.svg": fig_specialists,
}


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name, build in FIGURES_OUT.items():
        canvas = build()
        (FIGURES / name).write_text(canvas.svg(), encoding="utf-8")
        print(f"{name}: {canvas.width:.0f} x {canvas.height:.0f}")
