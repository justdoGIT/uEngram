from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean


def _json(repo: Path, name: str) -> dict:
    return json.loads((repo / "results" / name).read_text())


def _jsonl(repo: Path, name: str) -> list[dict]:
    with (repo / "results" / name).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _finite(values: list[float]) -> list[float]:
    return [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]


def _avg(values: list[float], digits: int = 2) -> str:
    usable = _finite(values)
    return f"{mean(usable):.{digits}f}" if usable else "n/a"


def _fmt(value: float | None, digits: int = 2) -> str:
    return f"{value:.{digits}f}" if value is not None and math.isfinite(value) else "n/a"


def _table(title: str, outcome: str, source: str, columns: list[str], rows: list[list[str]], note: str) -> dict:
    return {"title": title, "outcome": outcome, "source": source, "columns": columns, "rows": rows, "note": note}


def _metropt(repo: Path) -> dict:
    files = {
        "DSP d16": "metropt_runs.jsonl",
        "DSP d42": "metropt_d42_runs.jsonl",
        "DSP slope": "metropt_slope_runs.jsonl",
    }
    grouped: dict[tuple[str, str], list[dict]] = {}
    for label, filename in files.items():
        for item in _jsonl(repo, filename):
            if label == "DSP d16" and item["encoder"] != "dsp":
                continue
            grouped.setdefault((label, item["method"]), []).append(item)
    for item in _jsonl(repo, "metropt_runs.jsonl"):
        if item["encoder"] == "cnn":
            grouped.setdefault(("CNN d16", item["method"]), []).append(item)

    methods = sorted({method for _, method in grouped})
    labels = ["DSP d16", "CNN d16", "DSP d42", "DSP slope"]
    columns = ["Method"]
    for label in labels:
        columns.extend([f"{label} mean detected / total", f"{label} mean false alarms per week"])
    rows = []
    for method in methods:
        row = [method]
        for label in labels:
            runs = grouped.get((label, method), [])
            if runs:
                detected = _avg([r["metrics"].get("failures_detected", float("nan")) for r in runs], 1)
                total = len(runs[0]["details"]["failures"])
                false_rate = _avg([r["metrics"].get("false_alarm_per_week", float("nan")) for r in runs], 2)
                row.extend([f"{detected} / {total}", false_rate])
            else:
                row.extend(["—", "—"])
        rows.append(row)
    source = ", ".join(f"results/{name}" for name in files.values())
    return _table(
        "MetroPT alarm results by configuration", "negative", source,
        columns, rows,
        f"Detection and false-alarm rates are seed means across {len(methods)} methods. The table compares DSP d16, CNN d16, DSP d42, and DSP slope configurations.",
    )


def build(repo: Path) -> list[dict]:
    tables = [_metropt(repo)]

    # MetroPT-2: separate stream and dimension; values are seed means.
    metro2 = _jsonl(repo, "metropt2_d16_d24_d32_d42.jsonl")
    by_stream_encoder: dict[tuple[str, str], list[dict]] = {}
    for item in metro2:
        by_stream_encoder.setdefault((item["stream"], item["encoder"]), []).append(item)
    rows = []
    for stream, encoder in sorted(by_stream_encoder, key=lambda pair: (pair[0], int(pair[1].rsplit("d", 1)[-1]))):
        runs = by_stream_encoder[(stream, encoder)]
        metric = lambda key: _avg([r["metrics"].get(key, float("nan")) for r in runs], 3)
        clean_sec = _avg([r["details"].get("clean_alarm_seconds", float("nan")) for r in runs], 1)
        total_events = len(runs[0]["details"]["events"])
        rows.append([stream, encoder, f"{metric('events_detected')} / {total_events}", clean_sec, metric("false_alarm_per_week"), metric("auroc")])
    tables.append(_table(
        "MetroPT-2 dimension sweep", "mixed", "results/metropt2_d16_d24_d32_d42.jsonl",
        ["Stream", "Encoder / dimension", "Mean events detected / total", "Mean clean alarm time (seconds)", "Mean false alarms per week", "Mean AUROC"], rows,
        f"The {len(rows)} stream-dimension combinations report event detections, clean alarm duration in seconds, false alarms, and AUROC as means across seeds. Results vary across streams and dimensions, including missed events.",
    ))

    # Cross-domain transfer gate from the recorded frozen TSB-AD-U holdout.
    gate = _json(repo, "stronger_gate.json")["datasets"]["TSB-AD-U"]["results"]
    baseline = gate["baseline"]
    transfer = gate["variants"]
    rows = [["Frozen baseline", _fmt(baseline["auroc"], 3), _fmt(baseline["average_precision"], 3), str(baseline["windows"]), str(baseline["positive_windows"])]]
    for variant in transfer:
        overall = variant["overall"]
        rows.append([
            f"Teacher transfer ({variant['context_policy']}, instance stats={str(variant['include_instance_stats']).lower()})",
            _fmt(overall["auroc"], 3), _fmt(overall["average_precision"], 3),
            str(overall["windows"]), str(overall["positive_windows"]),
        ])
    tables.append(_table(
        "Stronger cross-dataset transfer gate", "null", "results/stronger_gate.json",
        ["Variant", "AUROC", "Average precision", "Windows", "positive windows"], rows,
        f"This table is the TSB-AD-U transfer slice of the stronger-frozen-cross-dataset gate. Baseline AUROC is {_fmt(baseline['auroc'], 3)} and transfer AUROCs range from {_fmt(min(v['overall']['auroc'] for v in transfer), 3)} to {_fmt(max(v['overall']['auroc'] for v in transfer), 3)}.",
    ))

    # CARE is large; summarize every dimension, pooling configurations and seeds.
    care = _jsonl(repo, "care_tabular_d16_d32.jsonl")
    care_by_dim: dict[str, list[dict]] = {}
    for item in care:
        care_by_dim.setdefault(item["encoder"], []).append(item)
    rows = []
    for encoder in sorted(care_by_dim):
        runs = care_by_dim[encoder]
        metrics = [r["metrics"] for r in runs]
        finite_auroc = _finite([m.get("auroc", float("nan")) for m in metrics])
        finite_fa = _finite([m.get("false_alarm_per_week", float("nan")) for m in metrics])
        valid_fraction = mean([sum(bool(v) for v in r["details"].get("valid", [])) / max(1, len(r["details"].get("valid", []))) for r in runs])
        unscored_fraction = mean([r["details"].get("unscored_windows", 0) / max(1, r["details"].get("test_windows", 0)) for r in runs])
        events = _finite([m.get("events_detected", float("nan")) for m in metrics])
        rows.append([
            encoder, str(len(runs)), _fmt(mean(finite_auroc) if finite_auroc else None, 3),
            _fmt(mean(finite_fa) if finite_fa else None, 2),
            _fmt(mean(events) if events else None, 2),
            f"{100 * valid_fraction:.1f}%", f"{100 * unscored_fraction:.1f}%",
        ])
    tables.append(_table(
        "CARE tabular comparison by dimension", "negative", "results/care_tabular_d16_d32.jsonl",
        ["Dimension", "Runs", "Mean finite AUROC", "Mean finite false alarms per week", "Mean events detected", "Mean valid fraction", "Mean unscored fraction"], rows,
        f"All {len(care)} CARE run records are pooled within dimension. Finite AUROC and false-alarm values are averaged, while valid and unscored fractions retain failed and incomplete runs.",
    ))

    # SCANIA held-out validation and its ablations/sweeps.
    validation = _json(repo, "scania_validation.json")
    independent = _json(repo, "scania_validation_independent.json")["independent"]
    control = validation["and_control"]
    rows = [[
        "and_control (scania_validation.json)", str(control["vehicles"]), str(control["alarm_vehicles"]),
        _fmt(control["healthy_alarm_rate"], 3), _fmt(control["near_failure_recall"], 3),
        _fmt(control["final_row_novelty_auroc"], 3),
    ]]
    item = independent["or_missingness_tripwire"]
    rows.append([
        "or_missingness_tripwire (scania_validation_independent.json)", str(item["vehicles"]), str(item["alarm_vehicles"]),
        _fmt(item["healthy_alarm_rate"], 3), _fmt(item["near_failure_recall"], 3), _fmt(item["final_row_novelty_auroc"], 3),
    ])
    tables.append(_table(
        "SCANIA held-out validation controls", "negative", "results/scania_validation.json, results/scania_validation_independent.json",
        ["Control key and source", "Vehicles", "Alarm vehicles", "Healthy alarm rate", "Near-failure recall", "Novelty AUROC"], rows,
        f"This is held-out SCANIA validation data with {validation['stream']['test_vehicles']} test vehicles. The independent or_no_evidence_control has the same values as and_control and is not a distinct result; the listed missingness tripwire has near-failure recall {_fmt(item['near_failure_recall'], 3)}.",
    ))

    dim_sweep = _json(repo, "scania_tab_d_sweep.json")["rows"]
    rows = []
    for item in dim_sweep:
        m = item["metrics"]
        rows.append([item["encoder"], str(m["alarm_vehicles"]), _fmt(m["healthy_alarm_rate"], 3), _fmt(m["near_failure_recall"], 3), _fmt(m["final_row_novelty_auroc"], 3)])
    tables.append(_table(
        "SCANIA dimension sweep", "negative", "results/scania_tab_d_sweep.json",
        ["Dimension", "Alarm vehicles", "Healthy alarm rate", "Near-failure recall", "Novelty AUROC"], rows,
        f"The {len(rows)} dimension settings have near-failure recalls from {_fmt(min(x['metrics']['near_failure_recall'] for x in dim_sweep), 3)} to {_fmt(max(x['metrics']['near_failure_recall'] for x in dim_sweep), 3)} and AUROCs from {_fmt(min(x['metrics']['final_row_novelty_auroc'] for x in dim_sweep), 3)} to {_fmt(max(x['metrics']['final_row_novelty_auroc'] for x in dim_sweep), 3)}.",
    ))

    reg_sweep = _json(repo, "scania_tab_reg_sweep.json")["rows"]
    rows = []
    for item in reg_sweep:
        m = item["metrics"]
        rows.append([item["encoder"], str(m["alarm_vehicles"]), _fmt(m["healthy_alarm_rate"], 3), _fmt(m["near_failure_recall"], 3), _fmt(m["final_row_novelty_auroc"], 3)])
    tables.append(_table(
        "SCANIA regularization sweep", "negative", "results/scania_tab_reg_sweep.json",
        ["Dimension", "Alarm vehicles", "Healthy alarm rate", "Near-failure recall", "Novelty AUROC"], rows,
        f"The {len(rows)} regularization settings have near-failure recalls from {_fmt(min(x['metrics']['near_failure_recall'] for x in reg_sweep), 3)} to {_fmt(max(x['metrics']['near_failure_recall'] for x in reg_sweep), 3)}. The sweep does not establish strong held-out discrimination.",
    ))
    return tables


