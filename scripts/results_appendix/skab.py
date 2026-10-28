from __future__ import annotations

import json
from pathlib import Path


def _load(repo: Path, name: str) -> dict:
    return json.loads((repo / "results" / name).read_text(encoding="utf-8"))


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _seed_note(data: dict) -> str:
    if "seed" not in data or data["seed"] is None:
        return "Seed count is not reported in this file."
    seeds = data["seed"]
    if isinstance(seeds, list):
        return f"Multi-seed result ({len(seeds)} seeds)."
    return f"Single-seed result (seed {seeds})."


def _table(title: str, source: str, outcome: str, columns: list[str], rows: list[list[str]], note: str) -> dict:
    return {
        "title": title,
        "outcome": outcome,
        "source": source,
        "columns": columns,
        "rows": rows,
        "note": note,
    }


def build(repo: Path) -> list[dict]:
    tables = []
    columns = ["Setting", "Events detected / total", "False alarm episodes", "AUROC", "AP"]

    validation = _load(repo, "skab_validation.json")
    validation_metrics = validation["metrics"]
    validation_details = validation["details"]
    total = validation["n_experiments"]
    validation_detected = int(validation_metrics["events_detected"])
    validation_false_alarms = validation_details["false_alarm_episodes"]
    validation_row = [
        validation["encoder"],
        f"{validation_detected} / {total}",
        str(validation_false_alarms),
        _fmt(validation_metrics["auroc"]),
        _fmt(validation_metrics["ap"]),
    ]

    slope = _load(repo, "skab_slope_ablation.json")
    slope_metrics = slope["metrics"]
    slope_details = slope["details"]
    slope_total = slope["n_experiments"]
    slope_detected = int(slope_metrics["events_detected"])
    slope_false_alarms = slope_details["false_alarm_episodes"]
    slope_row = [
        slope["encoder"],
        f"{slope_detected} / {slope_total}",
        str(slope_false_alarms),
        _fmt(slope_metrics["auroc"]),
        _fmt(slope_metrics["ap"]),
    ]
    tables.append(_table(
        "SKAB frozen validation and slope ablation",
        "results/skab_validation.json, results/skab_slope_ablation.json",
        "mixed",
        columns,
        [validation_row, slope_row],
        f"The slope ablation detected {slope_detected} of {slope_total} experiments versus {validation_detected} of {total} for frozen validation, with {slope_false_alarms} versus {validation_false_alarms} false alarm episodes. Frozen validation: {_seed_note(validation)} Slope ablation: {_seed_note(slope)}",
    ))

    data = _load(repo, "skab_d_sweep.json")
    rows = []
    for item in data["rows"]:
        metrics = item["metrics"]
        detected = int(metrics["events_detected"])
        rows.append([
            item["encoder"],
            f"{detected} / {total}",
            str(item["false_alarm_episodes"]),
            _fmt(metrics["auroc"]),
            _fmt(metrics["ap"]),
        ])
    best = max(data["rows"], key=lambda item: item["metrics"]["events_detected"])
    tables.append(_table(
        "SKAB embedding-dimension sweep",
        "results/skab_d_sweep.json, results/skab_validation.json",
        "mixed",
        columns,
        rows,
        f"The sweep shows all {len(rows)} embedding settings; {best['encoder']} detected {int(best['metrics']['events_detected'])} of {total} experiments. Seed count is not reported in this file.",
    ))

    data = _load(repo, "skab_persistence_sweep.json")
    baseline = next(item for item in data["rows"] if "baseline" in item["config"])
    rows = []
    for item in data["rows"]:
        detected = int(item["events_detected"])
        setting = item["config"] + (" (baseline)" if item is baseline else "")
        rows.append([
            setting,
            f"{detected} / {total}",
            str(item["false_alarm_episodes"]),
            _fmt(item["auroc"]),
            _fmt(item["ap"]),
        ])
    tables.append(_table(
        "SKAB persistence and threshold sweep",
        "results/skab_persistence_sweep.json, results/skab_validation.json",
        "mixed",
        columns,
        rows,
        f"All {len(rows)} persistence settings are shown, including settings that lose events; the baseline detected {int(baseline['events_detected'])} of {total}, and the best setting detected {max(int(item['events_detected']) for item in data['rows'])}. {_seed_note(data)}",
    ))

    data = _load(repo, "skab_shock_sweep.json")
    baseline = min(data["rows"], key=lambda item: item["shock_mult"])
    best = max(data["rows"], key=lambda item: item["events_detected"])
    auroc_values = {item["metrics"]["auroc"] for item in data["rows"]}
    ap_values = {item["metrics"]["ap"] for item in data["rows"]}
    rows = []
    for item in data["rows"]:
        detected = int(item["events_detected"])
        setting = _fmt(item["shock_mult"], 1) + (" (baseline)" if item is baseline else "")
        rows.append([
            setting,
            f"{detected} / {total}",
            str(item["false_alarm_episodes"]),
        ])
    if len(auroc_values) == 1 and len(ap_values) == 1:
        score_note = f"AUROC is {_fmt(next(iter(auroc_values)))} and AP is {_fmt(next(iter(ap_values)))}; neither varies with shock multiplier."
    else:
        score_note = f"AUROC ranges from {_fmt(min(auroc_values))} to {_fmt(max(auroc_values))}; AP ranges from {_fmt(min(ap_values))} to {_fmt(max(ap_values))}."
    tables.append(_table(
        "SKAB shock-multiplier sweep",
        "results/skab_shock_sweep.json, results/skab_validation.json",
        "mixed",
        ["Shock multiplier", "Events detected / total", "False alarm episodes"],
        rows,
        f"All {len(rows)} shock settings are shown; the baseline detected {int(baseline['events_detected'])} of {total}, while multiplier {_fmt(best['shock_mult'], 1)} detected {int(best['events_detected'])}. {score_note} Seed count is not reported in this file.",
    ))

    data = _load(repo, "skab_teacher_time_gate.json")
    total = data["dataset"]["fault_experiments"]
    rows = []
    for name, item in data["results"].items():
        rows.append([
            name,
            f"{item['events_detected']} / {total}",
            str(item["calibration_false_alarm_episodes"]),
            _fmt(item["window_auroc"]),
            _fmt(item["window_average_precision"]),
        ])
    tables.append(_table(
        "SKAB teacher-transfer time gate",
        "results/skab_teacher_time_gate.json",
        "negative",
        ["Model", *columns[1:]],
        rows,
        f"All {len(rows)} compact-model configurations detected no fault experiments out of {total}; false alarm counts are from healthy calibration. Seed count is not reported in this file.",
    ))

    data = _load(repo, "skab_sensor_fusion_gate.json")
    total = data["dataset"]["fault_experiments"]
    rows = []
    for name, item in data["results"].items():
        rows.append([
            name,
            f"{item['events_detected']} / {total}",
            str(item["calibration_false_alarm_episodes"]),
            _fmt(item["window_auroc"]),
            _fmt(item["window_average_precision"]),
        ])
    tables.append(_table(
        "SKAB sensor-fusion representation gate",
        "results/skab_sensor_fusion_gate.json",
        "negative",
        ["Representation", *columns[1:]],
        rows,
        f"All {len(rows)} sensor-fusion representations detected no fault experiments out of {total}; false alarm counts are from healthy calibration. Seed count is not reported in this file.",
    ))
    return tables
