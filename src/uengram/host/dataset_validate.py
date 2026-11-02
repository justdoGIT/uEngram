"""Validation and leakage-safe temporal splits for exported host datasets."""
from __future__ import annotations

import gzip
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from typing import Iterable

from . import schema
from .dataset import DatasetError, _normalise_incident


VALID_SPLITS = ("commission", "calibration", "test", "holdout")
VALID_PHASES = ("baseline", "pre_incident", "incident", "recovery")


def _rows(path: str) -> Iterable[tuple[int, dict]]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise DatasetError(f"invalid JSON at {path}:{line_no}") from e
            if not isinstance(row, dict):
                raise DatasetError(f"row is not an object at {path}:{line_no}")
            yield line_no, row


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _read_manifest(directory: str) -> tuple[dict | None, list[str]]:
    path = os.path.join(directory, "manifest.json")
    if not os.path.isfile(path):
        return None, ["missing manifest.json"]
    try:
        with open(path, encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return None, [f"cannot read manifest.json: {e}"]
    if not isinstance(manifest, dict):
        return None, ["manifest.json is not an object"]
    return manifest, []


def _load_incidents(directory: str, manifest: dict, errors: list[str]) -> dict[str, dict]:
    path = os.path.join(directory, str(manifest.get("labels", {}).get("path", "incidents.jsonl")))
    incidents: dict[str, dict] = {}
    if not os.path.isfile(path):
        errors.append(f"missing label file: {os.path.relpath(path, directory)}")
        return incidents
    for line_no, raw in _rows(path):
        try:
            incident = _normalise_incident(raw)
        except DatasetError as e:
            errors.append(f"incident line {line_no}: {e}")
            continue
        ident = str(incident["id"])
        if ident in incidents:
            errors.append(f"duplicate incident id: {ident}")
        incidents[ident] = incident
    return incidents


def _check_ticks(directory: str, node: dict, errors: list[str], warnings: list[str]) -> dict:
    path_value = node.get("tick_path")
    if not path_value:
        warnings.append(f"{node.get('name')}: raw ticks not included")
        return {"rows": 0, "kernels": [], "gaps": None}
    path = os.path.join(directory, path_value)
    if not os.path.isfile(path):
        errors.append(f"{node.get('name')}: missing tick file {path_value}")
        return {"rows": 0, "kernels": [], "gaps": None}
    count, kernels, previous, gaps, median_steps = 0, set(), None, 0, []
    for line_no, row in _rows(path):
        if not _finite(row.get("t")):
            errors.append(f"{node.get('name')}: invalid tick time at line {line_no}")
            continue
        t = float(row["t"])
        if previous is not None:
            step = t - previous
            if step <= 0:
                errors.append(f"{node.get('name')}: non-increasing tick time at line {line_no}")
            elif step > 0:
                median_steps.append(step)
        previous = t
        count += 1
        if row.get("kernel"):
            kernels.add(str(row["kernel"]))
        for field in schema.TICK_FIELDS:
            if field not in row:
                errors.append(f"{node.get('name')}: missing tick field {field} at line {line_no}")
            elif row[field] is not None and not _finite(row[field]):
                errors.append(f"{node.get('name')}: nonnumeric {field} at line {line_no}")
    median = sorted(median_steps)[len(median_steps) // 2] if median_steps else None
    if median:
        gaps = sum(step > 1.5 * median for step in median_steps)
    if node.get("ticks") is not None and count != node["ticks"]:
        errors.append(f"{node.get('name')}: tick count {count} != manifest {node['ticks']}")
    expected = set(node.get("quality", {}).get("kernel_releases", []))
    if expected and kernels != expected:
        errors.append(f"{node.get('name')}: kernel set differs from manifest")
    model_kernel = node.get("model_kernel")
    if model_kernel and kernels and kernels != {model_kernel}:
        errors.append(f"{node.get('name')}: data kernels {sorted(kernels)} != model {model_kernel}")
    return {"rows": count, "kernels": sorted(kernels), "gaps": gaps}


def _check_windows(directory: str, node: dict, incidents: dict[str, dict],
                   errors: list[str]) -> dict:
    path_value = node.get("window_path")
    name = str(node.get("name", ""))
    if not path_value:
        errors.append(f"{name}: missing window_path")
        return {"rows": 0, "start": None, "end": None, "phases": {}}
    path = os.path.join(directory, path_value)
    if not os.path.isfile(path):
        errors.append(f"{name}: missing window file {path_value}")
        return {"rows": 0, "start": None, "end": None, "phases": {}}
    K = float(node.get("window_seconds", 0))
    count, start, end, previous = 0, None, None, None
    phases = Counter()
    for line_no, row in _rows(path):
        if row.get("node") != name:
            errors.append(f"{name}: wrong node in window line {line_no}")
        if not _finite(row.get("t0")) or not _finite(row.get("window_end")) or K <= 0:
            errors.append(f"{name}: invalid window bounds at line {line_no}")
            continue
        t0, t1 = float(row["t0"]), float(row["window_end"])
        if not math.isclose(t1, t0 + K, rel_tol=0, abs_tol=1e-6):
            errors.append(f"{name}: window length mismatch at line {line_no}")
        if previous is not None and t0 <= previous:
            errors.append(f"{name}: overlapping or unordered window at line {line_no}")
        previous, start, end = t0, start if start is not None else t0, t1
        labels = row.get("labels")
        if not isinstance(labels, dict) or labels.get("status") not in ("unlabeled", "weak", "controlled", "confirmed"):
            errors.append(f"{name}: invalid labels at line {line_no}")
        else:
            phase = labels.get("phase")
            if phase not in VALID_PHASES:
                errors.append(f"{name}: invalid label phase at line {line_no}")
            phases[phase] += 1
            for incident in labels.get("incidents", []):
                ident = str(incident.get("id"))
                if ident not in incidents:
                    errors.append(f"{name}: unknown incident {ident} at line {line_no}")
        if "uengram_prediction" not in row:
            errors.append(f"{name}: missing uengram_prediction at line {line_no}")
        count += 1
    if node.get("windows") is not None and count != node["windows"]:
        errors.append(f"{name}: window count {count} != manifest {node['windows']}")
    return {"rows": count, "start": start, "end": end, "phases": dict(phases)}


def validate_dataset(directory: str) -> dict:
    """Return a machine-readable report; errors make ``valid`` false."""
    directory = os.path.abspath(os.path.expanduser(directory))
    manifest, errors = _read_manifest(directory)
    warnings: list[str] = []
    if manifest is None:
        return {"valid": False, "directory": directory, "errors": errors, "warnings": warnings}
    if manifest.get("format") != "uengram-host-dataset-v1":
        errors.append("unsupported dataset format")
    if manifest.get("uengram_outputs_are_predictions") is not True:
        errors.append("manifest does not declare predictions as non-labels")
    incidents = _load_incidents(directory, manifest, errors)
    if not incidents:
        warnings.append("dataset has no external incident labels; detection metrics are not valid")
    elif not any(bool(item.get("ground_truth_eligible")) for item in incidents.values()):
        warnings.append("dataset labels are not ground-truth eligible; use them only for controlled evaluation")
    nodes = manifest.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        errors.append("manifest has no nodes")
        nodes = []
    reports, names = [], set()
    for node in nodes:
        if not isinstance(node, dict) or not node.get("name"):
            errors.append("node entry lacks a name")
            continue
        name = str(node["name"])
        if name in names:
            errors.append(f"duplicate node: {name}")
        names.add(name)
        ticks = _check_ticks(directory, node, errors, warnings)
        windows = _check_windows(directory, node, incidents, errors)
        reports.append({"name": name, "ticks": ticks, "windows": windows})
    return {"valid": not errors, "directory": directory, "format": manifest.get("format"),
            "nodes": reports, "incidents": len(incidents), "errors": errors, "warnings": warnings}


def _assign(t0: float, t1: float, boundaries: list[float], gap: float) -> str:
    if t1 <= boundaries[0] - gap:
        return VALID_SPLITS[0]
    if t0 >= boundaries[0] + gap and t1 <= boundaries[1] - gap:
        return VALID_SPLITS[1]
    if t0 >= boundaries[1] + gap and t1 <= boundaries[2] - gap:
        return VALID_SPLITS[2]
    if t0 >= boundaries[2] + gap:
        return VALID_SPLITS[3]
    return "excluded_boundary"


def split_dataset(directory: str, out: str | None = None, fractions=(0.5, 0.15, 0.2, 0.15),
                  gap_seconds: float = 64.0) -> dict:
    """Create chronological split metadata without copying telemetry files."""
    if len(fractions) != 4 or any(float(x) <= 0 for x in fractions) or not math.isclose(sum(fractions), 1.0):
        raise DatasetError("split fractions must be four positive values summing to 1")
    if gap_seconds < 0 or not math.isfinite(gap_seconds):
        raise DatasetError("gap_seconds must be finite and non-negative")
    report = validate_dataset(directory)
    if not report["valid"]:
        raise DatasetError("dataset validation failed: " + "; ".join(report["errors"][:3]))
    with open(os.path.join(directory, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    spans = [(n["windows"]["start"], n["windows"]["end"]) for n in report["nodes"]]
    lo, hi = max(x[0] for x in spans), min(x[1] for x in spans)
    if not math.isfinite(lo) or hi <= lo:
        raise DatasetError("nodes have no common time range")
    span = hi - lo
    boundaries = [lo + span * sum(fractions[:i]) for i in range(1, 4)]
    assignments, incident_splits = {}, defaultdict(set)
    for node in report["nodes"]:
        name = node["name"]
        node_path = next(n["window_path"] for n in manifest["nodes"] if n["name"] == name)
        rows = []
        for index, row in _rows(os.path.join(directory, node_path)):
            t0, t1 = float(row["t0"]), float(row["window_end"])
            split = "excluded_outside_common_range" if t0 < lo or t1 > hi else _assign(t0, t1, boundaries, gap_seconds)
            refs = [str(x["id"]) for x in row["labels"].get("incidents", [])]
            rows.append({"row": index, "t0": t0, "window_end": t1, "split": split,
                         "phase": row["labels"].get("phase"), "incident_ids": refs})
            if split in VALID_SPLITS:
                for ident in refs:
                    incident_splits[ident].add(split)
        assignments[name] = rows
    conflicts = sorted(ident for ident, splits in incident_splits.items() if len(splits) > 1)
    for rows in assignments.values():
        for row in rows:
            if row["split"] in VALID_SPLITS and any(ident in conflicts for ident in row["incident_ids"]):
                row["split"] = "excluded_incident_boundary"
    counts = {name: dict(Counter(row["split"] for row in rows)) for name, rows in assignments.items()}
    result = {"format": "uengram-host-splits-v1", "dataset": os.path.abspath(directory),
              "fractions": dict(zip(VALID_SPLITS, map(float, fractions))), "gap_seconds": gap_seconds,
              "common_range": {"start": lo, "end": hi}, "boundaries": boundaries,
              "incident_boundary_conflicts": conflicts, "counts": counts, "assignments": assignments}
    path = os.path.abspath(os.path.expanduser(out or os.path.join(directory, "splits.json")))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".splits-", suffix=".tmp", dir=os.path.dirname(path))
    os.close(fd)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    result["path"] = path
    return result
