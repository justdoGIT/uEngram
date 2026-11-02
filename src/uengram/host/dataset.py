"""Export replayable host and cluster datasets with explicit label provenance.

The exporter combines the recorded telemetry with the deployed uEngram replay,
without turning uEngram alarms into ground truth.  A model can discover a
novel pattern; only an operator, maintenance record, or controlled experiment
can label what the pattern means.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import platform
import tempfile
from datetime import datetime, timezone
from typing import Iterable

import numpy as np

from . import schema
from .collector import spool_files
from .features import read_columns
from .node import NodeMemory
from .validate import replay


class DatasetError(ValueError):
    """Raised when an export would be ambiguous or unsafe."""


def _json_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return [_json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def _safe_name(name: str) -> str:
    if not name or name in (".", "..") or os.path.basename(name) != name:
        raise DatasetError(f"unsafe node name: {name!r}")
    return name


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write_gzip_jsonl(path: str, rows: Iterable[dict]) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".dataset-", suffix=".tmp", dir=os.path.dirname(path))
    os.close(fd)
    count = 0
    try:
        with gzip.open(tmp, "wt", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(_json_value(row), separators=(",", ":")) + "\n")
                count += 1
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    return count


def _write_jsonl(path: str, rows: Iterable[dict]) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".dataset-", suffix=".tmp", dir=os.path.dirname(path))
    os.close(fd)
    count = 0
    try:
        with open(tmp, "w", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(_json_value(row), separators=(",", ":")) + "\n")
                count += 1
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    return count


def _normalise_incident(raw: dict, default_source: str = "external") -> dict:
    if not isinstance(raw, dict):
        raise DatasetError("each incident must be a JSON object")
    if not raw.get("node"):
        raise DatasetError("incident needs a non-empty node")
    try:
        start, end = float(raw["start"]), float(raw["end"])
    except (KeyError, TypeError, ValueError) as e:
        raise DatasetError("incident start/end must be numeric") from e
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise DatasetError("incident end must be finite and later than start")
    for key in ("operator_confirmed", "ground_truth_eligible"):
        if key in raw and not isinstance(raw[key], bool):
            raise DatasetError(f"{key} must be boolean")
    source = str(raw.get("source") or default_source)
    confirmed = raw.get("operator_confirmed", False)
    eligible = raw.get("ground_truth_eligible", confirmed and source != "controlled_campaign")
    out = dict(raw)
    out.update(start=start, end=end, node=str(raw["node"]), source=source,
               operator_confirmed=confirmed, ground_truth_eligible=eligible)
    out.setdefault("id", f"{out['node']}:{start:g}:{end:g}")
    out.setdefault("type", "unspecified")
    return _json_value(out)


def load_incidents(path: str | None, default_source: str = "external") -> list[dict]:
    """Load and validate external JSONL incident labels.

    Labels are deliberately separate from model predictions.  A malformed
    interval fails the export instead of producing a silently misaligned label.
    """
    if not path:
        return []
    incidents = []
    with open(os.path.expanduser(path), encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise DatasetError(f"invalid incident JSON at line {line_no}") from e
            try:
                incidents.append(_normalise_incident(raw, default_source))
            except DatasetError as e:
                raise DatasetError(f"incident line {line_no}: {e}") from e
    return incidents


def _incident_matches(incident: dict, node: str, t0: float, t1: float) -> tuple[bool, str | None]:
    if incident.get("node") == node:
        role = "target"
    elif incident.get("peer") == node:
        role = "peer"
    else:
        return False, None
    return t0 < float(incident["end"]) and t1 > float(incident["start"]), role


def _labels(incidents: list[dict], node: str, t0: float, t1: float, horizon: float = 64.0) -> dict:
    matched = []
    for incident in incidents:
        overlaps, role = _incident_matches(incident, node, t0, t1)
        if role is None:
            continue
        phase = "incident" if overlaps else (
            "recovery" if t0 >= float(incident["end"]) and t0 - float(incident["end"]) <= horizon
            else "pre_incident" if t1 <= float(incident["start"]) and float(incident["start"]) - t1 <= horizon
            else None)
        if phase is not None:
            matched.append({"id": incident["id"], "type": incident.get("type"), "role": role,
                            "source": incident["source"], "phase": phase,
                            "operator_confirmed": incident["operator_confirmed"],
                            "ground_truth_eligible": incident["ground_truth_eligible"]})
    if not matched:
        return {"status": "unlabeled", "phase": "baseline", "incidents": []}
    statuses = {"controlled_campaign" if x["source"] == "controlled_campaign" else
                "confirmed" if x["operator_confirmed"] else "weak" for x in matched}
    status = "confirmed" if "confirmed" in statuses else "controlled" if "controlled_campaign" in statuses else "weak"
    phase = "incident" if any(x["phase"] == "incident" for x in matched) else (
        "recovery" if any(x["phase"] == "recovery" for x in matched) else "pre_incident")
    return {"status": status, "phase": phase, "incidents": matched}


def _tick_rows(cols) -> Iterable[dict]:
    fields = tuple(schema.TICK_FIELDS)
    for i, t in enumerate(cols["t"]):
        row = {"t": t, "kernel": str(cols["kernel"][i]) if cols["kernel"][i] else None}
        row.update({field: cols[field][i] for field in fields})
        yield row


def _quality(cols) -> dict:
    n = len(cols["t"])
    missing = {f: int(np.count_nonzero(~np.isfinite(cols[f]))) for f in schema.TICK_FIELDS}
    t = cols["t"]
    dt = np.diff(t)
    positive = dt[np.isfinite(dt) & (dt > 0)]
    median = float(np.median(positive)) if len(positive) else None
    gaps = int(np.count_nonzero(dt > 1.5 * median)) if median else None
    return {"ticks": n, "missing_values": missing, "gap_count_estimate": gaps,
            "median_timestamp_step_s": median, "kernel_releases": sorted(set(cols["kernel"]) - {""})}


def _prediction(rec: dict) -> dict:
    keys = ("alarm", "alarm_latent", "alarm_event", "saturation", "score", "persistent_score",
            "health", "persistent_health", "p_latent", "kind", "action", "item_id",
            "state_alarmed_at", "modality_d2", "health_modality_d2", "top_features")
    return {key: rec.get(key) for key in keys if key in rec}


def _window_rows(node: NodeMemory, cols, incidents: list[dict], name: str) -> Iterable[dict]:
    for rec in replay(node, cols):
        t0 = float(rec["t0"])
        labels = _labels(incidents, name, t0, t0 + node.K, horizon=2 * node.K)
        row = dict(rec)
        row.update(node=name, window_end=t0 + node.K, labels=labels,
                   uengram_prediction=_prediction(rec))
        yield _json_value(row)


def _node_export(out: str, name: str, tick_paths: list[str], model_path: str,
                 incidents: list[dict], include_ticks: bool) -> dict:
    name = _safe_name(name)
    paths = [os.path.expanduser(p) for p in tick_paths]
    if not paths:
        raise DatasetError(f"no tick files for node {name}")
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise DatasetError(f"missing tick source for {name}: {missing[0]}")
    if not os.path.isfile(model_path):
        raise DatasetError(f"missing model for {name}: {model_path}")
    meta, cols = read_columns(paths)
    node = NodeMemory.load(model_path)
    base = os.path.join(out, "nodes", name)
    tick_path = os.path.join(base, "ticks.jsonl.gz")
    window_path = os.path.join(base, "windows.jsonl.gz")
    tick_count = _write_gzip_jsonl(tick_path, _tick_rows(cols)) if include_ticks else 0
    window_count = _write_gzip_jsonl(window_path, _window_rows(node, cols, incidents, name))
    quality = _quality(cols)
    return {
        "name": name, "window_seconds": node.K, "ticks": quality["ticks"],
        "windows": window_count, "tick_path": os.path.relpath(tick_path, out) if include_ticks else None,
        "window_path": os.path.relpath(window_path, out), "quality": quality,
        "source_files": [os.path.basename(p) for p in paths],
        "source_meta": _json_value(meta), "model_file": os.path.basename(model_path),
        "model_sha256": _sha256(model_path), "model_kernel": node.meta.get("kernel"),
        "model_commission": _json_value(node.commission_report),
        "tick_rows_written": tick_count if include_ticks else None,
    }


def _write_manifest(out: str, manifest: dict) -> dict:
    path = os.path.join(out, "manifest.json")
    fd, tmp = tempfile.mkstemp(prefix=".manifest-", suffix=".tmp", dir=out)
    os.close(fd)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_json_value(manifest), f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    return manifest


def _export(out: str, specs: list[tuple[str, list[str], str]], incidents: list[dict],
            include_ticks: bool, kind: str, campaign: dict | None = None) -> dict:
    out = os.path.abspath(os.path.expanduser(out))
    os.makedirs(out, exist_ok=True)
    nodes = [_node_export(out, name, paths, model, incidents, include_ticks)
             for name, paths, model in specs]
    incident_count = _write_jsonl(os.path.join(out, "incidents.jsonl"), incidents)
    manifest = {
        "format": "uengram-host-dataset-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "producer": {"package": "uengram", "python": platform.python_version()},
        "dataset_kind": kind, "nodes": nodes,
        "telemetry": {"raw_ticks": include_ticks, "derived_windows": True,
                       "tick_schema": list(schema.TICK_FIELDS),
                       "prediction_fields_are_not_labels": True},
        "labels": {"path": "incidents.jsonl", "records": incident_count,
                   "unlabeled_windows_have_status": "unlabeled",
                   "ground_truth_requires_external_provenance": True},
        "uengram_outputs_are_predictions": True,
        "retention": {"source_spool_retention_days": 14,
                      "source_retention_is_deployment_default_not_proof": True,
                      "export_files_are_not_auto_deleted": True},
    }
    if campaign is not None:
        manifest["campaign"] = {key: campaign.get(key) for key in
                                 ("start", "clean_split", "commission", "end", "alpha", "nodes")
                                 if key in campaign}
    return _write_manifest(out, manifest)


def export_single(out: str, node: str, tick_paths: list[str], model_path: str,
                  incidents: Iterable[dict] = (), include_ticks: bool = True) -> dict:
    """Export one host's raw ticks, replayed windows, and optional labels."""
    labels = [_normalise_incident(x) for x in incidents]
    return _export(out, [(node, tick_paths, os.path.expanduser(model_path))], labels,
                   include_ticks, "single_host")


def export_campaign(campaign_dir: str, out: str, incidents_path: str | None = None,
                    include_ticks: bool = True) -> dict:
    from .validate import load_faults

    campaign_dir = os.path.abspath(os.path.expanduser(campaign_dir))
    with open(os.path.join(campaign_dir, "campaign.json"), encoding="utf-8") as f:
        campaign = json.load(f)
    incidents = []
    for fault in load_faults(campaign_dir):
        incident = dict(fault, source="controlled_campaign", operator_confirmed=False,
                        ground_truth_eligible=False)
        incidents.append(_normalise_incident(incident, "controlled_campaign"))
    incidents.extend(load_incidents(incidents_path))
    specs = []
    for name in campaign.get("nodes", {}):
        spool = os.path.join(campaign_dir, "nodes", name, "spool")
        paths = spool_files(spool, name)
        model = os.path.join(campaign_dir, "models", f"model-{name}.pkl")
        if not paths or not os.path.isfile(model):
            raise DatasetError(f"campaign node {name} lacks spool or commissioned model")
        specs.append((name, paths, model))
    return _export(out, specs, incidents, include_ticks, "cluster_campaign", campaign)
