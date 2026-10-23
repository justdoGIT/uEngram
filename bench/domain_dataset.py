"""Development-only weak report-association examples from frozen Laya traces."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from bench import advisory_eval as ev
from bench import laya_advisory as la
from bench.streams import load_stream

DEFAULT_FREEZE = Path("results/v3_normal_freeze.json")
DEFAULT_OUT_DIR = Path.home() / ".local/state/uengram/domain_dataset"
DEFAULT_MANIFEST = Path("results/v4_domain_dataset_manifest.json")
QUESTIONS = {
    "report_association": {
        "type": "choice",
        "instructions": (
            "Is this assessment window associated with a published fault/maintenance report interval? "
            "This is only a weak report-association proxy, not a verified physical-fault label or "
            "operator-review label. A detector in-reference window is not evidence of verified "
            "physical health."
        ),
        "criteria": {
            "reported_fault": "The entire assessment input window is contained in a published report interval.",
            "outside_report_buffer": "The assessment window and every causal context dependency are outside all report exclusion buffers.",
        },
    }
}
CAVEATS = [
    "Report intervals are coarse maintenance/report intervals, not verified pointwise physical faults or operator-review labels.",
    "All three MetroPT recordings were consumed earlier; asset independence is unknown and no split by file/window is valid evidence of asset independence.",
    "Labels are development-only report-association proxies; they are not diagnostic truth, physical-health certification, or operator-review decisions.",
]
MISSING_PREREQUISITES = [
    "Independent recordings/assets with verified asset identities and no overlap with consumed traces.",
    "Independently adjudicated pointwise physical-fault labels and operator-review labels with reliable timestamps.",
    "A preregistered independent validation/test design and untouched evaluation data.",
]


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _time_array(values):
    a = np.asarray(values)
    return a.astype("datetime64[s]")


def _interval_labels(stream, positions):
    """Report membership in the same test-relative coordinates as the frozen trace."""
    starts = _time_array(stream.start[positions])
    ends = _time_array(stream.end[positions])
    overlaps, contained = [], []
    for label_start, label_end in stream.events:
        a, b = np.datetime64(label_start, "s"), np.datetime64(label_end, "s")
        overlaps.append((starts <= b) & (ends >= a))
        contained.append((starts >= a) & (ends <= b))
    if not overlaps:
        return (np.zeros(len(starts), int), np.zeros(len(starts), int),
                np.full(len(starts), -1, dtype=int))
    overlap_n = np.sum(overlaps, axis=0)
    contained_n = np.sum(contained, axis=0)
    event_number = np.full(len(starts), -1, dtype=int)
    for j, mask in enumerate(contained):
        event_number[mask & (contained_n == 1)] = j
    return overlap_n, contained_n, event_number


def _deps(z, index, references):
    refs = references[int(z["state"][index])]
    stop = int(np.searchsorted(refs, index))
    refs = refs[max(0, stop - la.REFERENCE_WINDOWS):stop]
    return {"current": int(index), "previous": int(index - 1) if index > 0 else None,
            "reference": [int(x) for x in refs]}


def _time_text(value):
    return str(np.datetime_as_string(np.datetime64(value, "s"), unit="s"))


def _empty_counts():
    return {"scheduled": 0, "exported_positive": 0, "exported_negative": 0,
            "excluded": {}, "deduplicated": 0, "conflicting_input_excluded": 0}


def _exclude(counts, reason):
    counts["excluded"][reason] = counts["excluded"].get(reason, 0) + 1


def export_dataset(freeze_path=DEFAULT_FREEZE, out_dir=DEFAULT_OUT_DIR,
                   manifest_path=DEFAULT_MANIFEST, *, freeze=None, stream_loader=load_stream,
                   trace_loader=ev.load_trace):
    """Export scheduled eligible assessments; optional injected inputs support semantic tests."""
    freeze_path, out_dir, manifest_path = Path(freeze_path), Path(out_dir), Path(manifest_path)
    freeze = json.loads(freeze_path.read_text()) if freeze is None else freeze
    if freeze.get("schema") != 1 or not freeze.get("selected"):
        raise ValueError("unsupported or empty frozen normal schema")
    freeze_hash = _sha_bytes(freeze_path.read_bytes()) if freeze_path.is_file() else _sha_bytes(_canonical(freeze))
    groups, per_stream = {}, {}
    for name, record in freeze["selected"].items():
        z = trace_loader(record)
        stream = stream_loader(name)
        positions = np.flatnonzero(stream.test)
        zstart, zend = _time_array(z["start"]), _time_array(z["end"])
        if len(positions) != len(zstart) or not np.array_equal(_time_array(stream.start[positions]), zstart) or not np.array_equal(_time_array(stream.end[positions]), zend):
            raise ValueError(f"frozen trace does not align with test windows for {name}")
        if not np.array_equal(z["test_input_index"], positions):
            raise ValueError(f"frozen input indices do not align for {name}")
        due = ev.assessment_schedule(z["decision_state"], z["end"])
        states = la.due_states(z, due)
        overlap, contained, contained_event = _interval_labels(stream, positions)
        eligible = ev.metric_valid(stream, z, positions)
        references = la.StateBuilder(z).reference
        negative_window = stream.clean_negative()[positions]
        event_intervals = [(str(a), str(b)) for a, b in stream.events]
        counts = _empty_counts()
        trace_meta = record["trace"]
        for index, state_text in states.items():
            counts["scheduled"] += 1
            if not eligible[index]:
                _exclude(counts, "ineligible_assessment")
                continue
            deps = _deps(z, index, references)
            # Put all time-aligned dependencies into the same test-position coordinate system.
            dep_indices = [deps["current"]]
            if deps["previous"] is not None:
                dep_indices.append(deps["previous"])
            dep_indices.extend(deps["reference"])
            if overlap[index] > 1 or contained[index] > 1:
                _exclude(counts, "ambiguous_report_interval")
                continue
            if overlap[index] and not contained[index]:
                _exclude(counts, "straddling_report_interval")
                continue
            if contained[index] == 1:
                if contained_event[index] < 0:
                    _exclude(counts, "ambiguous_report_interval")
                    continue
                label = "reported_fault"
                event_label = {"basis": "full_window_contained_in_report_interval",
                               "event_index": int(contained_event[index]),
                               "interval": event_intervals[int(contained_event[index])]}
            else:
                # All direct and StateBuilder recent-normal dependencies must be clean.
                if not all(bool(negative_window[j]) for j in dep_indices):
                    _exclude(counts, "context_in_report_exclusion_buffer")
                    continue
                label = "outside_report_buffer"
                event_label = {"basis": "current_previous_and_reference_outside_report_buffers",
                               "event_index": None, "interval": None}
            row = {"state": state_text, "questions": QUESTIONS, "expected": {"report_association": label}}
            input_id = _sha_bytes(_canonical({"state": row["state"], "questions": row["questions"]}))
            provenance = {"id": input_id, "development_only": True, "stream": name,
                          "record_id": record.get("stream", name),
                          "asset_id": None, "asset_group": "shared_unverified_metropt_assets",
                          "trace_path": trace_meta["path"], "trace_sha256": trace_meta["sha256"],
                          "freeze_trace_sha256": trace_meta["sha256"],
                          "index": int(index), "start": _time_text(zstart[index]), "end": _time_text(zend[index]),
                          "input_window_index": int(positions[index]),
                          "dependency_coordinate_system": "test_relative_trace_window_indices",
                          "event_label_basis": event_label, "dependency_indices": deps,
                          "_target": label}
            item = groups.setdefault(input_id, {"row": row, "provenance": []})
            item["provenance"].append(provenance)
        per_stream[name] = {"trace_sha256": trace_meta["sha256"], "counts": counts,
                            "report_intervals": event_intervals,
                            "exclusion_buffer_hours": {"before": stream.before_h, "after": stream.after_h}}
    # Conflicting targets for identical inputs are dropped as a group, including all provenance.
    output_rows, provenance_rows = [], []
    global_excluded = {"conflicting_identical_inputs": 0}
    for input_id in sorted(groups):
        item = groups[input_id]
        origins = sorted(item["provenance"], key=lambda p: (p["stream"], p["index"]))
        if any(p["id"] != input_id for p in origins):
            raise AssertionError("provenance key mismatch")
        labels = {p["_target"] for p in origins}
        if len(labels) > 1:
            global_excluded["conflicting_identical_inputs"] += 1
            for provenance in origins:
                per_stream[provenance["stream"]]["counts"]["conflicting_input_excluded"] += 1
            continue
        label = labels.pop()
        for provenance in origins:
            counts = per_stream[provenance["stream"]]["counts"]
            counts["exported_positive" if label == "reported_fault" else "exported_negative"] += 1
        # One inference row represents all matching source assessments.
        for provenance in origins[1:]:
            per_stream[provenance["stream"]]["counts"]["deduplicated"] += 1
        output_rows.append(item["row"])
        for provenance in origins:
            del provenance["_target"]
            provenance_rows.append(provenance)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path, provenance_path = out_dir / "weak_development.jsonl", out_dir / "provenance.jsonl"
    dataset_bytes = b"".join(_canonical(row) + b"\n" for row in output_rows)
    provenance_bytes = b"".join(_canonical(row) + b"\n" for row in sorted(provenance_rows, key=lambda p: (p["id"], p["stream"], p["index"])))
    dataset_path.write_bytes(dataset_bytes)
    provenance_path.write_bytes(provenance_bytes)
    manifest = {"schema": 1, "dataset": "development_only_weak_report_association",
                "freeze_path": str(freeze_path), "freeze_sha256": freeze_hash,
                "code_sha256": {p: ev.sha256_file(p) for p in (
                    "bench/domain_dataset.py", "bench/laya_advisory.py", "bench/advisory_eval.py",
                    "bench/streams.py", "src/uengram/metropt.py", "src/uengram/metropt2.py")},
                "labels": {"reported_fault": "full input window contained in a published report interval", "outside_report_buffer": "current, previous, and all StateBuilder reference windows outside existing report margins"},
                "caveats": CAVEATS, "per_stream": per_stream, "excluded": global_excluded,
                "rows": len(output_rows), "provenance_rows": len(provenance_rows),
                "unique_class_counts": {
                    label: sum(row["expected"]["report_association"] == label for row in output_rows)
                    for label in QUESTIONS["report_association"]["criteria"]},
                "count_scope": "per-stream exported counts include all deduplicated source origins; rows and unique_class_counts count distinct inference inputs",
                "input_resolution": "Existing decimated model-window summaries; not full native source-row snapshots.",
                "outputs": {str(dataset_path): _sha_bytes(dataset_bytes), str(provenance_path): _sha_bytes(provenance_bytes)},
                "independent_evaluation_eligible": False, "production_ready": False,
                "evaluation_status": "no_independent_evaluation",
                "missing_prerequisites": MISSING_PREREQUISITES}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    print(json.dumps(export_dataset(args.freeze, args.out_dir, args.manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
