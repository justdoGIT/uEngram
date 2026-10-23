import hashlib
import json

import numpy as np
import pytest

from bench.domain_dataset import export_dataset
from bench.streams import Stream

BASE = np.datetime64("2020-01-01T00:00:00", "s")


def _fixture(candidate=45, interval=None, n=80):
    starts = BASE + np.arange(n) * np.timedelta64(600, "s")
    ends = starts + np.timedelta64(600, "s")
    states = np.ones(n, dtype=int)
    decision = np.ones(n, dtype=int)
    decision[candidate] = 2
    names = ["load_share", "comp_switches", "motor_current_a", "oil_temp_c", "tp2_bar",
             "tp3_bar", "h1_bar", "dv_pressure_bar", "reservoirs_min_bar", "lps_share",
             "oil_level", "pressure_switch"]
    features = np.tile(np.arange(len(names), dtype=float) + 1.0, (n, 1))
    features[:, names.index("lps_share")] = 123.456
    features[:, names.index("oil_level")] = 777.25
    features[:, names.index("pressure_switch")] = 888.75
    z = {"decision_state": decision, "state": states,
         "start": starts.astype(np.int64), "end": ends.astype(np.int64),
         "test_input_index": np.arange(n),
         "novelty": np.ones(n), "novelty_threshold": np.full(n, 2.0),
         "health": np.ones(n), "health_threshold": np.full(n, 2.0),
         "features": features, "feature_names": np.asarray(names),
         "baseline_counts": json.dumps({"1": {"novelty_threshold_supported": True,
                                                  "monitorable_windows": 100}}),
         "config": json.dumps({"per_state": True, "k_novelty": 3, "k_health": 0})}
    events = () if interval is None else ((str(interval[0]), str(interval[1])),)
    stream = Stream("test", np.empty((n, 0)), starts, ends, np.zeros(n, int), states,
                    np.ones(n, bool), np.zeros(n, bool), np.zeros(n, bool), np.ones(n, bool),
                    events, 0, features, tuple(names))
    raw = b"trace"
    record = {"trace": {"path": "synthetic.npz", "bytes": len(raw),
                         "sha256": hashlib.sha256(raw).hexdigest()}}
    return z, stream, record


def _export(tmp_path, entries):
    selected, zs, streams = {}, {}, {}
    for name, z, stream, record in entries:
        selected[name] = record
        zs[name] = z
        streams[name] = stream
    freeze = {"schema": 1, "selected": selected}
    return export_dataset(tmp_path / "freeze.json", tmp_path / "out", tmp_path / "manifest.json",
                          freeze=freeze, trace_loader=lambda r: zs[next(k for k, v in selected.items() if v is r)],
                          stream_loader=lambda name: streams[name])


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_labels_straddling_and_negative_dependency_exclusion(tmp_path):
    z, stream, rec = _fixture()
    cur_start, cur_end = BASE + 45 * np.timedelta64(600, "s"), BASE + 46 * np.timedelta64(600, "s")
    stream.events = ((str(cur_start + np.timedelta64(300, "s")), str(cur_end)),)
    manifest = _export(tmp_path, [("metropt3", z, stream, rec)])
    assert manifest["per_stream"]["metropt3"]["counts"]["excluded"]["straddling_report_interval"] == 1
    assert _jsonl(tmp_path / "out/weak_development.jsonl") == []

    z, stream, rec = _fixture()
    prior_start = BASE + 44 * np.timedelta64(600, "s")
    stream.events = ((str(prior_start + np.timedelta64(1, "s")), str(prior_start + np.timedelta64(599, "s"))),)
    manifest = _export(tmp_path, [("metropt3", z, stream, rec)])
    counts = manifest["per_stream"]["metropt3"]["counts"]
    assert counts["excluded"]["context_in_report_exclusion_buffer"] == 1
    assert _jsonl(tmp_path / "out/weak_development.jsonl") == []

def test_recent_reference_context_is_also_exclusion_checked(tmp_path):
    z, stream, rec = _fixture(candidate=457, n=500)
    report_start = BASE + 160 * np.timedelta64(600, "s")
    stream.events = ((str(report_start), str(report_start + np.timedelta64(600, "s"))),)
    clean = stream.clean_negative()
    assert clean[456] and clean[457]
    manifest = _export(tmp_path, [("metropt3", z, stream, rec)])
    counts = manifest["per_stream"]["metropt3"]["counts"]
    assert counts["excluded"]["context_in_report_exclusion_buffer"] == 1
    assert _jsonl(tmp_path / "out/weak_development.jsonl") == []


def test_positive_full_containment_and_dependency_provenance_not_in_state(tmp_path):
    z, stream, rec = _fixture()
    start = BASE + 45 * np.timedelta64(600, "s")
    end = start + np.timedelta64(600, "s")
    stream.events = ((str(start), str(end)),)
    manifest = _export(tmp_path, [("metropt3", z, stream, rec)])
    rows = _jsonl(tmp_path / "out/weak_development.jsonl")
    provenance = _jsonl(tmp_path / "out/provenance.jsonl")
    assert len(rows) == len(provenance) == 1
    assert rows[0]["expected"] == {"report_association": "reported_fault"}
    assert set(rows[0]) == {"state", "questions", "expected"}
    state = rows[0]["state"]
    for hidden in ("123.456", "777.25", "888.75", str(start), "reported_fault", "metropt3"):
        assert hidden not in state
    z["features"][46] += 1000
    _export(tmp_path / "future", [("metropt3", z, stream, rec)])
    assert _jsonl(tmp_path / "future/out/weak_development.jsonl")[0]["state"] == state

    assert provenance[0]["dependency_indices"] == {"current": 45, "previous": 44,
                                                    "reference": list(range(9, 45))}
    assert provenance[0]["event_label_basis"]["event_index"] == 0
    assert provenance[0]["event_label_basis"]["interval"] == [str(start), str(end)]
    assert provenance[0]["asset_id"] is None
    assert manifest["independent_evaluation_eligible"] is False


def test_identical_inputs_deduplicate_and_conflicting_labels_drop_all(tmp_path):
    first = _fixture()
    second = _fixture()
    start = BASE + 45 * np.timedelta64(600, "s")
    end = start + np.timedelta64(600, "s")
    first[1].events = ((str(start), str(end)),)
    second[1].events = ((str(start), str(end)),)
    manifest = _export(tmp_path, [("metropt3", *first), ("metropt2_jan", *second)])
    rows, provenance = _jsonl(tmp_path / "out/weak_development.jsonl"), _jsonl(tmp_path / "out/provenance.jsonl")
    assert len(rows) == 1 and len(provenance) == 2
    assert all(p["asset_id"] is None for p in provenance)
    assert all(p["development_only"] for p in provenance)
    assert len({p["asset_group"] for p in provenance}) == 1
    assert sum(x["counts"]["deduplicated"] for x in manifest["per_stream"].values()) == 1

    second[1].events = ((str(BASE + np.timedelta64(10, "D")),
                         str(BASE + np.timedelta64(10, "D") + np.timedelta64(600, "s"))),)
    manifest = _export(tmp_path, [("metropt3", *first), ("metropt2_jan", *second)])
    assert _jsonl(tmp_path / "out/weak_development.jsonl") == []
    assert _jsonl(tmp_path / "out/provenance.jsonl") == []
    assert manifest["excluded"]["conflicting_identical_inputs"] == 1
    assert all(x["counts"]["conflicting_input_excluded"] == 1
               for x in manifest["per_stream"].values())


def test_export_rejects_tampered_frozen_trace(tmp_path):
    z, stream, record = _fixture()
    trace_path = tmp_path / "trace.npz"
    np.savez(trace_path, **z)
    raw = trace_path.read_bytes()
    record["trace"] = {"path": str(trace_path), "bytes": len(raw),
                       "sha256": hashlib.sha256(raw).hexdigest()}
    trace_path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError, match="integrity mismatch"):
        export_dataset(tmp_path / "freeze.json", tmp_path / "out", tmp_path / "manifest.json",
                       freeze={"schema": 1, "selected": {"metropt3": record}},
                       stream_loader=lambda _: stream)
    assert not (tmp_path / "out").exists()


def test_report_labels_use_test_window_indices_after_commissioning(tmp_path):
    z, stream, record = _fixture()
    stream.test[:40] = False
    start, end = stream.start[45], stream.end[45]
    stream.events = ((str(start), str(end)),)
    n = len(stream.start)
    for key, values in z.items():
        if isinstance(values, np.ndarray) and len(values) == n:
            z[key] = values[40:]
    manifest = _export(tmp_path, [("metropt3", z, stream, record)])
    rows = _jsonl(tmp_path / "out/weak_development.jsonl")
    provenance = _jsonl(tmp_path / "out/provenance.jsonl")
    assert rows[0]["expected"] == {"report_association": "reported_fault"}
    assert provenance[0]["index"] == 5
    assert provenance[0]["input_window_index"] == 45
    assert manifest["unique_class_counts"]["reported_fault"] == 1


@pytest.mark.parametrize("ineligible", ["unmonitorable", "unscored", "evaluation_mask"])
def test_ineligible_candidates_are_not_labelled_as_training_examples(tmp_path, ineligible):
    z, stream, record = _fixture()
    if ineligible == "unmonitorable":
        stream.monitorable[45] = False
    elif ineligible == "unscored":
        z["novelty"][45] = np.nan
    else:
        stream.metric_mask = np.ones(len(stream.start), bool)
        stream.metric_mask[45] = False
    manifest = _export(tmp_path, [("metropt3", z, stream, record)])
    assert manifest["per_stream"]["metropt3"]["counts"]["excluded"]["ineligible_assessment"] == 1
    assert _jsonl(tmp_path / "out/weak_development.jsonl") == []
