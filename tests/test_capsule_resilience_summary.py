from __future__ import annotations

import json

import pytest

from bench.capsule_resilience_summary import summarize


def _artifact(label, crash="after_ingest", sha="a" * 64, capsule_ok=True, recovery_ns=1_000_000,
              reference=3, capsule_keys_ok=True):
    def cond(ok, keys_ok=False):
        return {"state_equivalent_to_uninterrupted": ok, "event_keys_match_reference": keys_ok,
                "missing_event_count": 0 if keys_ok else 1, "extra_event_count": 0 if keys_ok else 2,
                "vendor_event_count": 3, "ranked_history_count": 3, "recovery_ns": recovery_ns,
                "local_storage_bytes": 10}
    return {"protocol": "capsule-resilience-v3", "crash_point": crash,
            "input": {"stream": "smd", "sha256": sha, "window_count": 64},
            "reference": {"vendor_event_count": reference},
            "host": {"target_label": label, "hostname": label, "platform": "p", "machine": "x86_64",
                     "cpu_model": "test-cpu", "memory_mib": 4096,
                     "python": "3.12", "numpy": "2.5", "cpu_count": 4},
            "conditions": {"lite": cond(False), "sidecar": cond(False),
                           "capsule": cond(capsule_ok, capsule_keys_ok)}}


def _write(root, label, name, doc):
    (root / label).mkdir(parents=True, exist_ok=True)
    (root / label / name).write_text(json.dumps(doc))


def test_median_timing_and_equivalence_are_aggregated_per_host_and_boundary(tmp_path):
    for n, ns in enumerate((1_000_000, 3_000_000, 2_000_000), 1):
        _write(tmp_path, "hp", f"after_ingest-{n}.json", _artifact("hp", recovery_ns=ns))

    cell = summarize(tmp_path)["hosts"]["hp"]["crashes"]["after_ingest"]["capsule"]

    assert cell["recovery_ms"] == {"median": 2.0, "min": 1.0, "max": 3.0}
    assert cell["state_equivalent_runs"] == cell["runs"] == 3


def test_a_failing_capsule_run_is_counted_not_dropped(tmp_path):
    _write(tmp_path, "hp", "after_ingest-1.json", _artifact("hp"))
    _write(tmp_path, "hp", "after_ingest-2.json", _artifact("hp", capsule_ok=False))

    summary = summarize(tmp_path)

    assert summary["hosts"]["hp"]["crashes"]["after_ingest"]["capsule"]["state_equivalent_runs"] == 1
    assert summary["checks"]["capsule_state_equivalent"] is False


def test_mixed_input_streams_are_rejected(tmp_path):
    _write(tmp_path, "hp", "after_ingest-1.json", _artifact("hp", sha="a" * 64))
    _write(tmp_path, "asus", "after_ingest-1.json", _artifact("asus", sha="b" * 64))

    with pytest.raises(ValueError, match="one input stream"):
        summarize(tmp_path)


def test_label_must_match_its_directory(tmp_path):
    _write(tmp_path, "hp", "after_ingest-1.json", _artifact("asus"))

    with pytest.raises(ValueError, match="target label"):
        summarize(tmp_path)


def test_event_key_mismatch_is_reported_separately_from_state_equivalence(tmp_path):
    _write(tmp_path, "hp", "after_ingest-1.json", _artifact("hp", capsule_keys_ok=False))

    summary = summarize(tmp_path)
    capsule = summary["hosts"]["hp"]["crashes"]["after_ingest"]["capsule"]

    assert summary["checks"]["capsule_state_equivalent"] is True
    assert summary["checks"]["capsule_event_keys_match_reference"] is False
    assert capsule["missing_event_counts"] == [1] and capsule["extra_event_counts"] == [2]


def test_disagreeing_uninterrupted_references_are_rejected(tmp_path):
    _write(tmp_path, "hp", "after_ingest-1.json", _artifact("hp", reference=3))
    _write(tmp_path, "asus", "after_ingest-1.json", _artifact("asus", reference=4))

    with pytest.raises(ValueError, match="uninterrupted reference"):
        summarize(tmp_path)
