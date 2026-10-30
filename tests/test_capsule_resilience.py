from __future__ import annotations

import pytest

from bench.capsule_resilience import run


@pytest.mark.parametrize("crash", ("after_wal", "after_ingest", "after_checkpoint"))
def test_capsule_recovers_state_after_each_predeclared_crash_boundary(crash):
    result = run(crash)
    conditions = result["conditions"]

    assert result["protocol"] == "capsule-resilience-v2"
    assert conditions["capsule"]["state_equivalent_to_uninterrupted"]
    assert not conditions["lite"]["state_equivalent_to_uninterrupted"]
    assert not conditions["sidecar"]["state_equivalent_to_uninterrupted"]
    assert conditions["capsule"]["vendor_event_count"] == 4
    assert conditions["capsule"]["ranked_history_count"] >= conditions["lite"]["ranked_history_count"]



def test_fixture_result_records_host_and_input_provenance():
    result = run("after_ingest", stream="fixture", window_count=6, target_label="ci-target")

    assert result["input"]["stream"] == "fixture"
    assert result["input"]["window_count"] == 6
    assert len(result["input"]["sha256"]) == 64
    assert result["host"]["target_label"] == "ci-target"
    assert result["host"]["python"]


def test_target_label_is_required_when_requested():
    with pytest.raises(ValueError, match="target label"):
        run("after_ingest", require_target_label=True)