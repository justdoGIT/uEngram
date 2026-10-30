from __future__ import annotations

import pytest

from bench.capsule_resilience import run


@pytest.mark.parametrize("crash", ("after_wal", "after_ingest", "after_checkpoint"))
def test_capsule_recovers_state_after_each_predeclared_crash_boundary(crash):
    result = run(crash)
    conditions = result["conditions"]

    assert result["protocol"] == "capsule-resilience-v1"
    assert conditions["capsule"]["state_equivalent_to_uninterrupted"]
    assert not conditions["lite"]["state_equivalent_to_uninterrupted"]
    assert not conditions["sidecar"]["state_equivalent_to_uninterrupted"]
    assert conditions["capsule"]["vendor_event_count"] == 4
    assert conditions["capsule"]["ranked_history_count"] >= conditions["lite"]["ranked_history_count"]
