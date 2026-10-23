import json

import pytest

from bench.alarm_sensitivity import BASELINE, OFAT_GRIDS, _record, pairwise, summarize_ofat
from bench.streams import load_stream
from bench.v2_scoring import cached_scores
from dataclasses import replace
from pathlib import Path

STREAM = load_stream("metropt3")
SCORES = cached_scores(STREAM, "uengram", "dsp", 0,
                       str(Path.home() / ".local/state/uengram/v2scores"))


def test_baseline_matches_frozen_v2_metrics():
    row = _record(STREAM, SCORES, BASELINE, {"param": None, "value": None, "is_baseline": True})
    assert row["metrics"]["events_detected"] == 3.0
    assert row["event3_detected"] is False
    assert row["metrics"]["false_alarm_per_week"] == pytest.approx(0.7407187085012238)
    assert row["metrics"]["alarm_time_share_clean"] == pytest.approx(0.005996644230718761)


def test_too_tight_alpha_reports_calibration_unavailable_not_a_crash():
    cfg = replace(BASELINE, alpha=0.0003)
    row = _record(STREAM, SCORES, cfg, {"param": "alpha", "value": 0.0003, "is_baseline": False})
    assert row["metrics"] is None
    assert row["calibration_unavailable"]
    assert row["satisfies_recall_and_budget"] is False


def test_health_channel_requires_both_untrusted_baseline_relaxed_and_k_health_enabled(tmp_path):
    rows = pairwise(STREAM, SCORES, ("health_needs_baseline", "k_health"),
                    tmp_path / "pairwise.jsonl", open("/dev/null", "w"))
    by_cell = {(r["varied"]["health_needs_baseline"], r["varied"]["k_health"]): r for r in rows}
    # Neither margin alone unlocks event 3: an AND-gate, not an additive effect.
    for k in OFAT_GRIDS["k_health"]:
        assert by_cell[(True, k)]["metrics"]["events_detected"] == 3.0
    assert by_cell[(False, 0)]["metrics"]["events_detected"] == 3.0
    for k in (1, 3, 6, 10):
        assert by_cell[(False, k)]["metrics"]["events_detected"] == 4.0
        assert by_cell[(False, k)]["event3_detected"] is True


def test_new_state_alert_has_almost_no_effect_once_health_channel_is_open(tmp_path):
    rows = pairwise(STREAM, SCORES, ("new_state_alert", "k_health"),
                    tmp_path / "pairwise.jsonl", open("/dev/null", "w"),
                    fixed={"health_needs_baseline": False})
    by_cell = {(r["varied"]["new_state_alert"], r["varied"]["k_health"]): r for r in rows}
    for k in (3, 6, 10):
        off, on = by_cell[(False, k)], by_cell[(True, k)]
        assert off["metrics"]["events_detected"] == on["metrics"]["events_detected"] == 4.0
        assert off["metrics"]["alarm_time_share_clean"] == on["metrics"]["alarm_time_share_clean"]
    # k_health=1 (fastest open/close) keeps a small residual veto effect: the novelty path
    # occasionally extends or flicker-reopens an episode the health path alone would not.
    off1, on1 = by_cell[(False, 1)], by_cell[(True, 1)]
    assert off1["metrics"]["events_detected"] == on1["metrics"]["events_detected"] == 4.0
    assert off1["metrics"]["alarm_time_share_clean"] == pytest.approx(
        on1["metrics"]["alarm_time_share_clean"], rel=0.01)


def test_summarize_ofat_flags_calibration_unavailable_without_crashing(tmp_path):
    rows = [
        _record(STREAM, SCORES, BASELINE, {"param": None, "value": None, "is_baseline": True}),
        _record(STREAM, SCORES, replace(BASELINE, alpha=0.0003),
                {"param": "alpha", "value": 0.0003, "is_baseline": False}),
        _record(STREAM, SCORES, replace(BASELINE, alpha=0.01),
                {"param": "alpha", "value": 0.01, "is_baseline": False}),
    ]
    summary = summarize_ofat(rows)
    assert summary["by_parameter"]["alpha"]["calibration_unavailable_at"] == [0.0003]
    assert len(summary["by_parameter"]["alpha"]["rows"]) == 1
    assert "elasticity_vs_baseline" in summary["by_parameter"]["alpha"]


def test_no_single_or_explored_triple_config_satisfies_recall_and_budget():
    ofat_path = Path("results/v4_alarm_sensitivity_ofat.jsonl")
    rows = [json.loads(line) for line in ofat_path.read_text().splitlines()]
    assert not any(r["satisfies_recall_and_budget"] for r in rows)
