import json
import math

import numpy as np
import pytest

from uengram.retrieval import (
    RetrievalWeights, decode_vector, encode_vector, rank_events, score_event,
)


def _event(asset="pump-17", subsystem="bearing", mode="steady", vec=None, t_ms=None, severity=None):
    meta = {"asset": asset, "subsystem": subsystem, "mode": mode}
    if vec is not None:
        meta["vec"] = encode_vector(vec)
    if t_ms is not None:
        meta["t_ms"] = t_ms
    if severity is not None:
        meta["severity"] = severity
    return {"structured_value": json.dumps(meta)}


def test_encode_decode_vector_roundtrips_within_quantization_error():
    rng = np.random.default_rng(0)
    v = rng.normal(size=16) * 3
    restored = decode_vector(encode_vector(v))
    assert np.max(np.abs(restored - v)) < np.abs(v).max() / 254 + 1e-9


def test_decode_vector_rejects_malformed_snapshots():
    assert decode_vector(None) is None
    assert decode_vector({"q": "not base64!!", "s": 1.0}) is None
    assert decode_vector({"q": encode_vector(np.ones(4))["q"], "s": 0.0}) is None
    assert decode_vector("not a mapping") is None


def test_score_event_components_and_missing_fields():
    w = RetrievalWeights()
    q = np.zeros(4)
    sim, rec, sev, missing = score_event(json.loads(_event(vec=q, t_ms=1000, severity=2.0)["structured_value"]),
                                         q, now_ms=1000, w=w)
    assert sim == pytest.approx(1.0) and rec == pytest.approx(1.0) and missing == ()
    assert sev == pytest.approx(math.tanh(2.0))

    bare = json.loads(_event()["structured_value"])
    sim2, rec2, sev2, missing2 = score_event(bare, q, now_ms=1000, w=w)
    assert (sim2, rec2, sev2) == (0.0, 0.0, 0.0)
    assert set(missing2) == {"vector", "time", "severity"}


def test_recency_decays_by_half_at_the_configured_half_life():
    w = RetrievalWeights(half_life_ms=1000)
    meta = json.loads(_event(t_ms=0)["structured_value"])
    _, rec, _, _ = score_event(meta, np.zeros(2), now_ms=1000, w=w)
    assert rec == pytest.approx(0.5)


def test_negative_severity_floors_to_zero_not_a_negative_score():
    w = RetrievalWeights()
    meta = json.loads(_event(severity=-5.0)["structured_value"])
    _, _, sev, missing = score_event(meta, np.zeros(2), now_ms=0, w=w)
    assert sev == 0.0 and "severity" not in missing


def test_weights_reject_invalid_configuration():
    with pytest.raises(ValueError, match="similarity weight"):
        RetrievalWeights(similarity=-1.0)
    with pytest.raises(ValueError, match="at least one weight"):
        RetrievalWeights(similarity=0.0, recency=0.0, severity=0.0)
    with pytest.raises(ValueError, match="half_life_ms"):
        RetrievalWeights(half_life_ms=0.0)


def test_rank_events_orders_by_total_score_best_first():
    q = np.zeros(4)
    near = _event(vec=q, t_ms=1000, severity=0.1)
    far = _event(vec=q + 5, t_ms=1000, severity=0.1)
    severe = _event(vec=q, t_ms=1000, severity=10.0)
    ranked = rank_events([far, near, severe], q, now_ms=1000, asset_id="pump-17", subsystem_id="bearing")
    assert [r.event for r in ranked] == [severe, near, far]
    assert ranked[0].score >= ranked[1].score >= ranked[2].score


def test_rank_events_filters_strictly_by_identity_never_by_score():
    q = np.zeros(2)
    mine = _event(asset="pump-17", vec=q, t_ms=0)
    other_asset = _event(asset="pump-18", vec=q, t_ms=0)
    other_mode = _event(mode="startup", vec=q, t_ms=0)
    ranked_all = rank_events([mine, other_asset, other_mode], q, now_ms=0,
                             asset_id="pump-17", subsystem_id="bearing")
    assert {id(r.event) for r in ranked_all} == {id(mine), id(other_mode)}  # mode unfiltered here
    ranked_mode = rank_events([mine, other_mode], q, now_ms=0,
                              asset_id="pump-17", subsystem_id="bearing", operating_mode="steady")
    assert [r.event for r in ranked_mode] == [mine]


def test_rank_events_respects_limit_and_skips_malformed_entries():
    q = np.zeros(2)
    events = [_event(vec=q, t_ms=i) for i in range(5)] + [{"structured_value": "not json"}, object()]
    ranked = rank_events(events, q, now_ms=5, asset_id="pump-17", subsystem_id="bearing", limit=2)
    assert len(ranked) == 2


def test_rank_events_validates_query_and_limit():
    with pytest.raises(ValueError, match="query_vector"):
        rank_events([], np.array([np.nan, 0.0]), now_ms=0, asset_id="a", subsystem_id="b")
    with pytest.raises(ValueError, match="limit"):
        rank_events([], np.zeros(2), now_ms=0, asset_id="a", subsystem_id="b", limit=-1)
