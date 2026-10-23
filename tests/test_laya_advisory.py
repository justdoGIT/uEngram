import json
from pathlib import Path

import numpy as np
import pytest

from bench import advisory_eval as ev
from bench import laya_advisory as la
from bench.advisory_experiment import run_assessments
from bench.jev_advisory import QUESTIONS

EPOCH = 1583035794  # absolute start time that must never reach the request text


def _trace(n=4, lps=123.456):
    names = list(la.FEATURES) + ["lps_share", "oil_level", "pressure_switch"]
    feats = np.tile(np.arange(len(names), dtype=float) + 0.5, (n, 1))
    feats[:, names.index("lps_share")] = lps
    feats[:, names.index("oil_level")] = 777.25
    feats[:, names.index("pressure_switch")] = 888.75
    return {"decision_state": np.array([1, 1, 2, 3][:n]), "state": np.ones(n, int),
            "start": EPOCH + 600 * np.arange(n), "end": EPOCH + 600 * np.arange(1, n + 1),
            "novelty": np.array([1.0, 1.0, 3.0, 4.0][:n]), "novelty_threshold": np.full(n, 2.0),
            "health": np.full(n, 0.5), "health_threshold": np.full(n, 1.0),
            "features": feats, "feature_names": np.asarray(names),
            "baseline_counts": json.dumps({"1": {"novelty_threshold_supported": True,
                                                 "monitorable_windows": 400}}),
            "config": json.dumps({"per_state": True})}


def _answers(review=0.9, choice="review"):
    out = {}
    for name, q in QUESTIONS.items():
        opts = list(q["criteria"])
        rest = (1 - review) / (len(opts) - 1)
        probs = {o: (review if o == opts[0] else rest) for o in opts}
        out[name] = {"type": "choice", "choice": choice if name == "review" else opts[0],
                     "probabilities": probs, "confidence": 0.1, "answer_confidence": review}
    return out


class FakeAgent:
    def __init__(self, result=None, exc=None):
        self.result, self.exc, self.calls = result, exc, 0

    def predict(self, state, questions):
        self.calls += 1
        if self.exc:
            raise self.exc
        return self.result or {"model": "laya-rl-agent", "answers": _answers(),
                               "usage": {"truncated": False, "state_tokens_dropped": 0,
                                         "truncated_questions": []}}


def test_state_excludes_warning_channels_identifiers_and_absolute_time():
    text = la.StateBuilder(_trace()).build(3, 2)
    for banned in ("123.456", "777.25", "888.75", "lps", "LPS", "oil level", "pressure switch",
                   str(EPOCH), "metropt", "2020"):
        assert banned not in text
    assert "Hours since this flagged period began: 0.2" in text
    assert "TP2 pressure" in text and "2.00 times its calibrated threshold" in text


def test_state_reference_is_causal():
    z = _trace()
    before = la.StateBuilder(z).build(2, 2)
    z["features"][3] += 1000.0  # a later window must not change an earlier request
    assert la.StateBuilder(z).build(2, 2) == before


def test_due_states_only_candidates_on_schedule():
    z = _trace()
    due = ev.assessment_schedule(z["decision_state"], z["end"])
    assert list(la.due_states(z, due)) == [2]


def test_threshold_edges():
    assert la.elevates(_answers(0.80))
    assert not la.elevates(_answers(0.7999))
    assert not la.elevates(_answers(0.95, choice="uncertain"))
    assert la.typesafe_confidence({"a": 0.8, "b": 0.1, "c": 0.1}) == pytest.approx(0.7)


@pytest.mark.parametrize("agent", [
    FakeAgent(exc=RuntimeError("boom")),
    FakeAgent(result={"answers": _answers(), "usage": {"truncated": True}}),
    FakeAgent(result={"answers": {}, "usage": {"truncated": False}}),
    FakeAgent(result={"answers": _answers(), "usage": {"truncated": False, "state_tokens_dropped": 3}}),
])
def test_failures_are_explicitly_unavailable(agent):
    res = la.assess(agent, "synthetic")
    assert res["available"] is False and res["elevated"] is None and res["error"]


def test_unavailable_never_elevates_or_silences_baseline():
    states, ends = [2, 3, 3, 1], [0, 600, 1200, 1800]
    elevated, unavailable = ev.fresh_timeline(states, ends, {0: True, 1: None})
    assert elevated.tolist() == [True, False, False, False]
    assert unavailable.tolist() == [False, True, True, False]
    t = ev.interpretations([False, True, True, False], elevated, unavailable)
    assert t["advisory_assessment_only"].tolist() == [True, True, True, False]
    assert t["conservative_preserve_baseline"].tolist() == [True, True, True, False]


def test_elevation_expires_after_one_observed_hour():
    elevated, _ = ev.fresh_timeline([2, 2, 2], [0, 3600, 3601], {0: True})
    assert elevated.tolist() == [True, True, False]


def test_episode_openings_keep_policy_continuations():
    z = {"active": np.array([1, 0, 1, 0, 0, 0], bool), "opened": np.array([1, 0, 0, 0, 0, 0], bool)}
    assert ev.episode_openings(z, z["active"]).tolist() == z["opened"].tolist()
    extra = z["active"] | np.array([0, 0, 0, 0, 1, 0], bool)
    assert ev.episode_openings(z, extra).tolist() == [True, False, False, False, True, False]


def test_cache_resume_does_not_repeat_model_calls(tmp_path):
    prov = {"revision": la.REVISION, "laya_version": "test"}
    due = {2: "synthetic a", 5: "synthetic b"}
    agent = FakeAgent()
    first = run_assessments(agent, prov, due, tmp_path, None, "t")
    again = run_assessments(agent, prov, due, tmp_path, None, "t")
    assert agent.calls == 2 and first == again and len(list(tmp_path.iterdir())) == 2


def test_calibration_perfect_and_worst():
    assert ev.calibration([1, 0], [1, 0]) == {"n": 2, "positives": 1, "brier": 0.0, "ece": 0.0}
    assert ev.calibration([0, 1], [1, 0])["brier"] == 1.0


def test_token_budget_reports_truncation():
    pytest.importorskip("laya")
    from huggingface_hub import snapshot_download
    from laya.agent import Agent, _load_tokenizer
    try:
        root = Path(snapshot_download(la.MODEL_ID, revision=la.REVISION, local_files_only=True,
                                      allow_patterns=["rl_agent_config.json", "tokenizer/*"]))
    except Exception:
        pytest.skip("Laya checkpoint not cached")
    cfg = json.loads((root / "rl_agent_config.json").read_text())

    class TokAgent:
        tok = _load_tokenizer(str(root / "tokenizer"), cfg)
        _to_internal = staticmethod(Agent._to_internal)

    TokAgent.cfg = cfg
    short = la.budget(TokAgent, [la.StateBuilder(_trace()).build(3, 2)])
    assert short["truncated_requests"] == 0 and not short["head_truncated_questions"]
    assert short["max_sequence_tokens"] <= short["max_len"]
    assert la.budget(TokAgent, ["pressure " * 2000])["truncated_requests"] == 1
