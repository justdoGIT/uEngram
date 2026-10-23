import urllib.error
from unittest import mock

import pytest

from bench.advisory_eval import assessment_schedule
from bench.jev_advisory import QUESTIONS, assess, causal_state, parse_response
from bench.jev_experiment import advisory_elevation, advisory_timelines


def _response():
    answers = {}
    for name, q in QUESTIONS.items():
        opts = list(q["criteria"])
        p = {option: 1 / len(opts) for option in opts}
        answers[name] = {"type": "choice", "choice": opts[0], "confidence": 0,
                         "probabilities": p}
    return {"model": "typesafe/jev-1.14-20261014", "answers": answers, "usage": {"input_tokens": 1}}


def test_causal_input_excludes_identifiers_and_future_fields():
    state = causal_state({"novelty": 2, "fault_label": "secret", "stream": "asset-x",
                          "future": [1, 2], "start_date": "2026-01-01",
                          "physical": {"tp2_bar": 4, "lps_share": 1, "Oil_Level": 2,
                                       "Pressure_switch": 1}},
                         {"health": 1, "event_id": 4}, 3.5)
    assert state["current_candidate_window"] == {"novelty": 2, "physical": {"tp2_bar": 4}}


def test_causal_time_is_nonnegative():
    with pytest.raises(ValueError):
        causal_state({}, None, -1)


def test_unknown_and_malformed_responses_are_rejected():
    with pytest.raises(ValueError):
        parse_response({"error": "unavailable"})
    bad = _response()
    bad["answers"]["condition"]["probabilities"]["fault_consistent"] = 2
    with pytest.raises(ValueError):
        parse_response(bad)


def test_valid_response_preserves_distribution_and_usage():
    parsed = parse_response(_response())
    assert parsed["model"] == "typesafe/jev-1.14-20261014"
    assert parsed["usage"]["input_tokens"] == 1
    assert set(parsed["answers"]) == set(QUESTIONS)


def test_candidate_schedule_entries_and_hourly_assessments():
    states = [1, 2, 2, 2, 2, 0, 3, 3]
    ends = [0, 10, 1800, 3610, 5000, 6000, 7000, 10600]
    assert assessment_schedule(states, ends).tolist() == [False, True, False, True, False, False, True, True]


def test_unknown_or_low_confidence_never_elevates_and_review_expires():
    answer = {"choice": "review", "probabilities": {"review": .9}, "confidence": .6}
    assert advisory_elevation(answer)
    assert not advisory_elevation({**answer, "confidence": .49})
    assert not advisory_elevation({**answer, "choice": "uncertain"})
    states = [2, 2, 2, 2, 1, 3]
    ends = [0, 100, 3500, 3700, 4000, 5000]
    assessments = {i: {"available": True, "answers": {"review": answer}} for i in (0, 5)}
    timelines = advisory_timelines([False] * 6, states, ends, assessments)
    assert timelines["advisory_assessment_only"].tolist() == [True, True, True, False, False, True]


def test_advisory_rejection_and_outage_have_distinct_baseline_effects():
    review = {"choice": "review", "probabilities": {"review": .9}, "confidence": .6}
    reject = {"choice": "no_review", "probabilities": {"review": .1}, "confidence": .8}
    assessments = {
        0: {"available": True, "answers": {"review": review}},
        1: {"available": False, "answers": None},
        2: {"available": True, "answers": {"review": reject}},
    }
    timelines = advisory_timelines([False, True, True], [2, 3, 3], [0, 100, 200], assessments)
    assert timelines["baseline"].tolist() == [False, True, True]
    assert timelines["conservative_preserve_baseline"].tolist() == [True, True, True]
    assert timelines["advisory_assessment_only"].tolist() == [True, True, False]


def test_assess_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    result = assess({})
    assert result.available is False
    assert result.error == "missing OPENROUTER_API_KEY"
    assert result.answers is None


def test_assess_fail_closed_on_http_error(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    exc = urllib.error.HTTPError("https://openrouter.ai/api/alpha/decisions", 402,
                                 "Insufficient credits", {}, None)
    with mock.patch("urllib.request.urlopen", side_effect=exc):
        result = assess({})
    assert result.available is False
    assert result.error == "HTTP 402"
    assert result.answers is None  # never healthy on provider failure
