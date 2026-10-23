import numpy as np
import pytest

from bench.alarm_definition import Corroboration, Definition, vote
from bench.alarm_eval import apply_policy
from bench.alarm_sensitivity import BASELINE
from bench.error_gated_sweep import ALARM_CONFIGS
from bench.streams import load_stream
from uengram.alarm import AlarmConfig, AlarmPolicy

T0 = np.datetime64("2020-01-01T00:00:00", "s")


def _policy(k):
    rng = np.random.default_rng(0)
    return AlarmPolicy(AlarmConfig(alpha=0.05, k_novelty=k, k_health=0)).calibrate(
        rng.uniform(0, 1, 400), np.zeros(400), np.zeros(400, int), np.ones(400, bool))


def _run(policy, evidence):
    n = len(evidence)
    return policy.run(np.arange(n), np.full(n, 9.0), np.zeros(n), ["absorb_semantic"] * n,
                      np.zeros(n, int), np.ones(n, bool), evidence=evidence)


def test_false_evidence_counts_as_normal_and_unknown_holds_the_counters():
    active, opened = _run(_policy(2), [True, None, True, False, False, False])
    assert opened.tolist() == [0, 0, 1, 0, 0, 0]  # the unknown window neither raised nor reset
    assert active.tolist() == [0, 0, 1, 1, 0, 0]  # two uncorroborated windows drain k=2
    never, _ = _run(_policy(2), [True, False, True, False, True, False])
    assert not never.any()  # corroboration that keeps failing never accumulates


def test_vote_is_three_valued_and_counts_errors_within_the_lookback():
    end = T0 + np.arange(4) * np.timedelta64(600, "s")
    valid = np.ones(4, bool)
    true = np.array([[1, 0], [0, 0], [0, 0], [0, 0]], bool)
    known = np.array([[1, 1], [1, 1], [1, 0], [1, 0]], bool)
    assert vote(true, known, end, valid, Corroboration("x", 1, 0.0)).tolist() == [True, False, None, None]
    assert vote(true, known, end, valid, Corroboration("x", 1, 1 / 3)).tolist() == [True, True, True, None]
    # At 30 min the error is 30 min old (outside 20 min) and only one source is unknown:
    # even if it were true, 1 < m = 2, so the vote is definitely false.
    assert vote(true, known, end, valid, Corroboration("x", 2, 1 / 3)).tolist() == [False, False, None, False]


@pytest.fixture(scope="module")
def metropt3():
    return Definition(load_stream("metropt3"))


def test_all_true_corroboration_reproduces_the_actual_alarm(metropt3):
    reference, _, _ = apply_policy(metropt3.stream, metropt3.scores, BASELINE)
    every = np.full(len(metropt3.test), True, dtype=object)
    corroborated, _, _ = apply_policy(metropt3.stream, metropt3.scores, BASELINE, every)
    for key in ("events_detected", "false_alarm_per_week", "alarm_time_share_clean", "lead_hours"):
        assert corroborated[key] == pytest.approx(reference[key], nan_ok=True)


@pytest.mark.parametrize("index", [0, 24, 42])
@pytest.mark.parametrize("c", [Corroboration("dataset", 1, 0.0), Corroboration("dataset", 2, 24.0),
                               Corroboration("dataset", 1, 6.0)])
def test_two_valued_corroboration_only_removes_alarm_time(metropt3, index, c):
    cfg = ALARM_CONFIGS[index]
    policy, _, _ = metropt3.evaluate(cfg, None)
    corroborated, details, _ = metropt3.evaluate(cfg, c)
    assert corroborated["alarm_time_share_clean"] <= policy["alarm_time_share_clean"]
    assert corroborated["events_detected"] <= min(policy["events_detected"],
                                                  details["corroboration_ceiling_events"])


@pytest.mark.parametrize("index", [0, 20])  # single-counter configs: k_health == 0
@pytest.mark.parametrize("c", [None, Corroboration("dataset", 1, 6.0), Corroboration("calibrated", 1, 24.0)])
def test_recorded_exceedances_and_votes_rebuild_the_alarm(metropt3, index, c):
    """Condition 2 (the policy's counted exceedance), the vote and one on/off-delay counter
    reproduce the alarm exactly. The combined condition is the Kleene AND: exceedance AND
    TRUE raises, no exceedance or a FALSE vote lowers (even when the vote is unknown), and
    only exceedance AND UNKNOWN holds."""
    cfg = ALARM_CONFIGS[index]
    assert cfg.k_health == 0
    evidence = None if c is None else metropt3.evidence(cfg, c)
    _, d, pol = apply_policy(metropt3.stream, metropt3.scores, cfg, evidence)
    counter, on, active = 0, False, np.zeros(len(metropt3.test), bool)
    for j in np.flatnonzero(d["valid"]):
        v = True if evidence is None else evidence[j]
        if v is False or not pol.exceeded[j]:
            counter = max(counter - 1, 0)
        elif v is True:
            counter = min(counter + 1, cfg.k_novelty)
        on = (on and counter > 0) or counter >= cfg.k_novelty
        active[j] = on
    assert (active == (d["active"] & d["valid"])).all()
