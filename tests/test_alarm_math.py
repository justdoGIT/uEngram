import math

import numpy as np
import pytest

from bench.alarm_math import arl0, markov_fit, simulate_arl0
from uengram.alarm import AlarmConfig, AlarmLogic, AlarmPolicy
from uengram.persistence import make_filter


def _feed(model, k, ys, window=0):
    f, out = make_filter(model, k, window), []
    for y in ys:
        f.step(y)
        out.append((f.trigger, f.clear))
    return out


def test_ewma_needs_exactly_k_consecutive_and_k_to_clear():
    for k in (1, 2, 3, 5, 6):
        up = _feed("ewma", k, [True] * k)
        assert not any(t for t, _ in up[:-1]) and up[-1][0]
        down = _feed("ewma", k, [True] * 50 + [False] * k)
        assert not any(c for _, c in down[50:-1]) and down[-1][1]


def test_unknown_holds_every_filter():
    for model in ("counter", "consecutive", "m_of_n", "ewma"):
        triggers = [t for t, _ in _feed(model, 3, [True, None, True, None, True])]
        assert triggers == [False, False, False, False, True], model  # unknowns neither add nor reset


def test_consecutive_resets_on_false_but_counter_only_steps_down():
    ys = [True, True, False, True, True]
    assert not any(t for t, _ in _feed("consecutive", 3, ys))
    assert _feed("counter", 3, ys + [True])[-1][0]  # 1,2,1,2,3


def test_m_of_n_counts_within_the_window():
    assert _feed("m_of_n", 3, [True, False, True, False, True], window=6)[-1][0]
    assert not _feed("m_of_n", 3, [True, False, False, True, False, False, True], window=4)[-1][0]


@pytest.mark.parametrize("p,k", [(0.1, 3), (0.3, 2), (0.05, 4)])
def test_independent_consecutive_arl_matches_closed_form(p, k):
    assert arl0("consecutive", k, p, p) == pytest.approx((1 - p ** k) / ((1 - p) * p ** k), rel=1e-9)


@pytest.mark.parametrize("model", ["counter", "consecutive", "m_of_n", "ewma"])
def test_single_window_trigger_is_geometric(model):
    assert arl0(model, 1, 0.2, 0.2) == pytest.approx(5.0, rel=1e-6)


@pytest.mark.parametrize("model", ["counter", "consecutive", "m_of_n", "ewma"])
def test_exact_arl_matches_simulation_of_the_real_filter_under_clustering(model):
    exact = arl0(model, 3, 0.05, 0.5)
    mean, se = simulate_arl0(model, 3, 0.05, 0.5, reps=4000, seed=1)
    assert abs(exact - mean) < 4 * se


def test_clustering_shortens_time_to_false_alarm():
    p, b = 0.037, 0.774  # MetroPT3 healthy novelty exceedances: rate and P(next | now)
    a = p * (1 - b) / (1 - p)  # stationary rate p
    for model in ("counter", "consecutive"):
        assert arl0(model, 3, a, b) < arl0(model, 3, p, p) / 50


@pytest.mark.parametrize("model", ["counter", "consecutive", "m_of_n"])
def test_never_two_in_a_row_makes_k3_unreachable(model):
    # b = 0: an exceedance is always followed by a normal window, so k = 3 never fires
    assert arl0(model, 3, 0.2, 0.0, window=4 if model == "m_of_n" else 0) == float("inf")
    assert math.isfinite(arl0(model, 1, 0.2, 0.0))


def test_markov_fit_recovers_transition_probabilities():
    rng = np.random.default_rng(3)
    x, y = np.zeros(200000, bool), False
    for i in range(len(x)):
        y = rng.random() < (0.6 if y else 0.02)
        x[i] = y
    fit = markov_fit(x, np.ones(len(x), bool))
    assert fit["a"] == pytest.approx(0.02, abs=0.002) and fit["b"] == pytest.approx(0.6, abs=0.01)


def _policy(logic):
    rng = np.random.default_rng(0)
    return AlarmPolicy(AlarmConfig(alpha=0.05, k_novelty=3, k_health=0), logic=logic).calibrate(
        rng.uniform(0, 1, 400), np.zeros(400), np.zeros(400, int), np.ones(400, bool))


def test_kleene_corroboration_lowers_on_known_normal_even_when_evidence_unknown():
    p = _policy(AlarmLogic())
    nov = [9.0, 9.0, 0.0, 0.0, 9.0]  # exceed, exceed, normal, normal, exceed
    active, _ = p.run(np.arange(5), np.array(nov), np.zeros(5), ["absorb_semantic"] * 5,
                      np.zeros(5, int), np.ones(5, bool), evidence=[True, True, None, None, True])
    assert not active.any() and p.c_nov == 1  # 1, 2, 1, 0, 1: no exceedance means normal


def test_veto_hold_keeps_counts_that_veto_normal_would_drain():
    nov, acts = np.full(5, 9.0), ["absorb_semantic", "absorb_semantic", "new_candidate",
                                  "new_candidate", "absorb_semantic"]
    normal = _policy(AlarmLogic()).run(np.arange(5), nov, np.zeros(5), acts, np.zeros(5, int),
                                       np.ones(5, bool))[0]
    hold = _policy(AlarmLogic(veto="hold")).run(np.arange(5), nov, np.zeros(5), acts,
                                                np.zeros(5, int), np.ones(5, bool))[0]
    assert not normal.any() and hold.tolist() == [False, False, False, False, True]


def test_unseen_state_is_unknown_until_commissioned_then_judged():
    p = _policy(AlarmLogic(unbaselined="commission"))  # alpha 0.05 -> 19 commissioning windows
    need = math.ceil(0.95 / 0.05)
    n = need + 4
    nov = np.r_[np.full(need, 1.0), np.full(4, 50.0)]
    active, _ = p.run(np.arange(n), nov, np.zeros(n), ["absorb_semantic"] * n, np.full(n, 7),
                      np.ones(n, bool))
    assert 7 in p.baselines and p.baseline_counts[7]["commissioned_online"]
    assert not active[:need].any() and active[-1]  # judged only after its own reference exists
