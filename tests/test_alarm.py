import numpy as np
import pytest

from uengram.alarm import AlarmConfig, AlarmPolicy

T = np.arange(10000)


def _policy(cfg=AlarmConfig(), n=300, state=None, features=None, names=()):
    rng = np.random.default_rng(0)
    nov, health = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    state = np.zeros(n, int) if state is None else state
    return AlarmPolicy(cfg, names).calibrate(nov, health, state, np.ones(n, bool), features)


def _stream(p, nov, health=None, action="absorb_semantic", state=0, mon=True):
    n = len(nov)
    health = np.zeros(n) if health is None else health
    actions = [action] * n if isinstance(action, str) else action
    state = np.full(n, state) if np.isscalar(state) else state
    mon = np.full(n, mon) if np.isscalar(mon) else mon
    return p.run(T[:n], np.asarray(nov, float), np.asarray(health, float), actions, state, mon)


def test_isolated_exceedances_never_alarm_but_a_sustained_run_does():
    hi, lo = 5.0, 0.0
    active, _ = _stream(_policy(AlarmConfig(k_novelty=3, k_health=0)),
                        [hi, lo, hi, lo, hi, lo, hi, lo])
    assert not active.any()
    p = _policy(AlarmConfig(k_novelty=3, k_health=0))
    long_run = [hi] * 50 + [lo] * 5
    active, opened = _stream(p, long_run)
    # Opens on the third consecutive exceedance. The counter is capped at k, so
    # after fifty exceedances the alarm still clears after three normal windows
    # (a fixed off-delay), not after fifty.
    assert np.flatnonzero(opened).tolist() == [2]
    assert active[2:52].all() and not active[52:].any()
    assert p.episodes[0]["close_index"] == 52


def test_health_in_a_state_without_baseline_alerts_instead_of_counting():
    def run(flag):
        q = AlarmPolicy(AlarmConfig(k_novelty=0, k_health=1, health_needs_baseline=flag)).calibrate(
            np.zeros(400), np.random.default_rng(4).uniform(0, 1, 400), np.repeat([0, 1], 200),
            np.ones(400, bool))
        _, opened = _stream(q, [0.0], health=[9.0], state=7)  # state 7: no baseline
        return q, opened.any()
    q, alarmed = run(True)
    assert not alarmed and q.alerts[-1]["kind"] in ("unbaselined_state", "unbaselined_health")
    # Without the option the same exceedance falls back to the global baseline and alarms.
    q, alarmed = run(False)
    assert alarmed


def test_untrusted_health_does_not_turn_a_new_state_into_an_alarm():
    def policy():
        return AlarmPolicy(AlarmConfig(k_novelty=1, k_health=0, health_needs_baseline=True)).calibrate(
            np.random.default_rng(5).uniform(0, 1, 400), np.random.default_rng(6).uniform(0, 1, 400),
            np.repeat([0, 1], 200), np.ones(400, bool))
    # New candidate with high novelty and high health in a state without baseline:
    # health cannot be judged there, so the window is a new-state alert.
    q = policy()
    _, opened = _stream(q, [9.0], health=[9.0], action="new_candidate", state=7)
    assert not opened.any()
    # In a baselined state the same high health is trusted and novelty counts.
    _, opened = _stream(policy(), [9.0], health=[9.0], action="new_candidate", state=0)
    assert opened.any()



def test_disabled_health_branch_does_not_delay_novelty_alarm_clear():
    p = _policy(AlarmConfig(k_novelty=1, k_health=0))
    threshold, _ = p.thresholds(0)
    nov = [threshold + 1, threshold - 1, threshold - 1]
    health = [1e6, 1e6, 1e6]
    active, opened = _stream(p, nov, health=health)
    assert np.flatnonzero(opened).tolist() == [0]
    assert active.tolist() == [True, False, False]
    assert p.c_health == 0

def test_unmonitorable_windows_hold_the_counter():
    p = _policy(AlarmConfig(k_novelty=3, k_health=0))
    # exceed, exceed, stopped machine (score irrelevant), exceed: the third
    # counted exceedance opens; the stopped window neither adds nor subtracts.
    _, opened = _stream(p, [5, 5, 0, 5], mon=np.array([True, True, False, True]))
    assert np.flatnonzero(opened).tolist() == [3]


def _two_state_policy():
    rng = np.random.default_rng(1)
    state = np.repeat([0, 1], 200)
    nov = np.where(state == 0, rng.uniform(0, 1, 400), rng.uniform(10, 11, 400))
    return AlarmPolicy(AlarmConfig(k_novelty=1, k_health=0)).calibrate(
        nov, np.zeros(400), state, np.ones(400, bool))


def test_per_state_thresholds_and_fallback_for_unbaselined_state():
    p = _two_state_policy()
    assert p.thresholds(0)[0] < 1.0 < 10.0 < p.thresholds(1)[0]
    # A score routine for state 1 is an exceedance in state 0 only.
    assert _stream(_two_state_policy(), [5.0], state=0)[1].any()
    assert not _stream(_two_state_policy(), [5.0], state=1)[1].any()
    # State 7 had no calibration windows: global threshold, one alert per entry.
    q = _two_state_policy()
    _stream(q, [0.0, 0.0, 0.0, 0.0], state=np.array([7, 7, 0, 7]))
    assert [(a["index"], a["kind"]) for a in q.alerts] == [(0, "unbaselined_state"),
                                                           (3, "unbaselined_state")]
    assert q.thresholds(7) == q.thresholds(-1)


def test_new_candidate_with_normal_health_alerts_instead_of_counting():
    p = _policy(AlarmConfig(k_novelty=1, k_health=0))
    _, opened = _stream(p, [5.0], health=[0.0], action="new_candidate")
    assert not opened.any() and p.alerts[-1]["kind"] == "new_state"
    # Health also exceeds: this is not a benign new state, so novelty counts.
    _, opened = _stream(_policy(AlarmConfig(k_novelty=1, k_health=0)), [5.0],
                        health=[5.0], action="new_candidate")
    assert opened.any()


def test_adaptive_level_tracks_alpha_on_healthy_windows_only():
    rng = np.random.default_rng(2)
    cfg = AlarmConfig(alpha=0.05, k_novelty=10**9, k_health=0, aci_gamma=0.02)
    shifted = rng.uniform(0.5, 1.5, 6000)  # benign upward shift of the normal state
    p, exceeded = _policy(cfg), []
    for i, s in enumerate(shifted):
        exceeded.append(s > p.thresholds(0)[0])
        p.step(i, i, float(s), 0.0, "absorb_semantic", 0, True)
    assert abs(np.mean(exceeded[2000:]) - 0.05) < 0.02
    fixed = _policy(AlarmConfig(alpha=0.05, k_novelty=10**9, k_health=0))
    assert np.mean(shifted > fixed.thresholds(0)[0]) > 0.4
    # With health exceeding, windows are not judged healthy: the level is frozen.
    q = _policy(cfg)
    _stream(q, shifted[:500], health=np.full(500, 9.0))
    assert q.level == pytest.approx(0.05)


def test_episode_freeze_frame_records_state_and_largest_deviations():
    names = ("load_share", "current_a", "oil_c")
    rng = np.random.default_rng(3)
    feats = np.column_stack([rng.normal(0.1, 0.01, 300), rng.normal(2.0, 0.1, 300),
                             rng.normal(60, 1.0, 300)])
    p = _policy(AlarmConfig(k_novelty=1, k_health=0), features=feats, names=names)
    p.step(0, "2020-03-01T00:10", 5.0, 0.0, "absorb_episodic", 0, True,
           np.array([0.9, 2.05, 75.0]))
    ep = p.episodes[0]
    assert ep["state"] == 0 and ep["trigger"] == "novelty" and ep["action"] == "absorb_episodic"
    assert ep["snapshot"]["load_share"] == pytest.approx(0.9)
    assert [name for name, _ in ep["top_deviations"][:2]] == ["load_share", "oil_c"]
