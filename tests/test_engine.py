import numpy as np

from uengram import lifecycle as lc
from uengram.engine import Engram, EngramConfig
from uengram.stats import ewma_inflation

D = 16
SIG = 0.5


def make(**kw):
    return Engram(EngramConfig(**kw), D, SIG * np.ones(D), 1536)


def pt(rng, off=0.0):
    z = rng.normal(0, SIG, D)
    z[0] += off
    return z


def count(eng, *kinds):
    return int(np.isin(eng.kind, kinds).sum())


def test_stationary_single_semantic_slot():
    rng = np.random.default_rng(0)
    eng, flags = make(), []
    for t in range(5000):
        r = eng.ingest(pt(rng), t, t)
        if t >= 100:
            flags.append(bool(r.flagged))
    assert count(eng, lc.SEM) == 1
    assert count(eng, lc.SUPER) == 0  # no spurious step detections
    assert np.mean(flags) <= 0.01


def test_isolated_outlier_pinned_episodic():
    rng = np.random.default_rng(0)
    eng = make()
    res = {}
    for t in range(3000):
        off = 20.0 if t == 1000 else 0.0
        res[t] = eng.ingest(pt(rng, off), t, t)
        if t == 1000 + 9:
            iid = res[1000].item_id
            assert eng.kind[np.flatnonzero(eng.item_id == iid)[0]] == lc.EPI
    assert res[1000].action == "new_candidate"
    idx = np.flatnonzero(eng.item_id == res[1000].item_id)
    assert len(idx) == 1 and eng.kind[idx[0]] == lc.EPI


def test_sporadic_recurrence_never_promoted_and_never_masked():
    rng = np.random.default_rng(0)
    eng = make()
    recur_res = []
    for t in range(500 + 300 * 10):
        recur = t >= 500 and (t - 500) % 300 == 0
        r = eng.ingest(pt(rng, 20.0 if recur else 0.0), t, t)
        assert r.action != "promote" or t < 100
        if recur:
            recur_res.append(r)
    far = [i for i in np.flatnonzero(eng.kind != lc.EMPTY) if eng.mu[i, 0] > 10]
    assert len(far) == 1
    assert eng.kind[far[0]] == lc.EPI and eng.n[far[0]] == 10
    # the 2nd..10th occurrences are absorbed by the stored fault slot, yet still
    # score as far from normal and raise an alarm
    assert all(r.action == "absorb_episodic" for r in recur_res[1:])
    assert all(r.flagged and r.score > eng.tau for r in recur_res)


def test_episodic_promotion_needs_sustained_dwell_not_a_burst():
    rng = np.random.default_rng(0)
    eng = make()
    first = None
    for t in range(1100):
        # t=100: first sighting (expires to episodic); t=600-605: a 6-window
        # burst (two fault events back to back); t=1000-1019: a sustained dwell
        off = 20.0 if t == 100 or 600 <= t < 606 or 1000 <= t < 1020 else 0.0
        r = eng.ingest(pt(rng, off), t, t)
        if t == 100:
            first = r.item_id
        if t in (200, 700):
            assert eng.kind[np.flatnonzero(eng.item_id == first)[0]] == lc.EPI
        if t == 1018:
            assert r.action == "absorb_episodic"
        if t == 1019:
            assert r.action == "promote" and r.item_id == first
    assert eng.kind[np.flatnonzero(eng.item_id == first)[0]] == lc.SEM


def test_alarm_threshold_independent_of_absorption():
    rng = np.random.default_rng(0)
    eng = make(alarm_threshold=1e9)
    for t in range(1100):
        r = eng.ingest(pt(rng, 20.0 if t == 1000 else 0.0), t, t)
        assert not r.flagged
        if t == 1000:
            assert r.action == "new_candidate"  # still stored as novel


def test_persistent_step_promotes_and_supersedes_old():
    rng = np.random.default_rng(0)
    eng = make()
    old_id = promoted = None
    for t in range(5000):
        if t == 1999:
            old_id = int(eng.item_id[np.flatnonzero(eng.kind == lc.SEM)[0]])
        r = eng.ingest(pt(rng, 8.0 if t >= 2000 else 0.0), t, t)
        if r.action == "promote" and t >= 2000 and promoted is None:
            assert t <= 2008
            promoted = r.item_id
        if t == 2000 + 2000 + 200:
            old = np.flatnonzero(eng.item_id == old_id)[0]
            assert eng.kind[old] == lc.SUPER
            succ = np.flatnonzero(eng.item_id == eng.superseded_by[old])[0]
            assert eng.kind[succ] == lc.SEM and eng.mu[succ, 0] > 6
    assert promoted is not None
    assert any(r["type"] == "supersede" for r in eng.lineage)


def test_small_persistent_step_detected_and_superseded():
    # +1.5 on one dim = 3 noise sd: inside the absorption gate, so only a
    # sequential test can see it
    rng = np.random.default_rng(0)
    eng = make()
    old_id, events = None, []
    for t in range(2400):
        if t == 1999:
            old_id = int(eng.item_id[np.flatnonzero(eng.kind == lc.SEM)[0]])
        eng.ingest(pt(rng, 1.5 if t >= 2000 else 0.0), t, t)
        events += [(t, e) for e in eng.drain_events() if e[0] == "supersede"]
    assert len(events) == 1
    t_det, (_, old, new) = events[0]
    assert old == old_id and 2000 <= t_det <= 2030
    assert eng.kind[np.flatnonzero(eng.item_id == old_id)[0]] == lc.SUPER
    j = np.flatnonzero(eng.item_id == new)[0]
    assert eng.kind[j] == lc.SEM and abs(eng.mu[j, 0] - 1.5) < 0.3
    assert any(r["type"] == "supersede" and r["cause"] == "step" for r in eng.lineage)


def test_budget_invariant_with_merge_and_evict_events():
    rng = np.random.default_rng(0)
    eng = make(B_sem=4, B_epi=4)
    t, events = 0, []
    for b in range(100):
        eng.ingest(pt(rng, 1000 + 40 * b), t, t)
        t += 1
        for _ in range(5):
            eng.ingest(pt(rng, 40.0 * b), t, t)
            t += 1
            assert count(eng, lc.CAND, lc.EPI) <= 4
            assert count(eng, lc.SEM, lc.SUPER) <= 4
        events += eng.drain_events()
        assert count(eng, lc.CAND, lc.EPI) <= 4
    kinds = {e[0] for e in events}
    assert {"merge", "evict"} <= kinds


def test_episodic_admission_rejects_weaker_memory_without_eviction():
    eng = make(B_epi=2, B_sem=4)
    for surpr in (300.0, 5.0):
        k = eng._alloc()
        eng.kind[k], eng.n[k], eng.surpr[k] = lc.EPI, 1, surpr
        eng.t_last[k] = 0
        eng.item_id[k] = eng._new_id()
        eng.exemplars[int(eng.item_id[k])] = [0, 0]
    assert not lc.make_episodic_room(eng, 10, incoming_surprisal=2.0)
    assert sorted(eng.surpr[i] for i in np.flatnonzero(eng.kind == lc.EPI)) == [5.0, 300.0]


def test_episodic_admission_evicts_weaker_memory_for_stronger_one():
    eng = make(B_epi=2, B_sem=4)
    for surpr in (300.0, 5.0):
        k = eng._alloc()
        eng.kind[k], eng.n[k], eng.surpr[k] = lc.EPI, 1, surpr
        eng.t_last[k] = 0
        eng.item_id[k] = eng._new_id()
        eng.exemplars[int(eng.item_id[k])] = [0, 0]
    assert lc.make_episodic_room(eng, 10, incoming_surprisal=10.0)
    assert [eng.surpr[i] for i in np.flatnonzero(eng.kind == lc.EPI)] == [300.0]


def test_slow_drift_tracked_without_flags():
    rng = np.random.default_rng(0)
    eng, flags = make(), []
    T = 10_000
    for t in range(T):
        r = eng.ingest(pt(rng, 6.0 * t / T), t, t)
        if t >= 100:
            flags.append(bool(r.flagged))
    sem = np.flatnonzero(eng.kind == lc.SEM)
    assert len(sem) == 1 and count(eng, lc.SUPER) == 0
    assert np.mean(flags) <= 0.02
    assert sum(r["type"] == "checkpoint" for r in eng.lineage) >= 3


def _wander(rng, T, rho=0.9, tau=1.0):
    """One state whose mean wanders (AR(1), sd tau*SIG) under white noise SIG."""
    m, out = np.zeros(D), []
    for _ in range(T):
        m = rho * m + rng.normal(0, tau * SIG * np.sqrt(1 - rho ** 2), D)
        out.append(m + rng.normal(0, SIG, D))
    return np.array(out)


def test_calibrated_step_detector_ignores_correlated_noise():
    for seed in range(3):
        rng = np.random.default_rng(seed)
        infl = ewma_inflation(_wander(rng, 4000), EngramConfig.step_lambda, None)
        z = _wander(rng, 2000)
        steps = {}
        for name, f in (("raw", None), ("calibrated", infl)):
            eng = Engram(EngramConfig(), D, SIG * np.ones(D), 1536, step_inflation=f)
            steps[name] = sum(eng.ingest(z[t], t, t).action == "step_change" for t in range(2000))
        assert steps["raw"] >= 20  # the independence assumption fails without calibration
        assert steps["calibrated"] == 0


def test_calibrated_engine_still_follows_persistent_shift():
    for seed in range(3):
        rng = np.random.default_rng(seed)
        infl = ewma_inflation(_wander(rng, 4000), EngramConfig.step_lambda, None)
        z = _wander(rng, 2300)
        z[2000:, :4] += 5.0 * SIG
        eng = Engram(EngramConfig(), D, SIG * np.ones(D), 1536, step_inflation=infl)
        for t in range(2300):
            eng.ingest(z[t], t, t)
        target = np.r_[np.full(4, 5.0 * SIG), np.zeros(D - 4)]
        top = eng.recall(target, 2299, 1, "current")[0]
        k = int(np.flatnonzero(eng.item_id == top)[0])
        assert eng.kind[k] == lc.SEM
        assert np.abs(eng.mu[k, :4] - 5.0 * SIG).max() < 1.5 * SIG


def test_health_grows_with_drift_the_memory_accepts():
    rng = np.random.default_rng(0)
    eng = make()
    for t in range(500):
        eng.ingest(pt(rng), t, t)
    eng.set_reference(*eng.reference())
    T, scores, health = 10_000, [], []
    for t in range(500, 500 + T):
        r = eng.ingest(pt(rng, 6.0 * (t - 500) / T), t, t)
        scores.append(r.score)
        health.append(r.health)
    # the adaptive memory follows the drift, so novelty stays normal ...
    assert np.median(scores[-500:]) < eng.tau
    # ... while the distance to the frozen healthy reference keeps growing
    assert np.median(health[:500]) < eng.tau < np.median(health[-500:])


def test_recall_current_prefers_live_over_superseded():
    eng = make()
    for kind, off in ((lc.SEM, 0.0), (lc.SUPER, 0.0)):
        k = eng._alloc()
        eng.kind[k], eng.n[k], eng.t_last[k] = kind, 50, 100
        eng.mu[k] = np.zeros(D)
        eng.mu[k, 0] = off
        eng.item_id[k] = eng._new_id()
    q = np.zeros(D)
    assert eng.recall(q, 200, 5, "current") == [0]
    assert set(eng.recall(q, 200, 5, "history")) == {0, 1}


def test_unique_event_outlives_recurring_novelties_when_budget_overflows():
    # A unique extreme event and a family of mutually similar novelties are all
    # far from normal; retention priority must prefer the unique one.
    rng = np.random.default_rng(0)
    eng = make(B_epi=3)
    unique = None
    for t in range(700):
        z = pt(rng)
        if t == 100:
            z[0] += 60.0
        elif t >= 200 and t % 50 == 0:
            z[0] += 20.0
            z[1] += 6.0 * ((t - 200) // 50)  # each member 12 sd from the previous one
        r = eng.ingest(z, t, t)
        if t == 100:
            unique = r.item_id
        assert count(eng, lc.CAND, lc.EPI) <= 3
    assert unique in set(eng.items().ids.tolist())
