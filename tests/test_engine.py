import numpy as np

from uengram import lifecycle as lc
from uengram.engine import Engram, EngramConfig

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


def test_sporadic_recurrence_never_promoted():
    rng = np.random.default_rng(0)
    eng = make()
    for t in range(500 + 300 * 10):
        recur = t >= 500 and (t - 500) % 300 == 0
        r = eng.ingest(pt(rng, 20.0 if recur else 0.0), t, t)
        assert r.action != "promote" or t < 100
    far = [i for i in np.flatnonzero(eng.kind != lc.EMPTY) if eng.mu[i, 0] > 10]
    assert len(far) == 1
    assert eng.kind[far[0]] == lc.EPI and eng.n[far[0]] == 10


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


def test_episodic_eviction_prefers_low_surprisal():
    eng = make(B_epi=2, B_sem=4)
    for surpr in (300.0, 5.0):
        k = eng._alloc()
        eng.kind[k], eng.n[k], eng.surpr[k] = lc.EPI, 1, surpr
        eng.t_last[k] = 0
        eng.item_id[k] = eng._new_id()
        eng.exemplars[int(eng.item_id[k])] = [0, 0]
    lc.make_episodic_room(eng, 10)
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
