import numpy as np

from uengram import lifecycle as lc
from uengram.engine import Engram, EngramConfig

D = 4


def make(**kw):
    return Engram(EngramConfig(**kw), D, np.ones(D), 64)


def add_sem(eng, level, key, n=50):
    k = eng._alloc()
    eng.kind[k], eng.n[k], eng.mode_key[k] = lc.SEM, n, key
    eng.mu[k] = np.full(D, level)
    eng.S[k] = np.ones(D)  # per-slot variance, independent of n
    eng.t_last[k] = eng.t_created[k] = 0
    eng.item_id[k] = eng._new_id()
    return int(eng.item_id[k])


def kinds(eng, kind):
    return int((eng.kind == kind).sum())


def test_identical_states_under_different_keys_are_not_merged_by_budget_pressure():
    eng = make(B_sem=2, B_epi=2)
    add_sem(eng, 0.0, key=1)
    add_sem(eng, 0.01, key=2)  # the cheapest Ward pair, but a different identity
    assert not lc.make_semantic_room(eng, 10)
    assert kinds(eng, lc.SEM) == 2


def test_same_key_states_still_merge_with_exact_pooled_counts():
    eng = make(B_sem=2, B_epi=2)
    a = add_sem(eng, 0.0, key=1, n=30)
    b = add_sem(eng, 1.0, key=1, n=10)
    assert lc.make_semantic_room(eng, 10)
    (k,) = np.flatnonzero(eng.kind == lc.SEM)
    assert eng.n[k] == 40 and eng.item_id[k] == a  # larger n survives
    np.testing.assert_allclose(eng.mu[k], np.full(D, 10 / 40))  # (30*0 + 10*1) / 40
    assert b not in set(eng.item_id.tolist())


def test_consolidation_merges_duplicates_only_within_a_key():
    eng = make(B_sem=4, B_epi=2)
    add_sem(eng, 0.0, key=1)
    add_sem(eng, 0.0, key=2)  # an exact duplicate of the first, under another key
    add_sem(eng, 0.0, key=1)  # an exact duplicate of the first, same key
    lc.consolidate(eng, 10)
    keys = sorted(eng.mode_key[eng.kind == lc.SEM].tolist())
    assert keys == [1, 2]  # the two key-1 slots merged; key 2 untouched


def test_a_window_is_only_absorbed_by_a_slot_of_its_own_key():
    eng = make()
    z = np.zeros(D)
    first = eng.ingest(z, 0, 0, mode_key=7)
    same = eng.ingest(z, 1, 1, mode_key=7)
    other = eng.ingest(z, 2, 2, mode_key=8)  # identical vector, different identity
    assert first.action == "new_candidate" and same.item_id == first.item_id
    assert other.action == "new_candidate" and other.item_id != first.item_id
    assert sorted(eng.mode_key[eng.kind != lc.EMPTY].tolist()) == [7, 8]


def test_default_key_zero_behaves_as_before():
    eng = make()
    z = np.zeros(D)
    assert eng.ingest(z, 0, 0).item_id == eng.ingest(z, 1, 1).item_id


def test_step_successor_inherits_the_key_and_clear_resets_it():
    eng = make(B_sem=4, B_epi=2)
    add_sem(eng, 0.0, key=5)
    k = int(np.flatnonzero(eng.kind == lc.SEM)[0])
    new_iid = lc.split_on_step(eng, k, np.full(D, 9.0), 10)
    j = int(np.flatnonzero(eng.item_id == new_iid)[0])
    assert eng.mode_key[j] == 5
    lc.evict_slot(eng, j, 11, "evict_test")
    assert eng.mode_key[j] == 0


def test_dormant_supersession_only_picks_a_successor_of_the_same_key():
    eng = make(B_sem=4, B_epi=2, t_dormant=100)
    old = add_sem(eng, 0.0, key=1)
    other_key = add_sem(eng, 0.0, key=2)
    other_key_slot = int(np.flatnonzero(eng.item_id == other_key)[0])
    eng.t_created[other_key_slot] = 50  # younger than old's last hit, but another identity
    lc.consolidate(eng, 1000)
    old_slot = int(np.flatnonzero(eng.item_id == old)[0])
    assert eng.kind[old_slot] == lc.SEM  # no same-key successor: not superseded


def test_a_protected_slot_survives_a_duplicate_merge_as_the_survivor():
    eng = make(B_sem=4, B_epi=2)
    big = add_sem(eng, 0.0, key=1, n=90)
    small = add_sem(eng, 0.0, key=1, n=10)
    eng.protect(small)
    lc.consolidate(eng, 10)
    (k,) = np.flatnonzero(eng.kind == lc.SEM)
    assert eng.item_id[k] == small  # protection overrides "larger n survives"
    assert eng.n[k] == 100 and big not in set(eng.item_id.tolist())


def test_two_protected_slots_are_never_merged_with_each_other():
    eng = make(B_sem=4, B_epi=2)
    a = add_sem(eng, 0.0, key=1)
    b = add_sem(eng, 0.0, key=1)
    eng.protect(a)
    eng.protect(b)
    lc.consolidate(eng, 10)
    assert kinds(eng, lc.SEM) == 2
