"""Slot lifecycle: eviction, merging, consolidation (acts on an Engram)."""
from __future__ import annotations

import numpy as np

from .stats import bhattacharyya_diag, bhattacharyya_pairwise, merge_moments, ward_cost_pairwise

EMPTY, CAND, EPI, SEM, SUPER = 0, 1, 2, 3, 4
MAX_DUP_MERGES = 20
NO_HIT = -(1 << 62)  # sentinel in the per-slot hit-time ring


def init_slot(eng, k: int, iid: int, z, t: int, kind: int) -> None:
    """Fresh slot at index k holding the single point z."""
    eng.mu[k], eng.S[k], eng.n[k] = z, 0.0, 1
    eng.kind[k] = kind
    eng.t_created[k] = eng.t_last[k] = t
    eng.surpr[k] = 0.0
    eng.item_id[k] = iid
    eng.superseded_by[k] = -1
    eng.ckpt_mu[k] = z
    eng.ewma_e[k] = 0.0
    eng.hits[k] = NO_HIT
    eng.hits[k, -1] = t


def _supersede(eng, old: int, new_iid: int, t: int, cause: str) -> None:
    old_iid = int(eng.item_id[old])
    eng.kind[old] = SUPER
    eng.superseded_by[old] = new_iid
    eng.ewma_e[old] = 0.0
    eng.lineage.append({"type": "supersede", "old": old_iid, "new": new_iid, "t": t, "cause": cause})
    eng.events.append(("supersede", old_iid, new_iid))
    eng.ledger.mark_dirty(old_iid, eng.slot_bytes)


def split_on_step(eng, k: int, z, t: int) -> int | None:
    """A persistent shift was detected in semantic slot k: start a new semantic
    state at z (keeping k's variance as its prior) and supersede k.
    Returns the new item id, or None if k was merged away while making room."""
    old_iid = int(eng.item_id[k])
    if not make_semantic_room(eng, t):
        return None
    idx = np.flatnonzero(eng.item_id == old_iid)
    if len(idx) == 0 or eng.kind[idx[0]] != SEM:
        return None
    k = int(idx[0])
    j = eng._alloc()
    new_iid = eng._new_id()
    init_slot(eng, j, new_iid, z, t, SEM)
    eng.S[j] = eng.S[k]
    eng.ledger.write_now(new_iid, eng.slot_bytes)
    _supersede(eng, k, new_iid, t, "step")
    return new_iid


def merge_slots(eng, a: int, b: int, t: int) -> None:
    """Merge slot indices a,b; the one with larger n survives (lower id on ties)."""
    if (eng.n[a], -eng.item_id[a]) >= (eng.n[b], -eng.item_id[b]):
        dst, src = a, b
    else:
        dst, src = b, a
    n, mu, var = merge_moments(eng.n[dst], eng.mu[dst], eng.S[dst], eng.n[src], eng.mu[src], eng.S[src])
    eng.mu[dst], eng.S[dst], eng.n[dst] = mu, var, n
    eng.ewma_e[dst] = 0.0
    eng.t_created[dst] = min(eng.t_created[dst], eng.t_created[src])
    eng.t_last[dst] = max(eng.t_last[dst], eng.t_last[src])
    sid, did = int(eng.item_id[src]), int(eng.item_id[dst])
    eng.superseded_by[eng.superseded_by == sid] = did
    eng.ledger.write_now(did, eng.slot_bytes)
    eng.ledger.delete(sid)
    eng.events.append(("merge", sid, did))
    eng.lineage.append({"type": "merge", "src": sid, "dst": did, "t": t})
    _clear(eng, src)


def _clear(eng, idx: int) -> None:
    eng.exemplars.pop(int(eng.item_id[idx]), None)
    eng.kind[idx] = EMPTY
    eng.item_id[idx] = -1
    eng.superseded_by[idx] = -1


def evict_slot(eng, idx: int, t: int, kind_label: str) -> None:
    iid = int(eng.item_id[idx])
    eng.lineage.append({"type": kind_label, "item_id": iid, "t": t})
    eng.ledger.delete(iid)
    eng.events.append(("evict", iid))
    eng.superseded_by[eng.superseded_by == iid] = -1
    _clear(eng, idx)


def make_semantic_room(eng, t: int) -> bool:
    """Ensure the semantic group {3,4} has a free slot. False if impossible."""
    group = np.flatnonzero((eng.kind == SEM) | (eng.kind == SUPER))
    if len(group) < eng.cfg.B_sem:
        return True
    sup = np.flatnonzero(eng.kind == SUPER)
    if len(sup):
        victim = sup[np.argmin(eng.t_last[sup])]
        evict_slot(eng, int(victim), t, "evict_superseded")
        return True
    act = np.flatnonzero(eng.kind == SEM)
    if len(act) < 2:
        return False
    cost = ward_cost_pairwise(eng.n[act], eng.mu[act], eng.sigma0_sq)
    i, j = np.unravel_index(np.argmin(cost), cost.shape)
    merge_slots(eng, int(act[i]), int(act[j]), t)
    return True


def make_episodic_room(eng, t: int) -> None:
    """Ensure the episodic group {1,2} has a free slot."""
    group = np.flatnonzero((eng.kind == CAND) | (eng.kind == EPI))
    if len(group) < eng.cfg.B_epi:
        return
    epi = np.flatnonzero(eng.kind == EPI)
    if len(epi):
        age = t - eng.t_last[epi]
        imp = eng.surpr[epi] * (1.0 + np.log(eng.n[epi])) * 2.0 ** (-age / eng.cfg.half_life_epi)
        victim = epi[np.argmin(imp)]
    else:
        cand = np.flatnonzero(eng.kind == CAND)
        victim = cand[np.argmin(eng.t_created[cand])]
    evict_slot(eng, int(victim), t, "evict_episodic")


def consolidate(eng, t: int) -> None:
    cfg = eng.cfg
    # (a) duplicate merge
    for _ in range(MAX_DUP_MERGES):
        act = np.flatnonzero(eng.kind == SEM)
        if len(act) < 2:
            break
        db = bhattacharyya_pairwise(eng.mu[act], eng._gate_var(act))
        i, j = np.unravel_index(np.argmin(db), db.shape)
        if db[i, j] > cfg.db_merge:
            break
        merge_slots(eng, int(act[i]), int(act[j]), t)

    act = np.flatnonzero(eng.kind == SEM)
    # (b) drift checkpoints
    if len(act):
        drift = np.linalg.norm((eng.mu[act] - eng.ckpt_mu[act]) / eng.sigma0, axis=1)
        for idx in act[drift >= cfg.theta_drift]:
            idx = int(idx)
            eng.lineage.append({
                "type": "checkpoint", "item_id": int(eng.item_id[idx]), "t": t,
                "mu": eng.mu[idx].copy(), "var": eng._gate_var(np.array([idx]))[0],
                "n": int(eng.n[idx]),
            })
            eng.ledger.write_now(int(eng.item_id[idx]), eng.slot_bytes)
            eng.ckpt_mu[idx] = eng.mu[idx]

    # (c) supersession of dormant slots that have a younger successor
    act = np.flatnonzero(eng.kind == SEM)
    for o in act:
        o = int(o)
        if t - eng.t_last[o] < cfg.t_dormant:
            continue
        succ = act[(eng.kind[act] == SEM) & (eng.t_created[act] > eng.t_last[o]) & (act != o)]
        if len(succ) == 0:
            continue
        gv_o = eng._gate_var(np.array([o]))[0]
        gv_s = eng._gate_var(succ)
        dist = [bhattacharyya_diag(eng.mu[o], gv_o, eng.mu[s], gv_s[k]) for k, s in enumerate(succ)]
        best = int(succ[int(np.argmin(dist))])
        _supersede(eng, o, int(eng.item_id[best]), t, "dormant")
