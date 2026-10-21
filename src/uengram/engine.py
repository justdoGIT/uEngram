"""uEngram: fixed-budget statistical memory over a whitened latent space."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from . import lifecycle as lc
from .ledger import WriteLedger
from .stats import chi2_isf, ew_update, shrunk_var, surprisal
from .store import IngestResult, ItemsView

EMPTY, CAND, EPI, SEM, SUPER = lc.EMPTY, lc.CAND, lc.EPI, lc.SEM, lc.SUPER


@dataclass
class EngramConfig:
    B_sem: int = 192
    B_epi: int = 64
    alpha_gate: float = 1e-3
    beta_min: float = 0.01
    kappa0: float = 10.0
    m_promote: int = 5
    n_promote: int = 8
    consolidate_every: int = 200
    db_merge: float = 0.5
    t_dormant: int = 2000
    theta_drift: float = 3.0
    half_life: float = 2000.0
    half_life_epi: float = 20000.0
    flush_every: int = 50
    use_slot_var: bool = True
    use_promotion: bool = True


class Engram:
    in_ref_space = True

    def __init__(self, cfg: EngramConfig, d: int, sigma0: np.ndarray,
                 raw_window_bytes: int, name: str = "uengram"):
        self.cfg, self.d, self.name = cfg, d, name
        self.raw_window_bytes = raw_window_bytes
        self.sigma0 = np.asarray(sigma0, dtype=np.float64)
        self.sigma0_sq = self.sigma0 ** 2
        self.tau = chi2_isf(cfg.alpha_gate, d)
        self.n_eff_cap = (2.0 - cfg.beta_min) / cfg.beta_min if cfg.beta_min > 0 else np.inf
        self.slot_bytes = 2 * d * 4 + 32
        cap = cfg.B_sem + cfg.B_epi
        self.mu = np.zeros((cap, d))
        self.S = np.zeros((cap, d))
        self.ckpt_mu = np.zeros((cap, d))
        self.n = np.zeros(cap, dtype=np.int64)
        self.kind = np.zeros(cap, dtype=np.int8)
        self.t_created = np.zeros(cap, dtype=np.int64)
        self.t_last = np.zeros(cap, dtype=np.int64)
        self.surpr = np.zeros(cap)
        self.item_id = np.full(cap, -1, dtype=np.int64)
        self.superseded_by = np.full(cap, -1, dtype=np.int64)
        self.exemplars: dict[int, list[int]] = {}
        self.lineage: deque = deque(maxlen=256)
        self.events: list[tuple] = []
        self.ledger = WriteLedger(cfg.flush_every)
        self._next_id = 0

    # -- helpers ---------------------------------------------------------
    def _gate_var(self, idx: np.ndarray) -> np.ndarray:
        if not self.cfg.use_slot_var:
            return np.broadcast_to(self.sigma0_sq, (len(idx), self.d))
        n = self.n[idx].astype(np.float64)
        n_eff = np.where(self.kind[idx] == SEM, np.minimum(n, self.n_eff_cap), n)
        return shrunk_var(self.S[idx], n_eff[:, None], self.sigma0_sq, self.cfg.kappa0)

    def _pred_var(self, idx: np.ndarray) -> np.ndarray:
        """Predictive variance of a new point: slot variance inflated by the
        uncertainty of the slot mean, (1 + 1/n_eff)."""
        var = self._gate_var(idx)
        if not self.cfg.use_slot_var:
            return var
        n = self.n[idx].astype(np.float64)
        n_eff = np.where(self.kind[idx] == SEM, np.minimum(n, self.n_eff_cap), n)
        return var * (1.0 + 1.0 / n_eff)[:, None]

    def _alloc(self) -> int:
        return int(np.flatnonzero(self.kind == EMPTY)[0])

    def _new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    # -- ingest ----------------------------------------------------------
    def ingest(self, z, t: int, raw_idx: int, **_) -> IngestResult:
        cfg = self.cfg
        z = np.asarray(z, dtype=np.float64)
        gate = np.flatnonzero((self.kind == CAND) | (self.kind == EPI) | (self.kind == SEM))
        score, chosen = float("nan"), -1
        if len(gate):
            var = self._pred_var(gate)
            d2 = np.sum((z - self.mu[gate]) ** 2 / var, axis=1)
            score = float(d2.min())
            inside = d2 <= self.tau
            if inside.any():
                # Habit memory has priority: a point inside any semantic gate is
                # routine, so episodic/candidate slots cannot capture it and
                # drift into the nominal cluster.
                sem_in = inside & (self.kind[gate] == SEM)
                pool = sem_in if sem_in.any() else inside
                nll = np.where(pool, d2 + np.sum(np.log(var), axis=1), np.inf)
                chosen = int(gate[np.argmin(nll)])

        if chosen >= 0:
            res = self._absorb(chosen, z, t, raw_idx, score)
        else:
            res = self._new_slot(z, t, raw_idx, score)

        stale = np.flatnonzero((self.kind == CAND) & (t - self.t_created > cfg.n_promote))
        for i in stale:
            self.kind[i] = EPI
            self.ledger.mark_dirty(int(self.item_id[i]), self.slot_bytes)
        if (t + 1) % cfg.consolidate_every == 0:
            lc.consolidate(self, t)
        self.ledger.tick(t)
        return res

    def _absorb(self, k: int, z, t: int, raw_idx: int, score: float) -> IngestResult:
        cfg = self.cfg
        iid = int(self.item_id[k])
        n = int(self.n[k])
        alpha = max(1.0 / (n + 1), cfg.beta_min) if self.kind[k] == SEM else 1.0 / (n + 1)
        self.mu[k], self.S[k] = ew_update(self.mu[k], self.S[k], z, alpha)
        self.n[k] = n + 1
        self.t_last[k] = t
        self.ledger.mark_dirty(iid, self.slot_bytes)
        if self.kind[k] == SEM:
            return IngestResult(iid, score, False, "absorb_semantic")
        self.exemplars[iid][1] = raw_idx
        self.ledger.write_now(iid, self.raw_window_bytes)
        if (self.kind[k] == CAND and cfg.use_promotion and self.n[k] >= cfg.m_promote
                and t - self.t_created[k] <= cfg.n_promote and lc.make_semantic_room(self, t)):
            self.kind[k] = SEM
            self.ckpt_mu[k] = self.mu[k]
            self.exemplars.pop(iid, None)
            self.ledger.write_now(iid, self.slot_bytes)
            return IngestResult(iid, score, True, "promote")
        action = "absorb_candidate" if self.kind[k] == CAND else "absorb_episodic"
        return IngestResult(iid, score, True, action)

    def _new_slot(self, z, t: int, raw_idx: int, score: float) -> IngestResult:
        lc.make_episodic_room(self, t)
        k = self._alloc()
        iid = self._new_id()
        self.mu[k], self.S[k], self.n[k] = z, 0.0, 1
        self.kind[k] = CAND if self.cfg.use_promotion else EPI
        self.t_created[k] = self.t_last[k] = t
        self.surpr[k] = 0.0 if np.isnan(score) else surprisal(score, self.d)
        self.item_id[k] = iid
        self.superseded_by[k] = -1
        self.ckpt_mu[k] = z
        self.exemplars[iid] = [raw_idx, raw_idx]
        self.ledger.write_now(iid, self.slot_bytes + self.raw_window_bytes)
        return IngestResult(iid, score, True, "new_candidate")

    # -- retrieval -------------------------------------------------------
    def recall(self, q, t_now: int, k: int, mode: str) -> list[int]:
        kinds = (CAND, EPI, SEM) if mode == "current" else (CAND, EPI, SEM, SUPER)
        idx = np.flatnonzero(np.isin(self.kind, kinds))
        if len(idx) == 0:
            return []
        var = self._gate_var(idx)
        d2 = np.sum((np.asarray(q, dtype=np.float64) - self.mu[idx]) ** 2 / var, axis=1)
        score = -0.5 * (d2 + np.sum(np.log(var), axis=1))
        if mode == "current":
            n = self.n[idx].astype(np.float64)
            sem = self.kind[idx] == SEM
            prior = np.where(
                sem,
                np.log(n) - np.log(2.0) * (t_now - self.t_last[idx]) / self.cfg.half_life,
                np.log(n) + np.log1p(self.surpr[idx]),
            )
            score = score + prior
        order = np.lexsort((self.item_id[idx], -score))[:k]
        return [int(self.item_id[idx[i]]) for i in order]

    # -- views / accounting ---------------------------------------------
    def items(self) -> ItemsView:
        idx = np.flatnonzero(self.kind != EMPTY)
        if len(idx) == 0:
            z = np.zeros((0, self.d))
            return ItemsView(np.zeros(0, np.int64), z, z, np.zeros(0), np.zeros(0, np.int64))
        return ItemsView(self.item_id[idx].copy(), self.mu[idx].copy(), np.array(self._gate_var(idx)),
                         self.n[idx].astype(np.float64), self.t_last[idx].copy())

    def drain_events(self) -> list[tuple]:
        ev, self.events = self.events, []
        return ev

    def index_bytes(self) -> int:
        return int(np.sum(self.kind != EMPTY)) * self.slot_bytes

    def pinned_raw_bytes(self) -> int:
        return sum(len(set(v)) for v in self.exemplars.values()) * self.raw_window_bytes
