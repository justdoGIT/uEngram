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
    alpha_gate: float = 1e-4
    beta_min: float = 0.01
    kappa0: float = 10.0
    m_promote: int = 5  # candidate probation: M hits within N steps of creation
    n_promote: int = 8
    m_reconfirm: int = 20  # episodic -> semantic: a sustained dwell of M hits within N steps
    n_reconfirm: int = 30
    consolidate_every: int = 200
    db_merge: float = 0.5
    t_dormant: int = 2000
    theta_drift: float = 3.0
    half_life: float = 2000.0
    half_life_epi: float = 20000.0
    episodic_admission_ratio: float = 1.0
    flush_every: int = 50
    use_slot_var: bool = True
    use_promotion: bool = True
    # alarm threshold on the normal-state score; None -> the chi-square gate tau.
    # The harness sets it by conformal calibration on a normal commissioning run.
    alarm_threshold: float | None = None
    # multivariate EWMA step detector on semantic-slot innovations
    step_lambda: float = 0.1
    step_alpha: float = 1e-6
    use_step_detect: bool = True
    B_ref: int = 32  # max states kept in a frozen healthy reference


class Engram:
    in_ref_space = True

    def __init__(self, cfg: EngramConfig, d: int, sigma0: np.ndarray,
                 raw_window_bytes: int, name: str = "uengram",
                 step_inflation: np.ndarray | None = None):
        self.cfg, self.d, self.name = cfg, d, name
        self.raw_window_bytes = raw_window_bytes
        self.sigma0 = np.asarray(sigma0, dtype=np.float64)
        self.sigma0_sq = self.sigma0 ** 2
        # The step detector's MEWMA averages about (2-lambda)/lambda innovations
        # and its threshold assumes them independent. Slow within-state
        # variation correlates them and inflates the MEWMA variance by a
        # per-dimension factor f (stats.ewma_inflation on training latents), so
        # innovations enter it scaled by 1/sqrt(f), never amplified.
        self.step_scale = (np.ones(d) if step_inflation is None else
                           np.sqrt(np.maximum(np.asarray(step_inflation, dtype=np.float64), 1.0)))
        self.tau = chi2_isf(cfg.alpha_gate, d)
        self.alarm_tau = self.tau if cfg.alarm_threshold is None else float(cfg.alarm_threshold)
        self.step_h = chi2_isf(cfg.step_alpha, d)
        self.step_c = (2.0 - cfg.step_lambda) / cfg.step_lambda
        self.step_min_n = int(np.ceil(2.0 / cfg.step_lambda))
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
        self.ewma_e = np.zeros((cap, d))  # step-detector state per slot
        self.ref_mu, self.ref_var = np.zeros((0, d)), np.zeros((0, d))  # frozen healthy reference
        ring = max(cfg.m_promote, cfg.m_reconfirm)
        self.hits = np.full((cap, ring), lc.NO_HIT, dtype=np.int64)  # last hit times, oldest first
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

    # -- healthy reference --------------------------------------------------
    def reference(self):
        """The current normal states (largest first, at most B_ref) as
        (mean, predictive variance): a healthy reference to freeze."""
        sem = np.flatnonzero(self.kind == SEM)
        sem = sem[np.argsort(-self.n[sem], kind="stable")][: self.cfg.B_ref]
        return self.mu[sem].copy(), np.array(self._pred_var(sem))

    def set_reference(self, mu, var) -> None:
        """Freeze a healthy reference. Unlike the slots it never adapts, so the
        health score keeps growing with wear the memory has learned to accept."""
        self.ref_mu = np.array(mu, dtype=np.float64).reshape(-1, self.d)
        self.ref_var = np.array(var, dtype=np.float64).reshape(-1, self.d)
        self.ledger.write_now(-1, len(self.ref_mu) * 2 * self.d * 4)

    def health(self, z) -> np.ndarray:
        """Health score of each row of z: min Mahalanobis D^2 to the frozen
        healthy reference; NaN without a reference."""
        z = np.atleast_2d(np.asarray(z, dtype=np.float64))
        if not len(self.ref_mu):
            return np.full(len(z), np.nan)
        return np.min(np.sum((z[:, None, :] - self.ref_mu[None]) ** 2 / self.ref_var[None], axis=2), axis=1)

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
        score, rarity, chosen = float("nan"), float("nan"), -1
        if len(gate):
            var = self._pred_var(gate)
            d2 = np.sum((z - self.mu[gate]) ** 2 / var, axis=1)
            # Novelty is measured against normal (semantic) states only, so a
            # recurring fault stored in an episodic slot is not masked.
            ref = (self.kind[gate] == SEM) if cfg.use_promotion else np.ones(len(gate), bool)
            if ref.any():
                score = float(d2[ref].min())
            # Rarity among everything stored ranks episodic memories for
            # retention: a pattern that keeps recurring is less precious than a
            # unique one, even when both are far from normal.
            rarity = float(d2.min())
            inside = d2 <= self.tau
            if inside.any():
                # Habit memory has priority: a point inside any semantic gate is
                # routine, so episodic/candidate slots cannot capture it and
                # drift into the nominal cluster.
                sem_in = inside & (self.kind[gate] == SEM)
                pool = sem_in if sem_in.any() else inside
                nll = np.where(pool, d2 + np.sum(np.log(var), axis=1), np.inf)
                chosen = int(gate[np.argmin(nll)])
        alarm = bool(score > self.alarm_tau)  # NaN (no normal state yet) -> no alarm

        if chosen >= 0:
            res = self._absorb(chosen, z, t, raw_idx, score, alarm)
        else:
            res = self._new_slot(z, t, raw_idx, score, alarm, rarity)
        res.health = float(self.health(z)[0])

        stale = np.flatnonzero((self.kind == CAND) & (t - self.t_created > cfg.n_promote))
        for i in stale:
            self.kind[i] = EPI
            self.ledger.mark_dirty(int(self.item_id[i]), self.slot_bytes)
        if (t + 1) % cfg.consolidate_every == 0:
            lc.consolidate(self, t)
        self.ledger.tick(t)
        return res

    def _absorb(self, k: int, z, t: int, raw_idx: int, score: float, alarm: bool) -> IngestResult:
        cfg = self.cfg
        iid = int(self.item_id[k])
        n = int(self.n[k])
        sem = self.kind[k] == SEM
        alpha = max(1.0 / (n + 1), cfg.beta_min) if sem else 1.0 / (n + 1)
        if sem:
            e = (z - self.mu[k]) / np.sqrt(self._pred_var(np.array([k]))[0])
        self.mu[k], self.S[k] = ew_update(self.mu[k], self.S[k], z, alpha)
        self.n[k] = n + 1
        self.t_last[k] = t
        self.ledger.mark_dirty(iid, self.slot_bytes)
        if sem:
            if cfg.use_step_detect and self._step_detected(k, e):
                new_iid = lc.split_on_step(self, k, z, t)
                if new_iid is not None:
                    return IngestResult(new_iid, score, alarm, "step_change")
            return IngestResult(iid, score, alarm, "absorb_semantic")
        self.exemplars[iid][1] = raw_idx
        self.ledger.write_now(iid, self.raw_window_bytes)
        self.hits[k, :-1] = self.hits[k, 1:]
        self.hits[k, -1] = t
        # Candidates confirm a new state by M hits within N steps of creation.
        # An episodic slot is promoted only by a sustained dwell: operating
        # states persist, while bursts of a recurring fault stay short.
        if self.kind[k] == CAND:
            m, n_win = cfg.m_promote, cfg.n_promote
        else:
            m, n_win = cfg.m_reconfirm, cfg.n_reconfirm
        if (cfg.use_promotion and t - self.hits[k, -m] <= n_win
                and lc.make_semantic_room(self, t)):
            self.kind[k] = SEM
            self.ckpt_mu[k] = self.mu[k]
            self.ewma_e[k] = 0.0
            self.exemplars.pop(iid, None)
            self.ledger.write_now(iid, self.slot_bytes)
            return IngestResult(iid, score, alarm, "promote")
        action = "absorb_candidate" if self.kind[k] == CAND else "absorb_episodic"
        return IngestResult(iid, score, alarm, action)

    def _step_detected(self, k: int, e: np.ndarray) -> bool:
        """MEWMA on standardised innovations e / step_scale: Z <- lam e + (1-lam) Z;
        alarm when (2-lam)/lam * |Z|^2 exceeds the chi2_d quantile at step_alpha."""
        lam = self.cfg.step_lambda
        self.ewma_e[k] = lam * e / self.step_scale + (1.0 - lam) * self.ewma_e[k]
        return bool(self.n[k] >= self.step_min_n
                    and self.step_c * float(self.ewma_e[k] @ self.ewma_e[k]) > self.step_h)

    def _new_slot(self, z, t: int, raw_idx: int, score: float, alarm: bool,
                  rarity: float) -> IngestResult:
        incoming_surprisal = 0.0 if np.isnan(rarity) else surprisal(rarity, self.d)
        if not lc.make_episodic_room(self, t, incoming_surprisal):
            return IngestResult(-1, score, alarm, "discard")
        k = self._alloc()
        iid = self._new_id()
        lc.init_slot(self, k, iid, z, t, CAND if self.cfg.use_promotion else EPI)
        self.surpr[k] = incoming_surprisal
        self.exemplars[iid] = [raw_idx, raw_idx]
        self.ledger.write_now(iid, self.slot_bytes + self.raw_window_bytes)
        return IngestResult(iid, score, alarm, "new_candidate")

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
        return ItemsView(self.item_id[idx].copy(), self.mu[idx].copy(),
                         np.array(self._gate_var(idx)), self.n[idx].astype(np.float64),
                         self.t_last[idx].copy())

    def drain_events(self) -> list[tuple]:
        ev, self.events = self.events, []
        return ev

    def index_bytes(self) -> int:
        return int(np.sum(self.kind != EMPTY)) * self.slot_bytes + len(self.ref_mu) * 2 * self.d * 4

    def pinned_raw_bytes(self) -> int:
        return sum(len(set(v)) for v in self.exemplars.values()) * self.raw_window_bytes
