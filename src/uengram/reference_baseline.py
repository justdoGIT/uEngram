# Port of EdgeMem0Engine from docs/EdgeAIMemoryArchitecture.md; logic unchanged.
"""The architecture doc's engine, behind the common store protocol.

Only interface changes: input is L2-normalised internally (the doc expects unit
vectors), flash accounting goes through the shared WriteLedger, items carry
ids, and the severity input is either constant (label-free) or supplied by the
harness from ground truth (oracle; not deployable).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ledger import WriteLedger
from .store import IngestResult, ItemsView


@dataclass
class _Slot:
    centroid: np.ndarray
    timestamp: int
    mass: int
    severity: float
    is_episodic: bool
    raw_pointer_id: int
    item_id: int


class ReferenceBaseline:
    in_ref_space = False  # centroids live on the unit sphere, not the whitened space

    def __init__(self, max_slots, d, raw_window_bytes, sim_merge_thresh=0.90,
                 ewma_beta=0.05, severity_mode="labelfree", flush_every=50):
        assert severity_mode in ("labelfree", "oracle")
        self.name = f"reference_{severity_mode}"
        self.max_slots = max_slots
        self.d = d
        self.raw_window_bytes = raw_window_bytes
        self.sim_merge_thresh = sim_merge_thresh
        self.ewma_beta = ewma_beta
        self.severity_mode = severity_mode
        self.memories: list[_Slot] = []
        self.ledger = WriteLedger(flush_every)
        self.slot_bytes = d * 4 + 32
        self._events: list[tuple] = []
        self._next_id = 0

    def _append(self, slot: _Slot) -> None:
        self.memories.append(slot)
        nbytes = self.slot_bytes + (self.raw_window_bytes if slot.is_episodic else 0)
        self.ledger.write_now(slot.item_id, nbytes)

    def ingest(self, z, t, raw_idx, severity=None) -> IngestResult:
        v_t = z / (np.linalg.norm(z) + 1e-12)
        sev = 0.1 if self.severity_mode == "labelfree" else (0.1 if severity is None else severity)
        item_id = self._next_id

        if not self.memories:
            self._next_id += 1
            self._append(_Slot(v_t.copy(), t, 1, sev, sev > 0.5, raw_idx, item_id))
            self.ledger.tick(t)
            return IngestResult(item_id, float("nan"), False, "INSERT_INIT")

        centroids = np.stack([m.centroid for m in self.memories])
        sims = centroids @ v_t
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        target = self.memories[best_idx]
        score = 1.0 - best_sim

        if best_sim >= self.sim_merge_thresh and sev < 0.5:
            updated = (1.0 - self.ewma_beta) * target.centroid + self.ewma_beta * v_t
            target.centroid = updated / np.linalg.norm(updated)
            target.mass += 1
            target.timestamp = t
            self.ledger.mark_dirty(target.item_id, self.slot_bytes)
            self.ledger.tick(t)
            return IngestResult(target.item_id, score, False, "ABSORBED_EWMA")

        if len(self.memories) >= self.max_slots:
            self._evict_lowest_utility(t)

        is_anomaly = (best_sim < 0.75) or (sev >= 0.5)
        self._next_id += 1
        self._append(_Slot(v_t.copy(), t, 1, max(sev, 1.0 - best_sim), is_anomaly, raw_idx, item_id))
        self.ledger.tick(t)
        action = "PINNED_EPISODIC" if is_anomaly else "NEW_STATE_CLUSTER"
        return IngestResult(item_id, score, action == "PINNED_EPISODIC", action)

    def _evict_lowest_utility(self, current_t: int, decay_lambda: float = 0.001) -> None:
        centroids = np.stack([m.centroid for m in self.memories])
        sim_matrix = centroids @ centroids.T
        local_density = np.mean(np.maximum(sim_matrix, 0), axis=1) * np.log1p(
            [m.mass for m in self.memories]
        )
        utilities = []
        for i, m in enumerate(self.memories):
            recency = np.exp(-decay_lambda * (current_t - m.timestamp))
            episodic_boost = 10.0 if m.is_episodic else 1.0
            utilities.append((m.severity * episodic_boost * recency) / (local_density[i] + 1e-5))
        evict_idx = int(np.argmin(utilities))
        gone = self.memories.pop(evict_idx)
        self.ledger.delete(gone.item_id)
        self._events.append(("evict", gone.item_id))

    def recall(self, q, t_now, k, mode) -> list[int]:
        qn = q / (np.linalg.norm(q) + 1e-12)
        scored = []
        for m in self.memories:
            spatial_sim = float(np.dot(qn, m.centroid))
            time_decay = float(np.exp(-0.0005 * (t_now - m.timestamp)))
            utility = 0.6 * spatial_sim + 0.2 * time_decay + 0.2 * min(m.severity, 1.0)
            scored.append((utility, m.item_id))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [i for _, i in scored[:k]]

    def items(self) -> ItemsView:
        n = len(self.memories)
        mu = np.stack([m.centroid for m in self.memories]) if n else np.zeros((0, self.d))
        return ItemsView(
            np.array([m.item_id for m in self.memories], dtype=np.int64),
            mu, np.zeros_like(mu),
            np.array([m.mass for m in self.memories], dtype=np.float64),
            np.array([m.timestamp for m in self.memories], dtype=np.int64),
        )

    def drain_events(self) -> list[tuple]:
        ev, self._events = self._events, []
        return ev

    def index_bytes(self) -> int:
        return len(self.memories) * self.slot_bytes

    def pinned_raw_bytes(self) -> int:
        return sum(1 for m in self.memories if m.is_episodic) * self.raw_window_bytes
