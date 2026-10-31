"""Point-store baselines: naive append-only, FIFO ring, reservoir."""
from __future__ import annotations

import numpy as np

from .ledger import WriteLedger
from .store import IngestResult, ItemsView


class _PointStore:
    """Shared point-store mechanics; subclasses decide the slot to write."""

    name = "point"

    def __init__(self, d: int, raw_window_bytes: int, flush_every: int = 50):
        self.d = d
        self.raw_window_bytes = raw_window_bytes
        self.vec_bytes = d * 4 + 16
        self.ledger = WriteLedger(flush_every)
        self._events: list[tuple] = []
        self._next_id = 0

    # subclasses expose arrays via _view() -> (ids, X, t_last) of live items
    def _view(self):
        raise NotImplementedError

    def _score(self, z: np.ndarray) -> float:
        ids, X, _ = self._view()
        if len(ids) == 0:
            return float("nan")
        return float(np.min(np.sum((X - z) ** 2, axis=1)))

    def recall(self, q, t_now, k, mode) -> list[int]:
        ids, X, _ = self._view()
        if len(ids) == 0:
            return []
        d2 = np.sum((X - q) ** 2, axis=1)
        order = np.lexsort((ids, d2))[:k]
        return [int(ids[i]) for i in order]

    def items(self) -> ItemsView:
        ids, X, t_last = self._view()
        n = len(ids)
        return ItemsView(ids, X, np.zeros_like(X), np.ones(n), t_last)

    def drain_events(self) -> list[tuple]:
        ev, self._events = self._events, []
        return ev

    def index_bytes(self) -> int:
        return len(self._view()[0]) * self.vec_bytes

    def pinned_raw_bytes(self) -> int:
        return 0

    def _new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i


class NaiveStore(_PointStore):
    name = "naive"

    def __init__(self, d, raw_window_bytes, flush_every=50):
        super().__init__(d, raw_window_bytes, flush_every)
        self._X = np.empty((1024, d))
        self._ids = np.empty(1024, dtype=np.int64)
        self._tl = np.empty(1024, dtype=np.int64)
        self._n = 0

    def _view(self):
        n = self._n
        return self._ids[:n], self._X[:n], self._tl[:n]

    def ingest(self, z, t, raw_idx, **_):
        score = self._score(z)
        if self._n == len(self._ids):
            cap = 2 * self._n
            self._X = np.resize(self._X, (cap, self.d))
            self._ids = np.resize(self._ids, cap)
            self._tl = np.resize(self._tl, cap)
        i = self._new_id()
        self._X[self._n], self._ids[self._n], self._tl[self._n] = z, i, t
        self._n += 1
        self.ledger.write_now(i, self.vec_bytes)
        self.ledger.tick(t)
        return IngestResult(i, score, None, "append")


class FIFOStore(_PointStore):
    name = "fifo"

    def __init__(self, B, d, raw_window_bytes, flush_every=50):
        super().__init__(d, raw_window_bytes, flush_every)
        self.B = B
        self._X = np.zeros((B, d))
        self._ids = np.full(B, -1, dtype=np.int64)
        self._tl = np.zeros(B, dtype=np.int64)
        self._head = 0
        self._n = 0

    def _view(self):
        n = self._n
        return self._ids[:n], self._X[:n], self._tl[:n]

    def ingest(self, z, t, raw_idx, **_):
        score = self._score(z)
        i = self._new_id()
        if self._n == self.B:
            self._events.append(("evict", int(self._ids[self._head])))
        else:
            self._n += 1
        slot = self._head
        self._X[slot], self._ids[slot], self._tl[slot] = z, i, t
        self._head = (self._head + 1) % self.B
        self.ledger.write_now(i, self.vec_bytes)
        self.ledger.tick(t)
        return IngestResult(i, score, None, "append")


class ReservoirStore(_PointStore):
    name = "reservoir"

    def __init__(self, B, d, raw_window_bytes, seed=0, flush_every=50):
        super().__init__(d, raw_window_bytes, flush_every)
        self.B = B
        self.rng = np.random.default_rng(seed)
        self._X = np.zeros((B, d))
        self._ids = np.full(B, -1, dtype=np.int64)
        self._tl = np.zeros(B, dtype=np.int64)
        self._n = 0

    def _view(self):
        n = self._n
        return self._ids[:n], self._X[:n], self._tl[:n]

    def ingest(self, z, t, raw_idx, **_):
        score = self._score(z)
        slot = -1
        if t < self.B:
            slot = self._n
            self._n += 1
        elif self.rng.random() < self.B / (t + 1):
            slot = int(self.rng.integers(0, self.B))
            self._events.append(("evict", int(self._ids[slot])))
        if slot < 0:
            self.ledger.tick(t)
            return IngestResult(-1, score, None, "discard")
        i = self._new_id()
        self._X[slot], self._ids[slot], self._tl[slot] = z, i, t
        self.ledger.write_now(i, self.vec_bytes)
        self.ledger.tick(t)
        return IngestResult(i, score, None, "replace")
