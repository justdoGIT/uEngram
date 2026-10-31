"""Common memory-store protocol and item view."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .ledger import WriteLedger


@dataclass
class IngestResult:
    item_id: int
    score: float  # prequential novelty vs normal states, computed before the update
    flagged: bool | None  # alarm (None for stores without alarm logic)
    action: str
    # distance from a frozen healthy reference (NaN without one); grows with
    # wear that the adaptive memory has learned to treat as normal
    health: float = float("nan")


@dataclass
class ItemsView:
    ids: np.ndarray  # [n] int
    mu: np.ndarray  # [n,d]
    var: np.ndarray  # [n,d]
    mass: np.ndarray  # [n]
    t_last: np.ndarray  # [n]


class MemoryStore(Protocol):
    name: str
    ledger: WriteLedger

    def ingest(self, z: np.ndarray, t: int, raw_idx: int) -> IngestResult: ...

    def recall(self, q: np.ndarray, t_now: int, k: int, mode: str) -> list[int]: ...

    def items(self) -> ItemsView: ...

    def drain_events(self) -> list[tuple]: ...

    def index_bytes(self) -> int: ...

    def pinned_raw_bytes(self) -> int: ...
