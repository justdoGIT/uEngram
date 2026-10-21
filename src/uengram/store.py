"""Common memory-store protocol and item view."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .ledger import WriteLedger


@dataclass
class IngestResult:
    item_id: int
    score: float  # prequential novelty, computed before the update
    flagged: bool | None
    action: str


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
