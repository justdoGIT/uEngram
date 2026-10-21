"""Flash write accounting shared by every store."""
from __future__ import annotations

TOMBSTONE_BYTES = 16


class WriteLedger:
    def __init__(self, flush_every: int = 50):
        self.flush_every = flush_every
        self.writes = 0
        self.bytes_written = 0
        self._dirty: dict[int, int] = {}

    def write_now(self, item_id: int, nbytes: int) -> None:
        self.writes += 1
        self.bytes_written += nbytes

    def mark_dirty(self, item_id: int, nbytes: int) -> None:
        self._dirty[item_id] = nbytes

    def delete(self, item_id: int) -> None:
        self.writes += 1
        self.bytes_written += TOMBSTONE_BYTES
        self._dirty.pop(item_id, None)

    def tick(self, t: int) -> None:
        if (t + 1) % self.flush_every == 0 and self._dirty:
            self.writes += len(self._dirty)
            self.bytes_written += sum(self._dirty.values())
            self._dirty.clear()
