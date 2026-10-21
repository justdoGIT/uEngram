"""Harness-only ground-truth provenance per stored item (never visible to stores)."""
from __future__ import annotations

from collections import Counter


class _Item:
    __slots__ = ("anom", "rege", "events", "rul_sum", "rul_n", "eol_units")

    def __init__(self):
        self.anom: Counter = Counter()
        self.rege: Counter = Counter()
        self.events: set = set()
        self.rul_sum = 0.0
        self.rul_n = 0
        self.eol_units: set = set()


def _majority(c: Counter):
    """Most frequent key; ties broken by the smaller key."""
    if not c:
        return None
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


class Provenance:
    def __init__(self):
        self.items: dict[int, _Item] = {}

    def record(self, item_id: int, anomaly: int = 0, regime: int = -1, era: int = -1,
               event_id: int = -1, rul: float | None = None, unit: int | None = None,
               eol: bool = False) -> None:
        it = self.items.setdefault(item_id, _Item())
        it.anom[anomaly] += 1
        if anomaly == 0 and regime >= 0:
            it.rege[(regime, era)] += 1
        if event_id >= 0:
            it.events.add(event_id)
        if rul is not None:
            it.rul_sum += rul
            it.rul_n += 1
        if eol and unit is not None:
            it.eol_units.add(unit)

    def apply(self, events: list[tuple]) -> None:
        for ev in events:
            if ev[0] == "evict":
                self.items.pop(ev[1], None)
            elif ev[0] == "merge":
                src = self.items.pop(ev[1], None)
                if src is None:
                    continue
                dst = self.items.get(ev[2])
                if dst is None:
                    self.items[ev[2]] = src
                    continue
                dst.anom.update(src.anom)
                dst.rege.update(src.rege)
                dst.events |= src.events
                dst.rul_sum += src.rul_sum
                dst.rul_n += src.rul_n
                dst.eol_units |= src.eol_units

    def majority_anomaly(self, item_id: int):
        it = self.items.get(item_id)
        return None if it is None else _majority(it.anom)

    def majority_regime_era(self, item_id: int):
        it = self.items.get(item_id)
        return None if it is None else _majority(it.rege)

    def mean_rul(self, item_id: int):
        it = self.items.get(item_id)
        return None if it is None or it.rul_n == 0 else it.rul_sum / it.rul_n
