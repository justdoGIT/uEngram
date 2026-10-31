"""On-delay (persistence) and off-delay (release) filters for one alarm channel.

Each filter consumes one value per eligible window: True (the channel's condition is
met), False (it is not) or None (unknown; the filter state holds). After every step,
`trigger` says the channel asks for an alarm and `clear` says it holds no pending
evidence. An alarm opens when any channel triggers and closes when every channel is
clear.

  counter      up/down counter capped at k (AlarmPolicy v2): +1 on True, -1 on False
               (floor 0); trigger at k, clear at 0
  consecutive  ISA-18.2 on-delay and off-delay timers: trigger after k True in a row,
               clear after k False in a row
  m_of_n       generalized delay timer (Adnan et al., J. Process Control 23, 2013):
               trigger when at least k of the last n known windows are True; clear
               when none of them is
  ewma         low-pass filter of the binary condition (Jakobs et al. 2025, Eq. 1):
               z <- z + lam (y - z) with 1 - lam = 0.5 ** (1 / (k - 0.5)), so k True in
               a row from z = 0 cross 0.5 and k - 1 do not; trigger z > 0.5, clear z < 0.5

Staleness (`stale` > 0): a held state may not outlive the evidence. After more than
`stale` consecutive unknown inputs the filter resets to clear, so an alarm that can no
longer be confirmed does not stand indefinitely (ISA-18.2 standing and stale alarms).
"""
from __future__ import annotations

from collections import deque

MODELS = ("counter", "consecutive", "m_of_n", "ewma")


class Counter:
    def __init__(self, k: int):
        self.k, self.c = k, 0

    def reset(self) -> None:
        self.c = 0

    def step(self, y: bool | None) -> None:
        if y is not None:
            self.c = min(self.c + 1, self.k) if y else max(self.c - 1, 0)

    @property
    def trigger(self) -> bool:
        return self.c >= self.k

    @property
    def clear(self) -> bool:
        return self.c == 0


class Consecutive:
    def __init__(self, k: int):
        self.k = k
        self.reset()

    def reset(self) -> None:
        self.on_run, self.off_run = 0, self.k  # clear

    def step(self, y: bool | None) -> None:
        if y is True:
            self.on_run, self.off_run = self.on_run + 1, 0
        elif y is False:
            self.on_run, self.off_run = 0, self.off_run + 1

    @property
    def trigger(self) -> bool:
        return self.on_run >= self.k

    @property
    def clear(self) -> bool:
        return self.off_run >= self.k


class MOfN:
    def __init__(self, k: int, n: int):
        if n < k:
            raise ValueError("m_of_n needs n >= k")
        self.k, self.last = k, deque(maxlen=n)

    def reset(self) -> None:
        self.last.clear()

    def step(self, y: bool | None) -> None:
        if y is not None:
            self.last.append(bool(y))

    @property
    def trigger(self) -> bool:
        return sum(self.last) >= self.k

    @property
    def clear(self) -> bool:
        return not any(self.last)


class EWMA:
    def __init__(self, k: int):
        self.lam, self.z = 1.0 - 0.5 ** (1.0 / (k - 0.5)), 0.0

    def reset(self) -> None:
        self.z = 0.0

    def step(self, y: bool | None) -> None:
        if y is not None:
            self.z += self.lam * (float(y) - self.z)

    @property
    def trigger(self) -> bool:
        return self.z > 0.5

    @property
    def clear(self) -> bool:
        return self.z < 0.5


class Stale:
    """Wraps a filter: more than `limit` consecutive unknown inputs reset it to clear."""

    def __init__(self, inner, limit: int):
        self.inner, self.limit, self.unknown_run = inner, limit, 0

    def reset(self) -> None:
        self.inner.reset()

    def step(self, y: bool | None) -> None:
        self.unknown_run = self.unknown_run + 1 if y is None else 0
        if self.unknown_run > self.limit:
            self.inner.reset()
        else:
            self.inner.step(y)

    @property
    def trigger(self) -> bool:
        return self.inner.trigger

    @property
    def clear(self) -> bool:
        return self.inner.clear

    @property
    def c(self) -> int:
        return getattr(self.inner, "c", 0)


def make_filter(model: str, k: int, window: int = 0, stale: int = 0):
    """One channel's filter; m_of_n uses n = window, or 2k when window is 0; stale > 0
    bounds how long unknown inputs may hold its state."""
    if k < 1:
        raise ValueError("persistence k must be >= 1")
    if model == "counter":
        f = Counter(k)
    elif model == "consecutive":
        f = Consecutive(k)
    elif model == "m_of_n":
        f = MOfN(k, window or 2 * k)
    elif model == "ewma":
        f = EWMA(k)
    else:
        raise ValueError(f"unknown persistence model {model!r}; expected one of {MODELS}")
    return Stale(f, stale) if stale > 0 else f
