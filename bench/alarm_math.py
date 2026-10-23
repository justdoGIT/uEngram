"""False-alarm arithmetic for the persistence filters (src/uengram/persistence.py).

The per-window condition y_t in {0, 1} on healthy windows is modelled as a two-state
Markov chain: P(y_t = 1 | y_{t-1} = 0) = a, P(y_t = 1 | y_{t-1} = 1) = b. Independent
windows are the special case b = a = p. Clustered exceedances (b >> a) are what the
MetroPT3 scores show.

ARL0 is the expected number of windows, starting clear with y_0 = 0, until the filter
first triggers. It is exact for counter, consecutive and m_of_n (an absorbing Markov chain
over (filter state, last y)) and for ewma uses the same chain on a fine grid of z.
`simulate_episode_rate` runs the real filter code on a synthetic sequence instead.

Independent closed form for k consecutive (geometric run waiting time):
    ARL0 = (1 - p**k) / ((1 - p) * p**k)
"""
from __future__ import annotations

import numpy as np

from uengram.persistence import make_filter

RESOLUTION = 1e15  # largest ARL0 (windows) the float64 absorbing-chain solve can resolve


def markov_fit(x: np.ndarray, contiguous: np.ndarray) -> dict:
    """(p, a, b) from a boolean sequence; `contiguous[i]` says window i follows i - 1."""
    x = np.asarray(x, bool)
    prev, cur = x[:-1][contiguous[1:]], x[1:][contiguous[1:]]
    a = float(cur[~prev].mean()) if (~prev).any() else 0.0
    b = float(cur[prev].mean()) if prev.any() else 0.0
    return {"p": float(x.mean()), "a": a, "b": b, "pairs": int(contiguous[1:].sum())}


def _absorbing_arl(states: list, start, step) -> float:
    """Expected steps to absorption. step.next(state, y) -> next state or None (absorbed).

    Infinite when some state reachable from `start` can never be absorbed (for example
    b = 0 makes k >= 2 consecutive exceedances impossible); the linear system is then
    singular, so it is solved only on the reachable states once absorption is certain."""
    edges, absorbing = {}, set()
    for s in states:
        edges[s] = []
        for y, prob in step.transitions(s):
            if prob > 0:
                nxt = step.next(s, y)
                if nxt is None:
                    absorbing.add(s)
                else:
                    edges[s].append((nxt, prob))
    reach, todo = {start}, [start]
    while todo:
        for nxt, _ in edges[todo.pop()]:
            if nxt not in reach:
                reach.add(nxt)
                todo.append(nxt)
    can_absorb, changed = set(absorbing), True
    while changed:
        changed = False
        for s in reach - can_absorb:
            if any(nxt in can_absorb for nxt, _ in edges[s]):
                can_absorb.add(s)
                changed = True
    if not reach <= can_absorb:
        return float("inf")
    order = sorted(reach, key=states.index)
    index = {s: i for i, s in enumerate(order)}
    Q = np.zeros((len(order), len(order)))
    for s in order:
        for nxt, prob in edges[s]:
            Q[index[s], index[nxt]] += prob
    t = np.linalg.solve(np.eye(len(order)) - Q, np.ones(len(order)))
    arl = float(t[index[start]])
    # Beyond ~1e15 windows (2e10 years of 10-min windows) I - Q is singular to float64
    # precision and the solve can return garbage, even negative values: report "never".
    return arl if 1.0 <= arl <= RESOLUTION else float("inf")


class _Chain:
    """y follows the Markov chain; the filter state is part of the chain state."""

    def __init__(self, a: float, b: float):
        self.a, self.b = a, b

    def transitions(self, s):
        p1 = self.b if s[-1] else self.a
        return ((1, p1), (0, 1.0 - p1))


def arl0(model: str, k: int, a: float, b: float, window: int = 0, grid: int = 4000) -> float:
    if not 0 < a < 1 or not 0 <= b < 1:
        raise ValueError("need 0 < a < 1 and 0 <= b < 1")
    chain = _Chain(a, b)
    if model == "counter":
        states = [(c, y) for c in range(k) for y in (0, 1)]
        chain.next = lambda s, y: None if (c := min(s[0] + 1, k) if y else max(s[0] - 1, 0)) >= k else (c, y)
        return _absorbing_arl(states, (0, 0), chain)
    if model == "consecutive":
        states = [(r, y) for r in range(k) for y in (0, 1)]
        chain.next = lambda s, y: None if (r := s[0] + 1 if y else 0) >= k else (r, y)
        return _absorbing_arl(states, (0, 0), chain)
    if model == "m_of_n":
        n = window or 2 * k
        states = [(h, y) for h in range(1 << n) for y in (0, 1) if bin(h).count("1") < k]

        def nxt(s, y):
            h = ((s[0] << 1) | y) & ((1 << n) - 1)
            return None if bin(h).count("1") >= k else (h, y)
        chain.next = nxt
        return _absorbing_arl(states, (0, 0), chain)
    if model == "ewma":
        lam = 1.0 - 0.5 ** (1.0 / (k - 0.5))
        zs = np.linspace(0.0, 0.5, grid + 1)  # z <= 0.5 is not yet triggered
        states = [(i, y) for i in range(grid + 1) for y in (0, 1)]

        def nxt(s, y):
            z = zs[s[0]] + lam * (y - zs[s[0]])
            return None if z > 0.5 else (int(round(z / 0.5 * grid)), y)
        chain.next = nxt
        return _absorbing_arl(states, (0, 0), chain)
    raise ValueError(f"unknown model {model}")


def simulate_episode_rate(model: str, k: int, a: float, b: float, n: int, seed: int = 0,
                          window: int = 0) -> dict:
    """Run the real filter on n synthetic Markov windows: alarm openings per window,
    alarm-time share and mean windows to first trigger over restarts."""
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    f, active, y = make_filter(model, k, window), False, False
    openings = alarm = 0
    for i in range(n):
        y = u[i] < (b if y else a)
        f.step(bool(y))
        if active and f.clear:
            active = False
        elif not active and f.trigger:
            active, openings = True, openings + 1
        alarm += active
    return {"openings_per_window": openings / n, "alarm_share": alarm / n}


def simulate_arl0(model: str, k: int, a: float, b: float, reps: int, seed: int = 0,
                  window: int = 0, cap: int = 10**7) -> tuple[float, float]:
    """(mean, standard error) of windows to the first trigger of the real filter, starting
    clear with y_0 = 0; the Monte Carlo check of `arl0`."""
    rng = np.random.default_rng(seed)
    times = []
    for _ in range(reps):
        f, y, t = make_filter(model, k, window), False, 0
        while t < cap:
            t += 1
            y = rng.random() < (b if y else a)
            f.step(bool(y))
            if f.trigger:
                break
        times.append(t)
    times = np.asarray(times, float)
    return float(times.mean()), float(times.std(ddof=1) / np.sqrt(reps))
