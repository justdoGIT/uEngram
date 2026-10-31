import numpy as np

from uengram.reference_baseline import ReferenceBaseline
from uengram.baselines import FIFOStore, NaiveStore, ReservoirStore
from uengram.ledger import WriteLedger


def test_fifo_holds_last_B():
    s = FIFOStore(B=10, d=4, raw_window_bytes=100)
    rng = np.random.default_rng(0)
    res = [s.ingest(rng.normal(size=4), t, t) for t in range(37)]
    assert sorted(s.items().ids.tolist()) == [r.item_id for r in res[-10:]]
    ev = [e for e in s.drain_events() if e[0] == "evict"]
    assert len(ev) == 27


def test_reservoir_uniform_inclusion():
    T, B, trials = 100, 10, 4000
    counts = np.zeros(T)
    for seed in range(trials):
        s = ReservoirStore(B=B, d=1, raw_window_bytes=1, seed=seed)
        for t in range(T):
            s.ingest(np.array([float(t)]), t, t)
        counts[s.items().mu[:, 0].astype(int)] += 1
    freq = counts / trials
    assert np.all(np.abs(freq - B / T) < 0.02)


def test_naive_recall_bruteforce():
    rng = np.random.default_rng(1)
    s = NaiveStore(d=3, raw_window_bytes=1)
    X = rng.normal(size=(2500, 3))  # crosses the growth boundary
    for t, x in enumerate(X):
        s.ingest(x, t, t)
    q = rng.normal(size=3)
    assert s.recall(q, 0, 1, "current")[0] == int(np.argmin(((X - q) ** 2).sum(1)))
    assert s.ledger.writes == 2500


def test_ledger_dirty_flush_bound():
    led = WriteLedger(flush_every=50)
    for t in range(1000):
        led.mark_dirty(t % 3, 8)
        led.tick(t)
    assert led.writes <= 20 * 3
    led.delete(0)
    assert led.writes <= 20 * 3 + 1


def test_reference_budget_and_oracle_flag():
    rng = np.random.default_rng(2)
    g = ReferenceBaseline(max_slots=5, d=8, raw_window_bytes=10, severity_mode="oracle")
    assert g.name == "reference_oracle"
    flagged = 0
    for t in range(300):
        z = rng.normal(size=8)
        r = g.ingest(z, t, t, severity=0.9 if t % 50 == 0 else 0.1)
        flagged += bool(r.flagged)
        assert len(g.memories) <= 5
    assert flagged > 0 and any(e[0] == "evict" for e in g.drain_events())
