import numpy as np
import pytest

from uengram.host.node import NodeMemory
from uengram.host.synth import make_ticks

ANOMALIES = {"thrash": (1000, 1200), "spin": (2000, 2200), "hot": (2600, 2900), "oom": (3000, 3000)}


@pytest.fixture(scope="module")
def run():
    ticks = make_ticks(4 * 3600, seed=1)
    node = NodeMemory.commission(ticks, {"host": "synth"})
    start = ticks[-1]["t"] + 1
    start += (32 - start % 32) % 32
    test = make_ticks(3600, seed=2, t_start=start + 0.5, anomalies=ANOMALIES)
    recs = list(node.feed_ticks(test))
    for r in recs:
        r["k"] = r["t0"] - test[0]["t"]
    return node, recs


def _during(recs, name):
    lo, hi = ANOMALIES[name]
    return [r for r in recs if lo + 32 <= r["k"] <= hi - 32]


def test_threshold_is_calibrated_on_persistent_scores(run):
    node, _ = run
    rep = node.commission_report
    # single windows straddling a regime change are far out; requiring two
    # consecutive novel windows brings the calibrated threshold down by orders
    assert rep["alarm_threshold_d2"] < rep["single_window_q"] / 10


def test_normal_operation_stays_quiet(run):
    _, recs = run
    normal = [r for r in recs if not any(lo - 64 <= r["k"] <= hi + 64 for lo, hi in ANOMALIES.values())]
    assert len(normal) > 50
    assert np.mean([r["alarm"] for r in normal]) <= 0.02


@pytest.mark.parametrize("name,modality", [("thrash", "memory"), ("spin", "compute"),
                                           ("hot", "power_thermal")])
def test_sustained_anomaly_alarms_and_names_its_modality(run, name, modality):
    _, recs = run
    rs = _during(recs, name)
    first = next(r for r in rs if r["alarm_latent"])
    assert first["k"] <= ANOMALIES[name][0] + 96  # within three windows
    assert max(first["modality_d2"], key=first["modality_d2"].get) == modality
    # once the state is learned as a regime, later visits still carry the alarm tag
    assert all(r["state_alarmed_at"] is not None for r in rs if r["kind"] == "semantic")


def test_single_error_event_alarms_through_the_count_model(run):
    _, recs = run
    r = next(r for r in recs if r["k"] <= ANOMALIES["oom"][0] < r["k"] + 32)
    assert r["alarm_event"] and r["events"] == {"oom_kills": 1}


def test_saved_memory_continues_identically(tmp_path, run):
    node, _ = run
    path = str(tmp_path / "m.pkl")
    node.save(path)
    more = make_ticks(320, seed=9, t_start=node._recent[-1][0] * 32 + node.t_origin + 32.5)
    a = [r["score"] for r in node.feed_ticks(more)]
    b = [r["score"] for r in NodeMemory.load(path).feed_ticks(more)]
    assert a == b and len(a) == 10
