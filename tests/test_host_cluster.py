import pytest

from uengram.host.cluster import STALL_HOLD_S, load_now, placement

NOW = 1_800_000_000.0


def _tick(busy, total, psi_mem=0.0):
    return {"dt_s": 1.0, "cpu_busy_j": busy, "cpu_total_j": total, "psi_cpu_us": 0.0, "psi_mem_us": psi_mem,
            "psi_io_us": 0.0}


def _node(busy=10, up=True, age=10.0, pressure=None, alarm=False):
    state = {"t0": NOW - age, "alarm": alarm, "saturation": {}, "state_alarmed_at": None}
    return {"state": state, "ticks": [_tick(busy, 400)] * 5, "pressure": pressure or {},
            "link": {"up": up, "since": NOW - 100, "error": None if up else "exit 255"}}


def test_load_is_a_ratio_of_sums_over_the_last_ticks():
    # 0 of 100 jiffies busy, then 300 of 300: the mean of ratios would say 50 %
    assert load_now([_tick(0, 100), _tick(300, 300)])["busy"] == pytest.approx(0.75)


def test_placement_ranks_by_headroom_and_holds_back_stalling_stale_and_silent_nodes():
    snap = {"t": NOW, "nodes": {
        "busy": _node(busy=300),
        "idle": _node(busy=20),
        "stalling": _node(busy=0, pressure={"mem_pressure": NOW - 1}),
        "stale": _node(busy=0, age=600),
        "down": _node(busy=0, up=False),
        "alarmed": _node(busy=0, alarm=True),
    }}
    rows = placement(snap, NOW)
    assert [r["node"] for r in rows[:2]] == ["idle", "busy"]
    verdict = {r["node"]: r["verdict"] for r in rows}
    assert verdict["stalling"] == verdict["alarmed"] == "avoid"
    assert verdict["stale"] == verdict["down"] == "unknown"


def test_a_stall_stops_counting_after_two_trigger_windows():
    snap = {"t": NOW, "nodes": {"n": _node(pressure={"cpu_pressure": NOW - STALL_HOLD_S - 0.5})}}
    assert placement(snap, NOW)[0]["verdict"] == "ok"
