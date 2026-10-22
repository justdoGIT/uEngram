import numpy as np
import pytest

from uengram.host import schema
from uengram.host.features import windows
from uengram.host.synth import T0, make_ticks

NAMES = schema.feature_names()


def col(batch, name):
    return batch.F[:, NAMES.index(name)]


def test_only_complete_wall_aligned_windows_are_kept():
    ticks = make_ticks(32 * 3 + 5, seed=0, t_start=np.floor(T0 / 32) * 32 + 0.5)
    del ticks[40]  # a missed tick makes the second window incomplete
    b = windows(ticks, 32)
    assert np.all(b.t0 % 32 == 0)
    assert len(b) == 2 and b.t0[1] - b.t0[0] == 64


def test_ratios_are_ratios_of_window_sums():
    ticks = make_ticks(32, seed=0, t_start=np.floor(T0 / 32) * 32 + 0.5, order=("build",))
    ticks[3]["pmu_inst"], ticks[3]["pmu_cyc"] = 1.0, 1e3  # a nearly idle tick with IPC 0.001
    b = windows(ticks, 32)
    inst = sum(r["pmu_inst"] for r in ticks)
    cyc = sum(r["pmu_cyc"] for r in ticks)
    assert col(b, "ipc")[0] == pytest.approx(np.log(inst / cyc + 1e-6))
    # a mean of per-tick ratios would be pulled down by the idle tick
    mean_of_ratios = np.mean([r["pmu_inst"] / r["pmu_cyc"] for r in ticks])
    assert np.exp(col(b, "ipc")[0]) > mean_of_ratios * 1.02


def test_burst_ignores_a_single_step_but_tracks_fluctuation():
    start = np.floor(T0 / 32) * 32 + 0.5
    steady = make_ticks(32, seed=1, t_start=start, order=("io",))
    step = [dict(r) for r in steady]
    for r in step[16:]:  # a copy job starts mid-window: same noise, 50x the bytes
        r["disk_rd_bytes"] *= 50
        r["disk_wr_bytes"] *= 50
    noisy = [dict(r) for r in steady]
    rng = np.random.default_rng(0)
    for r in noisy:
        f = rng.lognormal(0, 1.0)
        r["disk_rd_bytes"] *= f
        r["disk_wr_bytes"] *= f
    b = [col(windows(x, 32), "disk_bytes.burst")[0] for x in (steady, step, noisy)]
    assert abs(b[1] - b[0]) < 0.2  # the step moves the level, not the burstiness
    assert b[2] - b[0] > 2.0


def test_pmu_attributes_are_missing_when_the_group_was_not_scheduled():
    ticks = make_ticks(32, seed=0, t_start=np.floor(T0 / 32) * 32 + 0.5)
    for r in ticks[:20]:
        r["pmu_running"] = 0.0
    b = windows(ticks, 32)
    for n in ("ipc", "freq_ratio", "branch_mpki", "llc_mpki"):
        assert np.isnan(col(b, n)[0])
    assert np.isfinite(col(b, "cpu_busy")[0])
