import pytest

from uengram.host.perf import PerfCounters, PerfUnavailable


def test_group_counts_own_instructions_and_cycles():
    try:
        pc = PerfCounters(system=False)  # own process, user space: allowed at paranoid 2
    except PerfUnavailable as e:
        pytest.skip(str(e))
    try:
        pc.delta()
        sum(i * i for i in range(300_000))
        d, running = pc.delta()
    finally:
        pc.close()
    if running == 0.0:
        # ref-cycles only fits fixed counter 2; a system-wide group (the
        # collector itself) holds it, so a per-task group is never scheduled
        pytest.skip("PMU fixed counter held by a system-wide event group")
    assert running == pytest.approx(1.0)  # five events fit: no multiplexing
    assert d["pmu_inst"] > 1e6 and d["pmu_cyc"] > 1e5
    assert 0.05 < d["pmu_inst"] / d["pmu_cyc"] < 8.0
    assert d["pmu_brmiss"] < d["pmu_inst"]
