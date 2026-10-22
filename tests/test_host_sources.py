import pytest

from uengram.host import sources

STAT = """cpu  1000 50 300 9000 200 10 20 5 400 0
cpu0 500 25 150 4500 100 5 10 2 200 0
intr 12345 1 2 3
ctxt 777
btime 1700000000
processes 4242
procs_running 2
"""


def test_stat_counts_guest_time_once():
    s = sources.parse_stat(STAT)
    # guest (400) is already inside user (1000) and must not be added again
    assert s["cpu_total_j"] == 1000 + 50 + 300 + 9000 + 200 + 10 + 20 + 5
    assert s["cpu_busy_j"] == 1000 + 50 + 300 + 10 + 20
    assert s["cpu_sys_j"] == 300 + 10 + 20
    assert (s["ctxt"], s["forks"]) == (777, 4242)


def test_diskstats_keeps_only_listed_whole_devices():
    text = ("   8       0 sda 100 0 800 50 200 0 1600 70 0 90 120 0 0 0 0\n"
            "   8       1 sda1 100 0 800 50 200 0 1600 70 0 90 120 0 0 0 0\n"
            "   7       0 loop0 5 0 40 1 0 0 0 0 0 1 1 0 0 0 0\n")
    d = sources.parse_diskstats(text, {"sda"})
    assert set(d) == {"sda"}  # the partition would count the same bytes twice
    assert d["sda"] == {"rd_ios": 100, "rd_sec": 800, "wr_ios": 200, "wr_sec": 1600,
                        "io_ticks": 90, "wq_ms": 120}


def test_net_dev_sums_physical_nics_and_fault_columns():
    text = ("Inter-|   Receive                                                |  Transmit\n"
            " face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed\n"
            "    lo: 999 9 0 0 0 0 0 0 999 9 0 0 0 0 0 0\n"
            "  eth0: 1000 10 1 2 3 4 0 7 2000 20 5 6 7 8 9 0\n")
    n = sources.parse_net_dev(text, {"eth0"})
    assert (n["net_rx_bytes"], n["net_rx_pkts"], n["net_tx_bytes"], n["net_tx_pkts"]) == (1000, 10, 2000, 20)
    assert n["net_drop_err"] == 1 + 2 + 3 + 4 + 5 + 6 + 7 + 9  # collisions and multicast excluded


def test_snmp_retrans_by_header_name():
    text = ("Tcp: RtoAlgorithm RtoMin RtoMax MaxConn ActiveOpens PassiveOpens AttemptFails EstabResets"
            " CurrEstab InSegs OutSegs RetransSegs InErrs OutRsts InCsumErrors\n"
            "Tcp: 1 200 120000 -1 10 5 0 0 3 1000 900 17 0 2 0\n")
    assert sources.parse_snmp_tcp(text) == 17


class _FakeSnap:
    """Scripted cumulative readings for Ticker."""

    def __init__(self, reads, rng):
        self.reads, self.rapl = iter(reads), [(None, rng)]

    def read(self):
        return next(self.reads)


def _reading(mono_s, energy, ctxt, wr_sec=0):
    base = {k: 0 for k in sources.COUNTERS}
    base.update(ctxt=ctxt, mono_ns=int(mono_s * 1e9), rapl=[energy], mem_used_frac=0.5,
                pkg_temp_c=50.0, pmu={}, pmu_running=1.0, read_us=1.0,
                disks={"sda": {"rd_ios": 0, "rd_sec": 0, "wr_ios": 1, "wr_sec": wr_sec,
                               "io_ticks": 0, "wq_ms": 0}})
    return base


def test_ticker_unwraps_rapl_and_nulls_counter_resets():
    rng = 262_143_328_850
    snap = _FakeSnap([_reading(0.0, rng - 1_000_000, 100, 8),
                      _reading(1.5, 4_000_000, 150, 16),  # energy wrapped; 1.5 s late tick
                      _reading(2.5, 9_000_000, 20, 24)], rng)  # ctxt reset
    tk = sources.Ticker(snap)
    a = tk.tick()
    assert a["dt_s"] == pytest.approx(1.5)
    assert a["pkg_energy_j"] == pytest.approx(5.0)  # (1e6 + 4e6) uJ across the wrap
    assert a["ctxt"] == 50 and a["disk_wr_bytes"] == 8 * 512
    b = tk.tick()
    assert b["ctxt"] is None and b["pkg_energy_j"] == pytest.approx(5.0)
