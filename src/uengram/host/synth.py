"""Synthetic 1 Hz host ticks with known physics, for the host tests.

Regimes switch every `block` seconds between idle, build (CPU-bound, many
forks) and io (disk + network). Package temperature follows a one-pole RC
model of package power (tau 20 s, R 0.8 C/W, T_amb 30 C) and is quantised to
1 C like coretemp. Anomalies are injected per tick range.
"""
from __future__ import annotations

import numpy as np

from uengram.host import schema

T0 = 1_800_000_000.5  # ticks at wall second + 0.5, like the collector
NCPU, HZ, F_NOM = 4, 100, 3.9e9
REGIMES = {  # busy, ipc, freq, forks/s, disk B/s, net B/s, mem_used
    "idle": (0.03, 0.8, 0.8, 2, 2e4, 5e3, 0.35),
    "build": (0.85, 1.3, 1.05, 60, 3e5, 2e5, 0.55),
    "io": (0.25, 0.9, 0.9, 5, 4e7, 3e7, 0.45),
}
TAU, R_TH, T_AMB = 20.0, 0.8, 30.0


def make_ticks(n: int, seed: int = 0, block: int = 300, anomalies: dict | None = None,
               t_start: float = T0, order=("idle", "build", "io")) -> list[dict]:
    """anomalies: {name: (first_tick, last_tick)} with name in thrash, spin, oom, hot."""
    rng = np.random.default_rng(seed)
    anomalies = anomalies or {}
    T = T_AMB + R_TH * 5.0
    out = []
    for k in range(n):
        reg = order[(k // block) % len(order)]
        busy, ipc, freq, forks, disk, net, mem = REGIMES[reg]
        act = {a for a, (lo, hi) in anomalies.items() if lo <= k <= hi}
        if "spin" in act:
            busy, ipc, forks = 1.0, 0.25, 0
        dt = 1.0 + rng.normal(0, 2e-4)
        tot = int(round(NCPU * HZ * dt))
        b = int(np.clip(round(busy * tot + rng.normal(0, 4)), 0, tot))
        cyc = b / HZ * F_NOM * freq * rng.lognormal(0, 0.01)
        ref = b / HZ * F_NOM * rng.lognormal(0, 0.01)
        inst = ipc * cyc * rng.lognormal(0, 0.05)
        llc_mpki = 30.0 if "spin" in act else 2.0
        P = 3.0 + 25.0 * (b / tot) * freq ** 2 + rng.normal(0, 0.3)
        hot = 12.0 if "hot" in act else 0.0
        T += (dt / TAU) * (T_AMB + hot + R_TH * P - T)
        thrash = "thrash" in act
        t = {
            "dt_s": dt, "cpu_total_j": tot, "cpu_busy_j": b, "cpu_sys_j": int(b * 0.1),
            "psi_cpu_us": max(0.0, (busy - 0.7) * 3e5 + rng.normal(0, 2e3)),
            "ctxt": rng.poisson(200 + 3000 * busy), "forks": rng.poisson(forks),
            "pmu_inst": inst, "pmu_cyc": cyc, "pmu_ref": ref,
            "pmu_brmiss": inst * 4e-3 * rng.lognormal(0, 0.05),
            "pmu_llcmiss": inst * llc_mpki * 1e-3 * rng.lognormal(0, 0.05),
            "mem_used_frac": (0.93 if thrash else mem) + rng.normal(0, 0.003),
            "psi_mem_us": 3e5 if thrash else 0.0,
            "vm_pgscan": rng.poisson(8000) if thrash else 0, "vm_pgmajfault": rng.poisson(600 if thrash else 0.2),
            "vm_pswpout": rng.poisson(3000) if thrash else 0, "vm_oom_kill": int("oom" in act and k == anomalies["oom"][0]),
            "disk_rd_bytes": rng.poisson(disk * 0.3 / 512) * 512, "disk_wr_bytes": rng.poisson(disk * 0.7 / 512) * 512,
            "disk_rd_ios": rng.poisson(disk * 0.3 / 65536 + 0.1), "disk_wr_ios": rng.poisson(disk * 0.7 / 65536 + 0.3),
            "disk_wq_ms": rng.poisson(disk / 1e6 * 20 + 1), "disk_busy_max": min(1.0, disk / 8e7 + 0.002),
            "psi_io_us": (5e4 if reg == "io" else 0.0) + abs(rng.normal(0, 1e2)), "disk_timeouts": 0,
            "net_rx_bytes": rng.poisson(net * 0.6), "net_tx_bytes": rng.poisson(net * 0.4),
            "net_rx_pkts": rng.poisson(net * 0.6 / 900 + 3), "net_tx_pkts": rng.poisson(net * 0.4 / 900 + 3),
            "tcp_retrans": rng.poisson(0.01), "net_drop_err": 0,
            "pkg_energy_j": P * dt, "pkg_temp_c": float(round(T)), "throttle": 0,
            "pmu_running": 1.0, "read_us": 500.0, "self_cpu_us": 550.0,
        }
        assert set(t) == set(schema.TICK_FIELDS)
        out.append({"t": t_start + k, **t})
    return out
