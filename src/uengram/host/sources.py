"""Kernel sources for one host snapshot, and the per-tick delta computation; stdlib only.

`Snapshot` reads cumulative counters and gauges from /proc and /sys through
file descriptors kept open (pread at offset 0 regenerates /proc and sysfs
content), timing the whole read so the skew between the first and last
source is known. `Ticker` turns consecutive snapshots into a tick record:
counter deltas over the CLOCK_MONOTONIC interval, gauges at tick end.

Accuracy rules applied here:
* /proc/stat CPU time: total = user+nice+system+idle+iowait+irq+softirq+steal;
  guest and guest_nice are already inside user/nice and are not added again.
* diskstats sectors are 512 B regardless of the device block size; only
  whole devices with a backing `device` link count (no partitions, loop,
  zram, dm), so a byte is never counted at two layers.
* NICs: only interfaces with a `device` link (no lo, bridges, tunnels).
* RAPL energy wraps at max_energy_range_uj and is unwrapped per domain;
  only package-* domains are summed (core/uncore/dram are nested inside).
* core_throttle_count is per physical core but exposed on every SMT sibling,
  package_throttle_count on every CPU of the package: both are deduplicated
  by topology before summing.
* Any other negative delta is a counter reset; that field is null for the tick.
"""
from __future__ import annotations

import glob
import os
import resource
import time

READ_CHUNK = 1 << 16


class _File:
    def __init__(self, path: str):
        self.path = path
        self.fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)

    def text(self) -> str:
        parts, off = [], 0
        while True:
            b = os.pread(self.fd, READ_CHUNK, off)
            if not b:
                break
            parts.append(b)
            off += len(b)
        return b"".join(parts).decode()


def _first_line_ints(text: str) -> list[int]:
    return [int(x) for x in text.split("\n", 1)[0].split()[1:]]


def _psi_some_total(text: str) -> int:
    for line in text.splitlines():
        if line.startswith("some"):
            return int(line.rsplit("total=", 1)[1])
    raise ValueError("no 'some' line in pressure file")


def parse_stat(text: str) -> dict[str, int]:
    out = {}
    for line in text.splitlines():
        if line.startswith("cpu "):
            v = [int(x) for x in line.split()[1:9]]  # user nice system idle iowait irq softirq steal
            user, nice, system, idle, iowait, irq, softirq, steal = v + [0] * (8 - len(v))
            out["cpu_total_j"] = user + nice + system + idle + iowait + irq + softirq + steal
            out["cpu_busy_j"] = user + nice + system + irq + softirq
            out["cpu_sys_j"] = system + irq + softirq
        elif line.startswith("ctxt "):
            out["ctxt"] = int(line.split()[1])
        elif line.startswith("processes "):
            out["forks"] = int(line.split()[1])
    return out


def parse_kv(text: str, keys: set[str]) -> dict[str, int]:
    out = {}
    for line in text.splitlines():
        k, _, rest = line.partition(" ")
        k = k.rstrip(":")
        if k in keys:
            out[k] = int(rest.split()[0])
    return out


def parse_diskstats(text: str, disks: set[str]) -> dict[str, dict[str, int]]:
    out = {}
    for line in text.splitlines():
        f = line.split()
        if len(f) >= 14 and f[2] in disks:
            out[f[2]] = {"rd_ios": int(f[3]), "rd_sec": int(f[5]), "wr_ios": int(f[7]),
                         "wr_sec": int(f[9]), "io_ticks": int(f[12]), "wq_ms": int(f[13])}
    return out


def parse_net_dev(text: str, nics: set[str]) -> dict[str, int]:
    tot = {"net_rx_bytes": 0, "net_rx_pkts": 0, "net_tx_bytes": 0, "net_tx_pkts": 0, "net_drop_err": 0}
    for line in text.splitlines()[2:]:
        name, _, rest = line.partition(":")
        if name.strip() not in nics:
            continue
        v = [int(x) for x in rest.split()]
        tot["net_rx_bytes"] += v[0]
        tot["net_rx_pkts"] += v[1]
        tot["net_tx_bytes"] += v[8]
        tot["net_tx_pkts"] += v[9]
        # rx errs drop fifo frame + tx errs drop fifo carrier
        tot["net_drop_err"] += v[2] + v[3] + v[4] + v[5] + v[10] + v[11] + v[12] + v[14]
    return tot


def parse_snmp_tcp(text: str, key: str = "RetransSegs") -> int:
    lines = [ln for ln in text.splitlines() if ln.startswith("Tcp:")]
    hdr, val = lines[0].split()[1:], lines[1].split()[1:]
    return int(val[hdr.index(key)])


def physical_block_devices(root: str = "/sys/block") -> list[str]:
    return sorted(d for d in os.listdir(root) if os.path.exists(f"{root}/{d}/device"))


def physical_nics(root: str = "/sys/class/net") -> list[str]:
    return sorted(d for d in os.listdir(root) if os.path.exists(f"{root}/{d}/device"))


def throttle_files(root: str = "/sys/devices/system/cpu") -> list[str]:
    """One core_throttle_count per physical core and one package_throttle_count per package."""
    seen_core, seen_pkg, out = set(), set(), []
    for cpu in sorted(glob.glob(f"{root}/cpu[0-9]*"), key=lambda p: int(p.rsplit("cpu", 1)[1])):
        tt = f"{cpu}/thermal_throttle"
        if not os.path.isdir(tt):
            continue
        with open(f"{cpu}/topology/physical_package_id") as f:
            pkg = f.read().strip()
        with open(f"{cpu}/topology/core_id") as f:
            core = (pkg, f.read().strip())
        if core not in seen_core:
            seen_core.add(core)
            out.append(f"{tt}/core_throttle_count")
        if pkg not in seen_pkg:
            seen_pkg.add(pkg)
            out.append(f"{tt}/package_throttle_count")
    return out


def rapl_packages(root: str = "/sys/class/powercap") -> list[tuple[str, int]]:
    out = []
    for d in sorted(glob.glob(f"{root}/intel-rapl:[0-9]*")):
        if d.count(":") != 1:
            continue  # sub-domains (core, uncore, dram) are nested in the package
        with open(f"{d}/name") as f:
            if not f.read().startswith("package"):
                continue  # psys covers the platform, not the package
        with open(f"{d}/max_energy_range_uj") as f:
            out.append((f"{d}/energy_uj", int(f.read())))
    return out


def coretemp_package_inputs(root: str = "/sys/class/hwmon") -> list[str]:
    out = []
    for h in sorted(glob.glob(f"{root}/hwmon*")):
        with open(f"{h}/name") as f:
            if f.read().strip() != "coretemp":
                continue
        for lab in sorted(glob.glob(f"{h}/temp*_label")):
            with open(lab) as f:
                if f.read().startswith("Package"):
                    out.append(lab.replace("_label", "_input"))
    return out


class Snapshot:
    """Open every source once; `read()` returns cumulative values + timing."""

    VM_KEYS = {"pgscan_kswapd", "pgscan_direct", "pgscan_khugepaged", "pgmajfault", "pswpout", "oom_kill"}

    def __init__(self, perf=None):
        self.perf = perf
        self.disks = physical_block_devices()
        self.nics = physical_nics()
        self.f_stat = _File("/proc/stat")
        self.f_psi = {k: _File(f"/proc/pressure/{k}") for k in ("cpu", "memory", "io")}
        self.f_mem = _File("/proc/meminfo")
        self.f_vm = _File("/proc/vmstat")
        self.f_disk = _File("/proc/diskstats")
        self.f_net = _File("/proc/net/dev")
        self.f_snmp = _File("/proc/net/snmp")
        self.ioerr = [_File(p) for d in self.disks
                      if os.path.exists(p := f"/sys/block/{d}/device/ioerr_cnt")]
        self.rapl = [(_File(p), rng) for p, rng in rapl_packages()]
        self.temps = [_File(p) for p in coretemp_package_inputs()]
        self.throttle = [_File(p) for p in throttle_files()]
        missing = [n for n, v in (("RAPL package", self.rapl), ("coretemp package", self.temps)) if not v]
        if missing:
            raise RuntimeError(f"missing sources: {', '.join(missing)}")
        self.clk_tck = os.sysconf("SC_CLK_TCK")

    def meta(self) -> dict:
        return {"disks": self.disks, "nics": self.nics, "clk_tck": self.clk_tck,
                "rapl": [f.path for f, _ in self.rapl], "temps": [f.path for f in self.temps],
                "throttle_files": len(self.throttle), "ioerr_disks": len(self.ioerr),
                "ncpu": os.cpu_count(), "pmu": self.perf is not None}

    def read(self) -> dict:
        t0 = time.monotonic_ns()
        c: dict = {}
        pmu, run = self.perf.delta() if self.perf else ({}, float("nan"))
        c.update(parse_stat(self.f_stat.text()))
        c["psi_cpu_us"] = _psi_some_total(self.f_psi["cpu"].text())
        c["psi_mem_us"] = _psi_some_total(self.f_psi["memory"].text())
        c["psi_io_us"] = _psi_some_total(self.f_psi["io"].text())
        m = parse_kv(self.f_mem.text(), {"MemTotal", "MemAvailable"})
        c["mem_used_frac"] = 1.0 - m["MemAvailable"] / m["MemTotal"]
        v = parse_kv(self.f_vm.text(), self.VM_KEYS)
        c["vm_pgscan"] = v.get("pgscan_kswapd", 0) + v.get("pgscan_direct", 0) + v.get("pgscan_khugepaged", 0)
        c["vm_pgmajfault"], c["vm_pswpout"], c["vm_oom_kill"] = v["pgmajfault"], v["pswpout"], v["oom_kill"]
        c["disks"] = parse_diskstats(self.f_disk.text(), set(self.disks))
        c["disk_ioerr"] = sum(int(f.text().strip(), 0) for f in self.ioerr)
        c.update(parse_net_dev(self.f_net.text(), set(self.nics)))
        c["tcp_retrans"] = parse_snmp_tcp(self.f_snmp.text())
        c["rapl"] = [int(f.text()) for f, _ in self.rapl]
        c["pkg_temp_c"] = max(int(f.text()) for f in self.temps) / 1000.0
        c["throttle"] = sum(int(f.text()) for f in self.throttle)
        t1 = time.monotonic_ns()
        c["mono_ns"] = (t0 + t1) // 2
        c["read_us"] = (t1 - t0) / 1e3
        c["pmu"], c["pmu_running"] = pmu, run
        return c


COUNTERS = ("cpu_total_j", "cpu_busy_j", "cpu_sys_j", "psi_cpu_us", "psi_mem_us", "psi_io_us", "ctxt",
            "forks", "vm_pgscan", "vm_pgmajfault", "vm_pswpout", "vm_oom_kill", "disk_ioerr",
            "net_rx_bytes", "net_rx_pkts", "net_tx_bytes", "net_tx_pkts", "net_drop_err",
            "tcp_retrans", "throttle")


class Ticker:
    """Consecutive snapshots -> tick records (dicts keyed by schema.TICK_FIELDS)."""

    def __init__(self, snap: Snapshot):
        self.snap = snap
        self.prev = snap.read()
        self._cpu_prev = self._cpu_s()

    @staticmethod
    def _cpu_s() -> float:
        r = resource.getrusage(resource.RUSAGE_SELF)
        return r.ru_utime + r.ru_stime

    def tick(self) -> dict:
        cur = self.snap.read()
        p = self.prev
        dt = (cur["mono_ns"] - p["mono_ns"]) / 1e9
        out: dict = {"dt_s": dt}
        for k in COUNTERS:
            d = cur[k] - p[k]
            out[k] = d if d >= 0 else None
        rd = wr = rio = wio = wq = 0
        busy = 0.0
        for name, s in cur["disks"].items():
            q = p["disks"].get(name)
            if q is None:
                continue
            dd = {k: s[k] - q[k] for k in s}
            if min(dd.values()) < 0:
                continue  # device reset/replaced this tick
            rd += dd["rd_sec"] * 512
            wr += dd["wr_sec"] * 512
            rio += dd["rd_ios"]
            wio += dd["wr_ios"]
            wq += dd["wq_ms"]
            busy = max(busy, min(dd["io_ticks"] / (1e3 * dt), 1.0))
        out.update(disk_rd_bytes=rd, disk_wr_bytes=wr, disk_rd_ios=rio, disk_wr_ios=wio,
                   disk_wq_ms=wq, disk_busy_max=busy)
        e = 0
        for (now, before), (_, rng) in zip(zip(cur["rapl"], p["rapl"]), self.snap.rapl):
            e += now - before if now >= before else now + rng - before
        out["pkg_energy_j"] = e / 1e6
        out["mem_used_frac"] = cur["mem_used_frac"]
        out["pkg_temp_c"] = cur["pkg_temp_c"]
        out.update(cur["pmu"])
        out["pmu_running"] = cur["pmu_running"]
        out["read_us"] = cur["read_us"]
        cpu = self._cpu_s()
        out["self_cpu_us"] = (cpu - self._cpu_prev) * 1e6
        self._cpu_prev = cpu
        self.prev = cur
        return out
