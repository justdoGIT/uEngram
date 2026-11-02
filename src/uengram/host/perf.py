"""System-wide hardware counters through perf_event_open(2); stdlib only.

One event group per online CPU, leader `cycles`, read with
PERF_FORMAT_GROUP | TOTAL_TIME_ENABLED | TOTAL_TIME_RUNNING. A group is
scheduled onto the PMU as a unit, so ratios formed inside it (ipc, freq_ratio,
MPKI) come from co-temporal counts. Five events fit without multiplexing on
Ivy Bridge..Kaby Lake: instructions and ref-cycles use fixed counters, the
rest use 3 of the 4 programmable counters available per SMT thread, which
leaves room when the NMI watchdog holds the fixed cycles counter and cycles
falls back to a programmable one.

If the kernel multiplexes anyway, each interval's delta is scaled by
d(time_enabled)/d(time_running) per CPU (an estimate; Mytkowicz et al. 2007)
and the smallest running fraction is reported so windows can be judged.

System-wide counting needs perf_event_paranoid <= 0 or CAP_PERFMON
(Documentation/admin-guide/perf-security.rst); the runit service grants
CAP_PERFMON with setpriv instead of lowering the sysctl.
"""
from __future__ import annotations

import ctypes
import os
import platform
import struct

_SYSCALL_NR = {"x86_64": 298, "aarch64": 241}
PERF_TYPE_HARDWARE = 0
# name -> (PERF_COUNT_HW_* config, tick field)
EVENTS = (
    ("cycles", 0, "pmu_cyc"),  # group leader
    ("instructions", 1, "pmu_inst"),
    ("ref-cycles", 9, "pmu_ref"),
    ("branch-misses", 5, "pmu_brmiss"),
    ("cache-misses", 3, "pmu_llcmiss"),  # LONGEST_LAT_CACHE.MISS on Intel cores
)
_FMT_ENABLED, _FMT_RUNNING, _FMT_GROUP = 1, 2, 8
_FLAG_EXCLUDE_KERNEL, _FLAG_EXCLUDE_HV = 1 << 5, 1 << 6
_FD_CLOEXEC = 8


class PerfUnavailable(RuntimeError):
    pass


class _Attr(ctypes.Structure):  # struct perf_event_attr, PERF_ATTR_SIZE_VER5
    _fields_ = [
        ("type", ctypes.c_uint32), ("size", ctypes.c_uint32), ("config", ctypes.c_uint64),
        ("sample_period", ctypes.c_uint64), ("sample_type", ctypes.c_uint64),
        ("read_format", ctypes.c_uint64), ("flags", ctypes.c_uint64),
        ("wakeup_events", ctypes.c_uint32), ("bp_type", ctypes.c_uint32),
        ("config1", ctypes.c_uint64), ("config2", ctypes.c_uint64),
        ("branch_sample_type", ctypes.c_uint64), ("sample_regs_user", ctypes.c_uint64),
        ("sample_stack_user", ctypes.c_uint32), ("clockid", ctypes.c_int32),
        ("sample_regs_intr", ctypes.c_uint64), ("aux_watermark", ctypes.c_uint32),
        ("sample_max_stack", ctypes.c_uint16), ("reserved_2", ctypes.c_uint16),
    ]


assert ctypes.sizeof(_Attr) == 112

_libc = ctypes.CDLL(None, use_errno=True)


def _open(config: int, pid: int, cpu: int, group_fd: int, flags: int) -> int:
    nr = _SYSCALL_NR.get(platform.machine())
    if nr is None:
        raise PerfUnavailable(f"no perf_event_open syscall number for {platform.machine()}")
    attr = _Attr(type=PERF_TYPE_HARDWARE, size=ctypes.sizeof(_Attr), config=config,
                 read_format=_FMT_GROUP | _FMT_ENABLED | _FMT_RUNNING, flags=flags)
    fd = _libc.syscall(ctypes.c_long(nr), ctypes.byref(attr), ctypes.c_int(pid), ctypes.c_int(cpu),
                       ctypes.c_int(group_fd), ctypes.c_ulong(_FD_CLOEXEC))
    if fd < 0:
        err = ctypes.get_errno()
        raise PerfUnavailable(f"perf_event_open(config={config}, cpu={cpu}): {os.strerror(err)}"
                              + (" (needs CAP_PERFMON or perf_event_paranoid <= 0)"
                                 if err in (1, 13) else ""))
    return fd


def online_cpus(path: str = "/sys/devices/system/cpu/online") -> list[int]:
    out: list[int] = []
    with open(path) as f:
        for part in f.read().strip().split(","):
            a, _, b = part.partition("-")
            out += list(range(int(a), int(b or a) + 1))
    return out


class PerfCounters:
    """`system=True`: every online CPU, user + kernel. `system=False`: the
    calling process only, user space only (allowed at perf_event_paranoid 2;
    used by tests)."""

    def __init__(self, system: bool = True):
        self.cpus = online_cpus() if system else [-1]
        pid = -1 if system else 0
        flags = _FLAG_EXCLUDE_HV | (0 if system else _FLAG_EXCLUDE_KERNEL)
        self.leaders: list[int] = []
        self._all: list[int] = []
        try:
            for cpu in self.cpus:
                lead = _open(EVENTS[0][1], pid, cpu, -1, flags)
                self._all.append(lead)
                for _, cfg, _ in EVENTS[1:]:
                    self._all.append(_open(cfg, pid, cpu, lead, flags))
                self.leaders.append(lead)
        except PerfUnavailable:
            self.close()
            raise
        self._size = 8 * (3 + len(EVENTS))
        self._prev = [self._read(fd) for fd in self.leaders]

    def _read(self, fd: int) -> tuple[int, int, tuple[int, ...]]:
        buf = os.read(fd, self._size)
        nr, enabled, running, *vals = struct.unpack(f"<{3 + len(EVENTS)}Q", buf)
        if nr != len(EVENTS):
            raise PerfUnavailable(f"group read returned {nr} events")
        return enabled, running, tuple(vals)

    def delta(self) -> tuple[dict[str, float], float]:
        """Counts since the previous call summed over CPUs (scaled where
        multiplexed), and the smallest running fraction over CPUs."""
        tot = [0.0] * len(EVENTS)
        min_run = 1.0
        for i, fd in enumerate(self.leaders):
            cur = self._read(fd)
            pe, pr, pv = self._prev[i]
            de, dr = cur[0] - pe, cur[1] - pr
            frac = dr / de if de > 0 else 0.0
            min_run = min(min_run, frac)
            scale = de / dr if dr > 0 else 0.0
            for k in range(len(EVENTS)):
                tot[k] += (cur[2][k] - pv[k]) * scale
            self._prev[i] = cur
        return {field: tot[k] for k, (_, _, field) in enumerate(EVENTS)}, min_run

    def close(self) -> None:
        for fd in self._all:
            try:
                os.close(fd)
            except OSError:
                pass
        self._all, self.leaders = [], []
