"""Host-state schema: what one tick measures and how each attribute is windowed.

Three layers, each checked by `validate()`:

* tick fields: what the collector emits once per second. Counters are emitted
  as deltas over the measured interval `dt_s`; gauges as their value at the end
  of the tick. Quality fields describe the measurement, not the machine.
* attributes: window-level state variables. Every attribute belongs to exactly
  one modality and *owns* its numerator fields; no field is owned twice, so no
  physical quantity enters the state vector through two modalities. Shared
  denominators (`dt_s`, CPU time, instructions) are normalisers, not copies.
* exclusions: signals that exist on these hosts but are deliberately not
  attributes, each with the reason (identity with kept attributes,
  pre-smoothed, unreliable, or nested inside a kept quantity).

This module is stdlib-only: the collector runs on nodes without numpy.
"""
from __future__ import annotations

from dataclasses import dataclass

MODALITIES = ("compute", "memory", "storage", "network", "power_thermal")

# name -> (unit, role); role is "counter" (delta per tick), "gauge" (value at
# tick end), "norm" (interval length), "quality" (measurement diagnostics)
TICK_FIELDS: dict[str, tuple[str, str]] = {
    "dt_s": ("s, CLOCK_MONOTONIC", "norm"),
    "cpu_total_j": ("USER_HZ ticks, all CPUs", "counter"),
    "cpu_busy_j": ("USER_HZ ticks: user+nice+system+irq+softirq", "counter"),
    "cpu_sys_j": ("USER_HZ ticks: system+irq+softirq", "counter"),
    "psi_cpu_us": ("us, /proc/pressure/cpu some total", "counter"),
    "ctxt": ("context switches", "counter"),
    "forks": ("processes created", "counter"),
    "pmu_inst": ("instructions retired (all CPUs)", "counter"),
    "pmu_cyc": ("unhalted core cycles", "counter"),
    "pmu_ref": ("unhalted reference (TSC-rate) cycles", "counter"),
    "pmu_brmiss": ("mispredicted branches", "counter"),
    "pmu_llcmiss": ("last-level cache misses", "counter"),
    "mem_used_frac": ("1 - MemAvailable/MemTotal", "gauge"),
    "psi_mem_us": ("us, /proc/pressure/memory some total", "counter"),
    "vm_pgscan": ("pages scanned by reclaim (kswapd+direct+khugepaged)", "counter"),
    "vm_pgmajfault": ("major page faults (file or swap read)", "counter"),
    "vm_pswpout": ("pages swapped out", "counter"),
    "vm_oom_kill": ("OOM kills", "counter"),
    "disk_rd_bytes": ("bytes read, physical disks", "counter"),
    "disk_wr_bytes": ("bytes written, physical disks", "counter"),
    "disk_rd_ios": ("reads completed", "counter"),
    "disk_wr_ios": ("writes completed", "counter"),
    "disk_wq_ms": ("weighted ms with I/O in flight (sum over disks)", "counter"),
    "disk_busy_max": ("max over disks of io_ticks/dt", "gauge"),
    "psi_io_us": ("us, /proc/pressure/io some total", "counter"),
    "disk_ioerr": ("SCSI ioerr_cnt increments", "counter"),
    "net_rx_bytes": ("bytes received, physical NICs", "counter"),
    "net_tx_bytes": ("bytes sent, physical NICs", "counter"),
    "net_rx_pkts": ("packets received", "counter"),
    "net_tx_pkts": ("packets sent", "counter"),
    "tcp_retrans": ("TCP segments retransmitted", "counter"),
    "net_drop_err": ("rx/tx errors + drops, physical NICs", "counter"),
    "pkg_energy_j": ("J, RAPL package domain", "counter"),
    "pkg_temp_c": ("degC, coretemp package sensor", "gauge"),
    "throttle": ("package + core thermal throttle events", "counter"),
    "pmu_running": ("min over CPUs of time_running/time_enabled", "quality"),
    "read_us": ("wall time to read all sources", "quality"),
    "self_cpu_us": ("collector CPU time this tick", "quality"),
}

# Window-level transforms ("kind"):
#   fraction        asin(sqrt(sum num * scale / sum den))       angular, defined on [0,1]
#   gauge_fraction  asin(sqrt(mean of the per-tick gauge))
#   log_ratio       log(sum num * scale / sum den)              ratio of sums, never mean of ratios
#   log1p_rate      log1p(sum num * scale / sum dt)             per second
#   log1p_count     log1p(sum num)                              events per window
#   log_rate        log(sum num / sum dt)                       strictly positive rates
#   residual        window mean of a physics-model residual     (thermal, see encoder)
#   event_count     raw count per window; NOT Gaussian-modelled (see below)
KINDS = ("fraction", "gauge_fraction", "log_ratio", "log1p_rate", "log1p_count",
         "log_rate", "residual", "event_count")
# Every attribute is handled by exactly one statistical model:
# * Gaussian latent (uEngram engine): all kinds except event_count. These are
#   continuous, or counts of activity whose magnitude and dependence on the
#   machine state matter (page scans, retransmits).
# * Exact count test: event_count attributes are error counters that are zero
#   almost always. A Gaussian score is not applicable to them: their robust
#   variance is zero, a single event would be either invisible (one noise unit
#   among 40 dimensions) or infinitely surprising. They are tested with the
#   Poisson-Gamma predictive tail fitted on commissioning (node.EventModel).


@dataclass(frozen=True)
class Attribute:
    name: str
    modality: str
    kind: str
    num: tuple[str, ...]  # owned fields, summed
    den: str | None = None  # normaliser field (not owned)
    scale: float = 1.0
    floor: float = 0.05  # smallest meaningful change of the level, used as the minimum robust scale
    drive: tuple[str, ...] = ()  # exogenous model inputs (residual kind only), not owned
    note: str = ""


ANG_FLOOR = 0.0316  # asin(sqrt(1e-3)): 0.1 % of time
ATTRIBUTES: tuple[Attribute, ...] = (
    # compute: utilisation, saturation, scheduler activity, pipeline quality
    Attribute("cpu_busy", "compute", "fraction", ("cpu_busy_j",), "cpu_total_j", floor=ANG_FLOOR,
              note="USE utilisation; iowait and steal count as not busy"),
    Attribute("cpu_sys_share", "compute", "fraction", ("cpu_sys_j",), "cpu_busy_j", floor=ANG_FLOOR,
              note="kernel share of busy time"),
    Attribute("cpu_pressure", "compute", "fraction", ("psi_cpu_us",), "dt_s", 1e-6, ANG_FLOOR,
              note="USE saturation: runnable but waiting"),
    Attribute("ctx_switch_rate", "compute", "log1p_rate", ("ctxt",)),
    Attribute("fork_rate", "compute", "log1p_rate", ("forks",)),
    Attribute("ipc", "compute", "log_ratio", ("pmu_inst",), "pmu_cyc", floor=0.01),
    Attribute("freq_ratio", "compute", "log_ratio", ("pmu_cyc",), "pmu_ref", floor=0.01,
              note="effective/nominal clock while unhalted; turbo > 1, throttling < 1"),
    Attribute("branch_mpki", "compute", "log_ratio", ("pmu_brmiss",), "pmu_inst", 1e3, 0.01),
    Attribute("llc_mpki", "compute", "log_ratio", ("pmu_llcmiss",), "pmu_inst", 1e3, 0.01,
              note="DRAM-bound intensity per instruction"),
    # memory
    Attribute("mem_used", "memory", "gauge_fraction", ("mem_used_frac",), floor=ANG_FLOOR),
    Attribute("mem_pressure", "memory", "fraction", ("psi_mem_us",), "dt_s", 1e-6, ANG_FLOOR),
    Attribute("page_scan", "memory", "log1p_count", ("vm_pgscan",), floor=0.693),
    Attribute("major_faults", "memory", "log1p_count", ("vm_pgmajfault",), floor=0.693,
              note="includes swap-in, so pswpin is excluded"),
    Attribute("swap_out", "memory", "log1p_count", ("vm_pswpout",), floor=0.693),
    Attribute("oom_kills", "memory", "event_count", ("vm_oom_kill",)),
    # storage
    Attribute("disk_read_bps", "storage", "log1p_rate", ("disk_rd_bytes",)),
    Attribute("disk_write_bps", "storage", "log1p_rate", ("disk_wr_bytes",)),
    Attribute("disk_read_iops", "storage", "log1p_rate", ("disk_rd_ios",)),
    Attribute("disk_write_iops", "storage", "log1p_rate", ("disk_wr_ios",)),
    Attribute("disk_queue", "storage", "log1p_rate", ("disk_wq_ms",), scale=1e-3,
              note="time-average requests in flight (Little's law L)"),
    Attribute("disk_busy", "storage", "gauge_fraction", ("disk_busy_max",), floor=ANG_FLOOR,
              note="busiest device; saturates early on parallel devices"),
    Attribute("io_pressure", "storage", "fraction", ("psi_io_us",), "dt_s", 1e-6, ANG_FLOOR,
              note="includes NFS waits"),
    Attribute("disk_errors", "storage", "event_count", ("disk_ioerr",)),
    # network
    Attribute("net_rx_bps", "network", "log1p_rate", ("net_rx_bytes",)),
    Attribute("net_tx_bps", "network", "log1p_rate", ("net_tx_bytes",)),
    Attribute("net_rx_pps", "network", "log1p_rate", ("net_rx_pkts",)),
    Attribute("net_tx_pps", "network", "log1p_rate", ("net_tx_pkts",)),
    Attribute("tcp_retrans", "network", "log1p_count", ("tcp_retrans",), floor=0.693),
    Attribute("net_faults", "network", "event_count", ("net_drop_err",)),
    # power and thermal
    Attribute("pkg_power", "power_thermal", "log_rate", ("pkg_energy_j",), "dt_s", floor=0.01),
    Attribute("thermal_residual", "power_thermal", "residual", ("pkg_temp_c",), floor=0.1,
              drive=("pkg_energy_j", "dt_s"),
              note="package temperature minus the RC-model prediction from package power"),
    Attribute("throttle_events", "power_thermal", "event_count", ("throttle",)),
)


@dataclass(frozen=True)
class BurstBlock:
    """Second-order feature of one modality's activity series: the robust
    tick-to-tick noise scale, log(1.4826 * MAD(diff(s)) / sqrt(2)).

    Periodogram band energies (uEngram's DSP encoder) are not used: they
    assume a stationary window, and host activity is step-like (jobs start
    and stop), so a single step inside a window dominates every band with its
    1/f^2 spectrum whatever the process does. One step changes one of K-1
    differences and leaves their median in place. (In the DSP encoder the log
    std is also a Parseval function of the band energies.)"""
    name: str
    modality: str
    num: tuple[str, ...]
    den: str  # per-tick denominator
    transform: str  # "angular" or "log1p"
    scale: float = 1.0


BURST: tuple[BurstBlock, ...] = (
    BurstBlock("cpu_busy", "compute", ("cpu_busy_j",), "cpu_total_j", "angular"),
    BurstBlock("disk_bytes", "storage", ("disk_rd_bytes", "disk_wr_bytes"), "dt_s", "log1p"),
    BurstBlock("net_bytes", "network", ("net_rx_bytes", "net_tx_bytes"), "dt_s", "log1p"),
)
BURST_FLOOR = 0.05

# Outcome metrics are reported per window and per learned state; they are not
# encoded, because each is a product of kept attributes (no new information).
OUTCOMES = {
    "ips": "instructions per second = ipc * freq_ratio * busy * ncpu * f_nominal",
    "pkg_watts": "package power, W",
    "nj_per_inst": "package energy per instruction, nJ",
}

EXCLUDED: dict[str, str] = {
    "/proc/loadavg": "1/5/15-min exponential averages of runnable + D-state tasks: pre-smoothed "
                     "(window statistics would describe the filter) and a mix of cpu_pressure and io_pressure",
    "PSI avg10/avg60/avg300": "pre-smoothed; the total counters are used instead",
    "PSI full lines": "near-duplicate of 'some' (full <= some); cpu full is 0 by definition system-wide",
    "/proc/pressure/irq": "its utilisation counterpart (irq+softirq time) is inside cpu_sys_share",
    "/proc/stat iowait": "documented as unreliable (proc(5)); IO stall is measured by io_pressure",
    "/proc/stat intr, softirq counts": "driven by disk completions, packets and timers already counted",
    "procs_running, procs_blocked": "instantaneous samples of the queues that PSI integrates",
    "scaling_cur_freq, aperf/mperf": "duplicate of freq_ratio (cycles / ref-cycles)",
    "ref-cycles rate": "duplicate of cpu_busy (unhalted time at nominal rate)",
    "instructions per second": "log ips = log ipc + log freq_ratio + log busy + const: a linear identity",
    "cache-references, LLC miss ratio": "second normalisation of LLC traffic already in llc_mpki",
    "topdown level-1 slots": "needs 4 programmable counters, and the NMI watchdog takes fixed counter 1 "
                             "on 4 of 5 nodes; ipc and branch_mpki carry retiring and bad speculation",
    "uncore_imc DRAM bandwidth": "~ llc_mpki * ips * 64 B, a combination of kept attributes",
    "RAPL core/uncore/dram domains": "nested inside the package domain or a function of DRAM traffic",
    "per-core temperatures, raw package temperature": "explained by package power through the thermal RC model;"
                                                      " only its residual is new information",
    "fan rpm, battery": "present on 2 and 1 of 5 nodes; cooling faults appear in thermal_residual",
    "pswpin": "counted again by pgmajfault (swap-in faults are major faults)",
    "pgsteal": "bounded by and coupled to page_scan",
    "MemFree, Cached, Buffers, SwapFree": "components or integrals of mem_used and swap_out",
    "disk latency (ticks/ios), request size": "Little's law / byte identities with disk_queue, iops and bps",
    "ios in progress": "instantaneous sample of what disk_queue time-averages",
    "TCP InSegs/OutSegs": "mostly the NIC packet counts again",
    "C-state residency": "its energy consequence is in pkg_power; idle share is 1 - cpu_busy",
}


def owned_fields() -> dict[str, str]:
    """field -> owning attribute; raises if a field is owned twice."""
    owner: dict[str, str] = {}
    for a in ATTRIBUTES:
        for f in a.num:
            if f in owner:
                raise ValueError(f"field {f} owned by {owner[f]} and {a.name}")
            owner[f] = a.name
    return owner


def validate() -> None:
    """Structural checks; tests/test_host_schema.py adds the numerical rank test."""
    names = [a.name for a in ATTRIBUTES]
    if len(set(names)) != len(names):
        raise ValueError("duplicate attribute names")
    owner = owned_fields()
    by_name = {a.name: a for a in ATTRIBUTES}
    for a in ATTRIBUTES:
        if a.modality not in MODALITIES or a.kind not in KINDS:
            raise ValueError(f"{a.name}: bad modality/kind")
        for f in (*a.num, *((a.den,) if a.den else ()), *a.drive):
            if f not in TICK_FIELDS:
                raise ValueError(f"{a.name}: unknown field {f}")
        if a.kind in ("fraction", "log_ratio", "log_rate") and a.den is None:
            raise ValueError(f"{a.name}: {a.kind} needs a denominator")
    for f, (_, role) in TICK_FIELDS.items():
        used = f in owner or any(f == a.den or f in a.drive for a in ATTRIBUTES)
        if role in ("counter", "gauge", "norm") and not used:
            raise ValueError(f"field {f} is measured but used by no attribute")
    for b in BURST:
        for f in b.num:
            if by_name[owner[f]].modality != b.modality:
                raise ValueError(f"burst block {b.name} reads {f} from another modality")


def gaussian_attributes() -> list[Attribute]:
    return [a for a in ATTRIBUTES if a.kind != "event_count"]


def event_attributes() -> list[Attribute]:
    return [a for a in ATTRIBUTES if a.kind == "event_count"]


def feature_names() -> list[str]:
    """Order of the Gaussian feature vector produced by `features.windows`."""
    out = [a.name for a in gaussian_attributes()]
    out += [f"{b.name}.burst" for b in BURST]
    return out


def feature_modality() -> list[str]:
    mod = [a.modality for a in gaussian_attributes()]
    mod += [b.modality for b in BURST]
    return mod


def feature_floors() -> list[float]:
    """Smallest meaningful change per feature (minimum robust scale and noise floor)."""
    return [a.floor for a in gaussian_attributes()] + [BURST_FLOOR] * len(BURST)
