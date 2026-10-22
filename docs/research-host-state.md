# Accurate snapshots of running Linux computers

## 1. Signal type, units, and differencing

A 1 Hz collector should retain a monotonic-clock timestamp around every read and calculate counter rates with the measured interval, not assume exactly one second:
\[
\Delta C=C_t-C_{t-1},\qquad r=\Delta C/\Delta t,\quad \Delta t=(t^{mono}_t-t^{mono}_{t-1})\text{ seconds}.
\]
For wraparound counters of known modulus M use \((C_t-C_{t-1})\bmod M\) only if a single wrap is physically plausible; distinguish reset/device replacement from wrap, and mark intervals invalid if multiple wraps cannot be excluded. The Linux monotonic clock does not jump with wall-clock adjustments. [clock_gettime(2)](https://man7.org/linux/man-pages/man2/clock_gettime.2.html); [Linux procfs stat documentation](https://docs.kernel.org/filesystems/proc.html)

| Source / field | Type and correct handling | Units / granularity |
|---|---|---|
| `/proc/stat`: CPU times, `intr`, `ctxt`, `processes`, `procs_running`, `procs_blocked`, `softirq` | CPU-time fields and cumulative event fields are counters; difference and divide by actual dt. Running/blocked are instantaneous counts. | CPU times in USER_HZ ticks; query `sysconf(_SC_CLK_TCK)`, do not assume 100. Interrupt/context-switch/fork values are cumulative counts. [proc(5)](https://man7.org/linux/man-pages/man5/proc.5.html); [procfs documentation](https://docs.kernel.org/filesystems/proc.html) |
| `/proc/vmstat` | Most `pg*`, `pswp*`, `pgfault`, `pgmajfault`, `allocstall*`, and `oom_kill`-type event fields are cumulative counters; memory state fields are not rates. Interpret each named field according to its kernel definition. | Counts / pages (page size is system-specific). [procfs documentation](https://docs.kernel.org/filesystems/proc.html) |
| `/proc/meminfo` | Gauges (available/free/cached/etc.), not counters; do not difference as though they represented throughput. | kB as reported (kernel convention is KiB). [procfs documentation](https://docs.kernel.org/filesystems/proc.html) |
| `/proc/diskstats` | Read/write completed/merged counts, sectors, and cumulative read/write/IO/weighted time are counters. Difference, then normalize by actual dt. | Time fields milliseconds; sectors are 512-byte sectors (regardless of device logical block size); weighted time is accumulated I/O time weighted by in-flight count. [kernel iostats](https://docs.kernel.org/admin-guide/iostats.html) |
| `/proc/net/dev`, `/proc/net/snmp` | RX/TX bytes, packets, errors, drops and protocol counters are cumulative; difference for interval rates. | Bytes and packet/event counts. [kernel networking statistics](https://docs.kernel.org/networking/statistics.html); [procfs documentation](https://docs.kernel.org/filesystems/proc.html) |
| `/proc/pressure/{cpu,memory,io}` | `total` is cumulative stalled microseconds; use delta / measured dt. `avg10`, `avg60`, `avg300` are already exponentially smoothed percentages, not instantaneous samples. | `total` in µs; average values in percent over EWMA timescales 10, 60, 300 seconds. [PSI documentation](https://docs.kernel.org/accounting/psi.html) |
| `/proc/loadavg` | 1/5/15-minute exponentially damped runnable-task averages plus instantaneous running/total process counts and last PID. The averages are not instantaneous load gauges. | Load average is dimensionless task count; process counts. [procfs documentation](https://docs.kernel.org/filesystems/proc.html) |
| RAPL `energy_uj` | Monotonic energy counter until wrap/reset; compute power from energy delta / actual dt. `max_energy_range_uj` gives modulus/range for wrap handling. | µJ. [Linux powercap documentation](https://docs.kernel.org/power/powercap/powercap.html) |
| coretemp `temp*_input` | Temperature gauge: the current sensor temperature (after driver conversion), not an accumulated count. | millidegrees Celsius in hwmon sysfs; sensor resolution 1°C. [coretemp documentation](https://docs.kernel.org/hwmon/coretemp.html); [hwmon sysfs interface](https://docs.kernel.org/hwmon/sysfs-interface.html) |
| perf events | Hardware/software event counts are counters; use `read()` values and enabled/running times when multiplexed. | Event-specific counts / cycles / instructions. [perf_event_open(2)](https://man7.org/linux/man-pages/man2/perf_event_open.2.html) |

A window over an EWMA series estimates variation of the filter output, not variation of the original pressure/load signal: smoothing induces serial correlation and suppresses high-frequency changes, so ordinary sample variance, autocorrelation, or confidence estimates assuming raw observations are wrong. Prefer deltas of PSI `total` (and interval counters); use smoothed fields only as explicitly slow context or model their filter state. [PSI documentation](https://docs.kernel.org/accounting/psi.html); [procfs load average description](https://docs.kernel.org/filesystems/proc.html)

## 2. `perf_event_open`: access, multiplexing, and counter capacity

Linux `perf_event_paranoid` policy is: `-1` no perf-specific scope/access restrictions; `0` disallow raw/ftrace tracepoint access to unprivileged users; `1` disallow CPU event access to unprivileged users; `2` disallow kernel profiling to unprivileged users (and more restrictive values may be configured). Thus unprivileged **system-wide per-CPU CPU-event counting requires level 0 or -1**; `CAP_PERFMON` grants privileged perf-monitoring access without broad `CAP_SYS_ADMIN`. Check the running kernel's documentation/configuration; distro policy can be stricter. [kernel perf security docs](https://docs.kernel.org/admin-guide/perf-security.html); [perf_event_open(2)](https://man7.org/linux/man-pages/man2/perf_event_open.2.html)

When requested events exceed available PMCs, the kernel time-multiplexes them. `time_enabled` is time event was enabled; `time_running` is time it actually occupied a counter. The usual scaled estimate is
\[
\hat C=C_{observed}\frac{t_{enabled}}{t_{running}}.
\]
The scaling assumes the event rate during running and not-running portions is representative. It is an estimate, not recovered exact count: phase changes, workload bursts, and interactions can cause substantial error; short epochs or low running fraction worsen it. Include both timing fields in each sample; reject or flag zero/low running fraction rather than silently treating scaled values as exact. [perf_event_open(2)](https://man7.org/linux/man-pages/man2/perf_event_open.2.html); [Mytkowicz et al., MICRO 2007](https://doi.org/10.1109/MICRO.2007.27); [Weaver et al., ISPASS 2013](https://web.eece.maine.edu/~vweaver/projects/deterministic/ispass2013_deterministic.pdf)

Put events whose ratios matter in one perf event group: group scheduling aims to run leader and members together, allowing ratios to be formed from co-temporal counts. A group is scheduled only when all members can be scheduled; too-large groups can remain unscheduled. Grouping does not eliminate event skid, architectural errata, SMT interference, or inherent non-determinism. [perf_event_open(2), groups](https://man7.org/linux/man-pages/man2/perf_event_open.2.html); [Weaver et al. 2013](https://web.eece.maine.edu/~vweaver/projects/deterministic/ispass2013_deterministic.pdf)

Skylake-family (and Sandy Bridge onward) core PMUs have three fixed counters per logical processor (instructions, unhalted cycles, unhalted reference cycles) and eight programmable counters per physical core, split as four per logical processor when SMT is enabled and eight when it is disabled ([Intel PerfMon event notes: "the four (eight when Hyperthreading is disabled) programmable counters"](https://perfmon-events.intel.com/platforms/skylakex/core-events/core/); [Intel community answer](https://community.intel.com/t5/Software-Tuning-Performance/hardware-performance-counters/m-p/1181572)). [Correction: the first draft of this note said the four programmable counters are shared by SMT siblings; that is wrong.] The NMI watchdog (`kernel.nmi_watchdog=1`, set on 4 of the 5 nodes) holds a pinned per-CPU cycles event, which takes fixed counter 1 when free, so a second cycles event moves to a programmable counter. `ref-cycles` can only use fixed counter 2: two groups that both need it cannot run at the same time (observed on msi: while the collector's system-wide group runs, a per-task group containing `ref-cycles` is never scheduled, `time_running` = 0). [perf x86 PMU source](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/arch/x86/events/intel/core.c)

## 3. Top-down analysis and compositional counters

Yasin's top-down method partitions pipeline slots into four level-1 categories. For counts over a common interval, let `total-slots` be the denominator and `slots-issued`, `slots-retired`, `fetch-bubbles`, `recovery-bubbles` be the kernel's topdown event counts:
\[
Retiring=slots\_retired/total\_slots;\quad FrontendBound=fetch\_bubbles/total\_slots;
\]
\[
BadSpeculation=(slots\_issued-slots\_retired+recovery\_bubbles)/total\_slots;
\]
\[
BackendBound=(total\_slots-fetch\_bubbles-slots\_issued-recovery\_bubbles)/total\_slots.
\]
The four fractions sum to 1 under the top-down accounting definitions, so they are compositional (only three free dimensions). Linux exports this event set for supported Sandy Bridge through Kaby Lake generations via sysfs topdown events; discover actual names/availability from the target PMU, because event names and implementation vary. [Yasin, ISPASS 2014](https://doi.org/10.1109/ISPASS.2014.6844459); [Linux x86 perf event definitions](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/arch/x86/events/intel/core.c)

The partition is a ratio of common slot counts, not four independent Gaussian features. Intel top-down accounting is most straightforward for a single-thread workload; SMT sibling activity shares core resources and can change attribution/interpretation, while some events count per logical processor and others per core. Avoid interpreting each sibling's partition as an isolated physical-core decomposition. [Intel top-down analysis documentation](https://www.intel.com/content/www/us/en/docs/vtune-profiler/cookbook/2023-0/top-down-microarchitecture-analysis-method.html); [Yasin 2014](https://doi.org/10.1109/ISPASS.2014.6844459)

Euclidean distance or arithmetic averaging of proportions ignores their closure constraint and can produce misleading covariance/relationships: increasing one part forces relative changes in others. For positive composition \(x=(x_1,\ldots,x_D)\), \(\sum x_i=1\), use log-ratio coordinates, e.g. additive log-ratio \(z_i=\log(x_i/x_D)\), or isometric log-ratio (orthonormal balances). There are \(D-1\) independent coordinates (three for four categories). Zero counts require a justified zero-replacement/model, not taking `log(0)`. [Aitchison, *The Statistical Analysis of Compositional Data* (1986)](https://doi.org/10.1007/978-94-009-4109-0); [Aitchison (1982)](https://doi.org/10.1111/j.2517-6161.1982.tb01195.x)

## 4. RAPL energy readings

RAPL exposes energy counters by power domain; average power over an interval is \(P_{avg}=\Delta E/\Delta t\), with energy in µJ and time in seconds yielding µW (divide by \(10^6\) for W). The reported `max_energy_range_uj` is the counter range; modulo-difference `energy_uj` at wrap, and keep dt sufficiently short that a domain cannot make multiple wraps between reads. [Linux powercap documentation](https://docs.kernel.org/power/powercap/powercap.html)

The energy unit is hardware-defined; package energy resolution/update cadence is approximately 1 ms in commonly studied Intel implementations, but read latency, quantization, platform firmware, domain, and CPU generation matter. Do not mistake high-rate repeated reads for independent high-resolution energy observations. RAPL domains commonly include package, `core`/PP0, uncore/PP1, DRAM, and sometimes platform/PSys; availability is model/platform-dependent. PP0 (cores) is contained in package, so package and PP0 are overlapping—not additive independent—measurements. [Hähnel et al. 2012](https://doi.org/10.1145/2425248.2425252); [Khan et al., RAPL in Action (2018)](https://doi.org/10.1145/3177754); [Linux powercap documentation](https://docs.kernel.org/power/powercap/powercap.html)

RAPL is an energy model/measurement interface rather than a universal externally calibrated wall-power meter: validation found domain- and workload-dependent accuracy, notably limitations in DRAM energy measurement. Treat readings as calibrated platform indicators and state the domain; validate against external power measurement when absolute accuracy matters. [Desrochers et al. 2016](https://doi.org/10.1145/2989081.2989088); [Khan et al. 2018](https://doi.org/10.1145/3177754)

Unprivileged access to RAPL energy was restricted following the Platypus side-channel disclosure (CVE-2020-8694); modern kernels commonly require root or an explicitly authorized access path for energy sysfs. This is a security policy, not a sensor defect. Linux perf also exposes power/energy PMU events when platform drivers register them; discover available events with `perf list` and check permissions rather than assuming availability. [Platypus paper](https://platypusattack.com/); [Linux powercap documentation](https://docs.kernel.org/power/powercap/powercap.html); [perf list manual](https://man7.org/linux/man-pages/man1/perf-list.1.html)

## 5. Core temperature is a lagged gauge

Intel coretemp reads the DTS, reports Celsius with 1°C resolution (sysfs millidegrees represent that integer-degree granularity), and caches sensor readings for 1 second in the driver. Consecutive 1 Hz reads can therefore repeat cached data; timestamp and do not treat repeats as newly independent physical samples. [coretemp documentation](https://docs.kernel.org/hwmon/coretemp.html); [coretemp driver source](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/drivers/hwmon/coretemp.c)

A lumped first-order thermal model is
\[
C\frac{dT}{dt}=P-\frac{T-T_{amb}}{R_{th}},\qquad \tau=R_{th}C.
\]
For constant power over sample interval \(\Delta t\), exact discrete response is
\[
T_{t+1}=T_{amb}+R_{th}P+(T_t-T_{amb}-R_{th}P)e^{-\Delta t/\tau}.
\]
Equivalently, with target \(T^*=T_{amb}+R_{th}P\), \(T_{t+1}=T_t+\alpha(T^*-T_t)\), \(\alpha=1-e^{-\Delta t/\tau}\). The RC form captures lag, not all spatial gradients, cooling controls, or sensor offset. [Skadron et al., HotSpot (2003)](https://doi.org/10.1109/HPCA.2003.1183537); [Skadron et al., HotSpot: A Dynamic Compact Thermal Model (2004)](https://doi.org/10.1109/TVLSI.2004.833888)

HotSpot and later thermal estimation work motivate using power inputs to predict die/package temperature; measured package power is useful as model input but requires thermal parameters, ambient/cooling context, and calibration. Energy delta gives interval-average power; do not map instantaneous 1 Hz package power directly to instantaneous coretemp without the thermal state model. [Skadron et al. 2003](https://doi.org/10.1109/HPCA.2003.1183537); [Hähnel et al. 2012](https://doi.org/10.1145/2425248.2425252)

## 6. PSI, disk queueing, and CPU iowait

PSI `some` measures time in which at least some tasks are stalled on a resource; `full` measures time in which all non-idle tasks are simultaneously stalled (system-wide productivity loss). "CPU full is undefined at the system level, but has been reported since 5.13, so it is set to zero for backward compatibility" (kernel docs) [correction: the first draft said CPU full is meaningful on modern kernels]. Prefer interval pressure from `Δtotal / (Δt × 10^6)` as a fraction; `avg10/60/300` are smoothed context, not additive interval measures; `total` is collected on read, so its delta is current. [PSI documentation](https://docs.kernel.org/accounting/psi.html)

For diskstats, over interval \(\Delta t\), Little's-law estimate of average outstanding I/O is
\[
\bar Q\approx\frac{\Delta weighted\_io\_ms}{1000\Delta t}.
\]
Mean completion latency for a matching operation class can be estimated as \(\Delta time\_in\_ms/\Delta completed\_ios\); `ticks/ios` is a per-I/O average when numerator and denominator describe the same class. `io_ticks` measures periods with I/O in progress, not additive utilization across parallel requests; concurrent requests make it a poor saturation/utilization estimate for SSD/NVMe. Prefer queue depth, weighted time, latency, throughput, and PSI together. [Linux iostats documentation](https://docs.kernel.org/admin-guide/iostats.html)

`/proc/stat` `iowait` is not a reliable measure of time a CPU waits for I/O: the kernel documents accounting ambiguities, including migration of tasks and possible decrease of the value. Do not use it as a direct CPU idle subtype or device utilization gauge. [procfs documentation](https://docs.kernel.org/filesystems/proc.html)

## 7. Prior art: representing and remembering system state

| Work | Representation / dependence handling | What it demonstrated |
|---|---|---|
| Cohen et al., OSDI 2004, “Correlating Instrumentation Data to System States” | Bayesian/tree-augmented Bayesian networks over performance metrics and expert-labeled system states; TAN structure models conditional dependencies rather than independent metric thresholds. | Small metric subsets could classify system states and support diagnosis/control, showing that correlated instrumentation patterns can identify causes. [OSDI paper](https://www.usenix.org/legacy/events/osdi04/tech/full_papers/cohen/cohen_html/index.html) |
| Cohen et al., SOSP 2005, “Capturing, Indexing, Clustering, and Retrieving System History” | Time-series system data organized into episodes, feature summaries, clusters, and retrieval indexes; metrics are correlated as multivariate history/state vectors rather than independent alarms. | System history can be captured, indexed, clustered, and retrieved to find analogous prior operating conditions and aid troubleshooting. [SOSP DOI](https://doi.org/10.1145/1095810.1095821) |
| Bodik et al., EuroSys 2010, “Fingerprinting the Datacenter” | Performance-counter / host-metric fingerprints, learned normal behavior and classification of performance crises; combines signals to distinguish classes. | Automated classification can detect and identify performance crises in datacenter services. [EuroSys DOI](https://doi.org/10.1145/1755913.1755926) |
| Zhang et al., EuroSys 2013, CPI² | CPI (cycles per instruction) from hardware counters as a compact workload-performance indicator; compares CPI performance isolation against shared-cluster interference. | CPI reveals interference/performance loss and informs isolation/throttling of noisy neighbors. [CPI² DOI](https://doi.org/10.1145/2465351.2465388) |
| Lo et al., ISCA 2015, Heracles | Joint workload/latency and resource-utilization signals with coordinated CPU, memory-bandwidth, and network resource controls; recognizes resource contention/correlation. | Coordinated resource management improved efficiency while meeting latency goals at scale. [Heracles DOI](https://doi.org/10.1145/2749469.2749475) |
| Gregg, USE method | Per-resource utilization, saturation, and errors rather than one aggregate host score; explicitly separates resource dimensions. | A practical systematic checklist helps find bottlenecks: utilization alone misses queues and errors. [Gregg, USE Method](https://www.brendangregg.com/usemethod.html) |

These works motivate storing time-aligned multivariate windows with normalized, typed signals, preserving counter rates and resource relationships; none implies that one fingerprint is a universal causal model. [Cohen et al. 2004](https://www.usenix.org/legacy/events/osdi04/tech/full_papers/cohen/cohen_html/index.html); [Bodik et al. 2010](https://doi.org/10.1145/1755913.1755926)

## 8. Multivariate SPC: retained and residual subspaces

For standardized centered observation \(x\), PCA model \(x=P_k t+e\) yields retained-score Hotelling statistic
\[
T^2=t'\Lambda_k^{-1}t
\]
for deviations within the retained principal-component subspace, and squared prediction error / Q statistic
\[
Q=e'e=\|x-P_kP_k'x\|^2
\]
for deviations orthogonal to that subspace. T² alone misses faults expressed only in discarded PCs; monitor both statistics with calibrated limits. [Jackson & Mudholkar (1979)](https://doi.org/10.1080/00401706.1979.10489779); [Kourti & MacGregor (1995)](https://doi.org/10.1016/0967-0661(95)80036-L)

Important failure case: a channel exactly constant during training has zero training variance and is normally removed/assigned no retained loading. At runtime it can drift while PCA's standardized transform has no valid scale, or if represented as a discarded/zero-variance direction its residual may be omitted by implementation; the change is then invisible to retained-space T² and potentially to computed Q. Preserve raw constants as explicit invariants/range checks (or model their measurement uncertainty), and validate residual monitoring with injected changes. [Jackson & Mudholkar 1979](https://doi.org/10.1080/00401706.1979.10489779); [Kourti & MacGregor 1995](https://three-mode.leidenuniv.nl/pdf/k/kourtimacgregor1995cils.pdf)

## 9. Rare counts are discrete

NIC errors/drops, thermal throttle events, and swap-in counts are nonnegative integer counts; under a constant-rate independent-event approximation, \(K\sim Poisson(\lambda\Delta t)\), with variance equal to mean. At low expected count, Gaussian z-scores have poor discrete tail calibration and can be undefined when estimated variance is zero. [NIST Poisson distribution reference](https://www.itl.nist.gov/div898/handbook/eda/section3/eda366j.htm)

Use exact Poisson/binomial-tail probabilities or randomized/ mid-p tail probabilities (mid-p assigns half the probability mass at the observed boundary) before normal-score transform \(z=\Phi^{-1}(F_{mid}(k))\); clip probabilities away from 0 and 1 in finite precision. The ordinary Poisson CDF transform is discrete and thus not exactly uniform; mid-p is a practical less-conservative adjustment, not an exact continuous PIT. For approximately Gaussian downstream modeling, Anscombe variance stabilization \(A(K)=2\sqrt{K+3/8}\) is a common Poisson transform; low counts remain discrete and should not be misrepresented as truly normal. [Lancaster (1961), mid-p](https://doi.org/10.1093/biomet/48.3-4.267); [Anscombe (1948)](https://doi.org/10.1093/biomet/35.3-4.246)

## 10. DVFS and energy efficiency

Instructions per joule \(I/E\) is a valid measured work-per-energy ratio only when the instruction count represents comparable useful work and energy scope includes the relevant system components. It is not a general cross-frequency efficiency score: spin loops retire instructions while doing no useful work, and memory-bound workloads may gain little from higher core frequency while power rises. State workload/completion and energy domain; report performance and energy jointly. [CPI² paper](https://doi.org/10.1145/2465351.2465388); [Le Sueur & Heiser, HotPower 2010](https://www.usenix.org/legacy/events/hotpower10/tech/full_papers/Le_Sueur.pdf)

Race-to-idle can save energy when finishing faster permits a sufficiently deep/long low-power idle interval; otherwise reduced frequency may lower active power yet increase execution time and total energy. Le Sueur and Heiser's measurements show diminishing returns and workload/platform dependence, not a universal winner. [Le Sueur & Heiser (2010)](https://www.usenix.org/legacy/events/hotpower10/tech/full_papers/Le_Sueur.pdf)

Dynamic CMOS switching power is often approximated as \(P_{dyn}=\alpha C V^2f\), with activity factor α, effective capacitance C, voltage V, frequency f. This is not total system power: leakage, uncore, memory, regulators, and voltage/frequency coupling matter; the approximation does not prove energy improves monotonically with f. [Le Sueur & Heiser (2010)](https://www.usenix.org/legacy/events/hotpower10/tech/full_papers/Le_Sueur.pdf)

## 11. Snapshot timing and cross-node consistency

Reads of separate `/proc` and sysfs files are not an atomic whole-machine snapshot: fields may be sampled at different points and may update during collection. Read monotonic timestamps before and after each file (or read group), retain the interval/midpoint and observed skew, and reject/mark windows whose skew exceeds the intended tolerance. For a monotonically accumulating field, the observation time lies within its read interval; timing bounds let downstream analysis distinguish real changes from collection skew. [procfs documentation](https://docs.kernel.org/filesystems/proc.html); [clock_gettime(2)](https://man7.org/linux/man-pages/man2/clock_gettime.2.html)

Chrony/NTP-synchronized wall clocks accurate to milliseconds are generally adequate to align 1 Hz windows across nodes when analysis windows are seconds long; retain monotonic local time for interval rates and wall time only for cross-node alignment. Verify actual offset/uncertainty rather than claiming a fixed guarantee. [chrony documentation](https://chrony-project.org/doc/4.6/chronyc.html)

Chandy–Lamport distributed snapshots solve a different problem: a consistent global cut of process state and in-flight messages in an asynchronous distributed system. It is unnecessary for independent periodic host telemetry windows unless causal consistency of inter-node messages/state is required; timestamp/skew bounds suffice for ordinary synchronized metrics. [Chandy & Lamport (1985)](https://doi.org/10.1145/214451.214456)
