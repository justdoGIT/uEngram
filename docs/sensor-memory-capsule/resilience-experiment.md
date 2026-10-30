# Capsule Resilience Experiment

## Claim under test

The capsule preserves uEngram recovery semantics across a process loss, and its durability
has a measurable cost. The study covers four questions: does a restarted detector diverge
from an uninterrupted run, does the capsule prevent that, does recovery survive real process
death, and what does durability cost on real storage. It is not an experiment of
anomaly-detection accuracy, fault lead time, field value, target-device latency, or flash
endurance.

## Fixed conditions

| Condition | Durable detector state | Event delivery | Historical retrieval |
|---|---:|---|---|
| `lite` | No | Direct Lite-like idempotent store | Latest keyword candidate only (stand-in); all matches up to a limit (real store) |
| `sidecar` | No | File outbox | Exact file-vector sidecar |
| `capsule` | SQLite input WAL + checkpoint | SQLite outbox | Exact SQLite vector index |

All arms consume the same ordered replay prefix, uEngram configuration,
idempotent Lite-like store, event identifiers, and query. The default `fixture`
uses deterministic 2-D windows. The `smd` option uses existing SMD window-mean
features as systems inputs only; it neither reads labels nor tunes the detector.

## Crash boundaries

1. `after_wal`: input is durable; uEngram has not consumed it.
2. `after_ingest`: uEngram and event projection have run; no engine checkpoint exists.
3. `after_checkpoint`: input and resulting engine state are durable.

The `lite` and `sidecar` arms restart with a fresh engine because they have no engine
checkpoint. The capsule arm restores its checkpoint and replays only WAL rows newer
than that checkpoint.

The restart is injected by closing and reopening the capsule inside one process, at the
midpoint of the replay prefix unless a restart position is given. It exercises logical
recovery ordering. The SIGKILL campaigns below kill a real process. Neither is a power-cut
or torn-write test, and this study makes no claim about those.

## Endpoints and interpretation

| Endpoint | Required capsule outcome | Interpretation |
|---|---|---|
| `state_equivalent_to_uninterrupted` | `true` at every crash boundary | Engine arrays exactly match an uninterrupted replay |
| `event_keys_match_reference` | `true` at every crash boundary | The set of lifecycle idempotency keys sent to the vendor store equals the uninterrupted run's set |
| `missing_event_count`, `extra_event_count` | 0 and 0 | Reference records never emitted, and records emitted that the reference never produced |
| `ranked_history_count` | Report only | Candidates returned by that arm's own retrieval path; judge it against the reference record set, not against another arm's count |
| `recovery_ns` | Report only | Restart-to-completion time over the remaining windows, including opening the database, restoring the checkpoint, replay and a commit per window. Whether that is CPU or fsync cost depends on the storage: check `host.storage` in the artifact. On tmpfs fsync is nearly free |
| `local_storage_bytes` | Report only | SQLite file sizes are page-granular; do not extrapolate per window |

Event **count** is deliberately not an endpoint. After losing detector state, a restarted
arm can emit records the uninterrupted run never produced and omit records it did, so equal
or near-equal counts can hide both errors. Compare idempotency-key sets.

## Representative-stream and host artifacts

Every v3 JSON result records the stream name, input-window count, feature dimension,
SHA-256 digest of the ordered feature bytes, the uninterrupted reference's lifecycle-record
count, and host metadata (hostname, platform, CPU model, CPU count, memory, Python and
NumPy versions and, in later runs, `storage`: the filesystem, device and rotational flag
backing the temporary directory). `--target-label` is an operator-supplied identifier, not
a hardware certification. Use `--require-target-label` for target-device artifacts so
unnamed runs fail closed.

Run an SMD replay prefix on a named target:

```bash
python -m bench.capsule_resilience \
  --stream smd --machine machine-1-1 --windows 64 \
  --crash after_ingest \
  --target-label msi --require-target-label \
  --out ~/.local/state/uengram/capsule-smd-machine-1-1.json
```

`scripts/capsule_target_run.sh` runs an experiment on each fleet host (`msi` locally; other
hosts receive a minimal tree and use system Python with NumPy). Two switches choose what runs:

- `EXPERIMENT=resilience|cost` (default `resilience`).
- `STORAGE=tmpfs|disk`: `tmpfs` uses the system temp directory, which is tmpfs on these hosts,
  so fsync is nearly free; `disk` points `TMPDIR` at a directory on the host's own disk. Trust
  the `host.storage` field recorded in each artifact, not the mode name.

Each run collects from its own remote subdirectory, so stale artifacts from earlier runs are
never mixed in. `REPEATS`, `COST_WINDOWS`, `COST_REPEATS` and `REMOTE_TIMEOUT` size the runs;
use small sizes on slow disks, where one durable commit can take hundreds of milliseconds.

```bash
EXPERIMENT=resilience STORAGE=disk REPEATS=3 OUT=~/.local/state/uengram/capsule-targets-disk \
  scripts/capsule_target_run.sh msi hp asus-worker
rsync -a ~/.local/state/uengram/capsule-targets-disk/ results/capsule_resilience_disk/
python -m bench.capsule_resilience_summary --root results/capsule_resilience_disk \
  --out results/capsule_resilience_disk_summary.json
```

The aggregator rejects artifacts that do not share one input digest, disagree on the
reference, or carry a label that differs from their directory. Correctness failures are
counted in the summary, never dropped. Compare `recovery_ns` only between artifacts with a
matching stream digest, declared host metadata and the same storage.

Run one boundary:

```bash
python -m bench.capsule_resilience \
  --crash after_ingest \
  --out ~/.local/state/uengram/capsule-resilience.json
```

Run all three boundaries by invoking the command once per `--crash` value.

## Results (2026-10-31)

All inputs are SMD windows and labels are never read. The first four-host runs used the
system temp directory, which is tmpfs, and did not record the storage; later runs record it.

### Restart recovery at the midpoint (tmpfs)

Input: `machine-1-1`, first 64 test windows, 38 features, digest
`47257349ba07964c08e0dd6e529cb9efd80116021e153970697eed7236de67fd`; restart at window 32; four
hosts, three boundaries, five repeats: 60 runs per arm
(`results/capsule_resilience/`, `results/capsule_resilience_summary.json`). The uninterrupted
reference emits three lifecycle records: `new_candidate` (step 0), `promote` (step 4) and
`alarm_open` (step 39).

| Arm | State equals uninterrupted | Event keys equal reference | Missing records | Extra records |
|---|---:|---:|---:|---:|
| `lite` | 0 / 60 | 0 / 60 | 1 | 2 |
| `sidecar` | 0 / 60 | 0 / 60 | 1 | 2 |
| `capsule` | 60 / 60 | 60 / 60 | 0 | 0 |

The outcome was identical on every host and boundary. Without detector state the restarted
engine re-emitted `new_candidate` (step 32) and `promote` (step 36) as if the machine were new,
and never emitted the `alarm_open` that the uninterrupted run raised at step 39.

The stand-in store returns only its newest record on recall, so the earlier "ranked history"
column (1 for `lite`) was a property of the stand-in, not of Lite. The real store returns every
matching record up to its limit, so retrieval is not compared across arms here.

**Correction.** An earlier version of this document reported the capsule at about five to six
times the memory-only restart time and attributed it to durable commits. Those timings came
from tmpfs, where fsync is nearly free; they measured CPU work in checkpoint serialization.
The cost on real storage is in the last section.

### Restart recovery with temporary storage on disk

The same experiment with `TMPDIR` on each host's disk
(`results/capsule_resilience_disk/`, `results/capsule_resilience_disk_summary.json`): 42
artifacts (five repeats on `msi`, three on each of `hp`, `asus` and `lenovo`). The capsule
matched the uninterrupted run in 42 / 42 runs, `lite` and `sidecar` in 0 / 84. Median
restart-to-completion time at `after_ingest` (ms):

| Host (temporary storage) | `lite` | `sidecar` | `capsule` |
|---|---:|---:|---:|
| msi (ext4, SSD) | 7.3 | 29.9 | 207.7 |
| lenovo (ext4, SSD) | 21.6 | 69.0 | 379.0 |
| asus (ext4, hard disk) | 7.9 | 460.2 | 3374.0 |
| hp (overlay) | 7.6 | 318.5 | 2336.0 |

`hp`'s storage was about six times faster during this run than during the cost study below, so
its figures are indicative only.

### Real EdgeMemory Lite

`--vendor real` runs the same experiment against the real shared library (it needs the vendor
binding and `cffi`, which are not project dependencies; see `bench/capsule_real_lite.py`).
Records are identified by content because the library's recall hides the idempotency key.
Fifteen runs on `msi` (`results/capsule_resilience_real/`): the capsule matched the reference
in 15 / 15, `lite` and `sidecar` in 0 / 30.

### Restart position and machine sweep

`python -m bench.capsule_restart_sweep` restarts at evenly spaced positions across machines
(`results/capsule_restart_sweep.json`). All 28 SMD machines, 256-window prefix, 16 positions:
448 cells. The capsule matched in 448 / 448. `lite` and `sidecar` matched in 0 / 448 and lost at
least one alarm-open record in 406 / 448 cells (90.6%); on average 3.6 of the reference's 8.2
alarm-open records were missing, and every cell held spurious records. Against the real library
(`results/capsule_restart_sweep_real.json`, 3 machines, 8 positions, 128 windows) the capsule
matched in 24 / 24 and the stateless arms lost an alarm-open in 18 / 24. The alarm records use
the harness's fixed `alarm_threshold=1.0`, not the calibrated alarm policy, so this measures
divergence from an uninterrupted run, not missed faults.

### Why the alarm is lost

`python -m bench.capsule_warmup_ablation` (`results/capsule_warmup_ablation.json`). The
reference's score climbs from 0.17 at step 32 to 1.21 at step 39; the restarted engine's scores
are undefined for five steps and then stay below 0.25. Replaying the last K pre-restart windows
into a fresh engine:

| K | First alarm | Records after the restart | Equals reference |
|---:|---:|---|---|
| 0, 2, 4, 8 | none | none (or spurious `new_candidate` / `promote`) | no |
| 16 | step 44 | `alarm_open` x2, `alarm_clear` x2 | no |
| 24, 32 | step 39 | `alarm_open` x1 | yes |

This supports, without proving, that a cold restart re-baselines on whatever it sees first, so
a drift that began before the restart is absorbed as normal. One machine, one restart position.

### SIGKILL campaigns

`python -m bench.capsule_kill_test` runs a capsule-backed ingest loop in a child process,
SIGKILLs it at a random moment, then starts a second process that recovers, drains the outbox
and finishes the stream from the recovered cursor. Four checks per trial: lifecycle records
equal the reference, no duplicate vendor rows, engine arrays equal the uninterrupted run after
recovering from the final on-disk state, SQLite integrity. `TMPDIR` was on `msi`'s ext4 SSD.
`--vendor real` uses the real library; `--vendor-post-commit-ms N` pauses the victim's vendor
store after each commit so kills land between a vendor commit and the checkpoint;
`--sync` and `--checkpoint-every` select the capsule configuration.

| Campaign | Trials | Killed mid-stream | Input logged, not checkpointed | Vendor ahead of checkpoint | Passed |
|---|---:|---:|---:|---:|---:|
| stand-in, natural | 150 | 127 | 86 | 14 | 150 |
| stand-in, 40 ms pause | 150 | 132 | 96 | 47 | 150 |
| real library, natural | 150 | 123 | 73 | 9 | 150 |
| real library, 40 ms pause | 150 | 140 | 101 | 46 | 150 |
| NORMAL sync, every window (pause) | 60 | 52 | 42 | 29 | 60 |
| FULL sync, every 16 windows (pause) | 60 | 47 | 45 | 34 | 60 |
| NORMAL sync, every 16 windows (pause) | 60 | 47 | 43 | 39 | 60 |

Trials that finished before the kill landed are counted in the totals. Natural timing rarely
killed the process between a vendor commit and the checkpoint, which is why the pause exists.
SIGKILL leaves the OS page cache intact, so this does not test power loss or torn writes, and
the relaxed rows show only that those configurations survive process death.
### Power-loss crash-image campaign

`scripts/capsule_power_loss_campaign.sh` records the capsule's file operations under an
`LD_PRELOAD` shim, then replays the trace into crash images at whole-syscall write
boundaries. The model simulates a power cut after each selected prefix; it does not model
sub-write tearing, block-device reordering below the page cache, or hardware recovery.
The campaign covered three SMD machines, `FULL` and `NORMAL`, checkpoint intervals 1 and
16 over 48 windows, and one 400-window run for each mode. It also ran `sync-off` and
`no-journal` negative controls on `machine-1-1`.

| Configuration | Trials | Capsule consistent | Images losing acknowledged windows | Max lost |
|---|---:|---:|---:|---:|
| `FULL`, every window, 48-window prefix | 1,491 | 1,491 / 1,491 | 0 | 0 |
| `FULL`, every 16, 48-window prefix | 1,471 | 1,471 / 1,471 | 0 | 0 |
| `FULL`, every window, 400-window prefix | 337 | 337 / 337 | 0 | 0 |
| `NORMAL`, every window, 48-window prefix | 1,491 | 1,491 / 1,491 | 949 | 47 |
| `NORMAL`, every 16, 48-window prefix | 1,472 | 1,472 / 1,472 | 766 | 48 |
| `NORMAL`, every window, 400-window prefix | 337 | 337 / 337 | 223 | 169 |
| `sync-off` negative control | 481 | 481 / 481 | 337 | 48 |
| `no-journal` negative control | 483 | 317 / 483 | 199 | 48 |

All `FULL` and `NORMAL` crash images remained internally consistent and the final vendor
state matched the reference. `NORMAL` nevertheless lost acknowledged input windows in
the modeled images; the input log commits every window, but those commits are not all
durable at a power cut. The `no-journal` control produced 166 inconsistent images,
including malformed SQLite databases. These results support the capsule's modeled
whole-syscall recovery behavior and the durability trade-off; they do not establish
hardware power-loss safety.

The artifacts are under `results/capsule_power_loss/`; the result ledger's Table A56
contains every cell. SMD labels are never read.

---

### Cost of durability


`python -m bench.capsule_cost_study` (`results/capsule_cost/`). Mean ingest cost per window
(ms), medians over interleaved repeats (5 x 256 windows on `msi`; 3 x 64 on the other three hosts):

| Host (storage) | Memory-only | FULL, every window | FULL, every 16 | FULL, every 64 | NORMAL, every window | NORMAL, every 16 |
|---|---:|---:|---:|---:|---:|---:|
| msi (ext4, SSD) | 0.14 | 7.77 | 3.64 | 3.23 | 1.47 | 0.31 |
| lenovo (ext4, SSD) | 0.38 | 9.95 | 4.10 | 3.61 | 2.93 | 0.75 |
| asus (ext4, hard disk) | 0.44 | 104.61 | 56.99 | 56.42 | 1.76 | 0.73 |
| hp (overlay) | 0.14 | 489.87 | 243.93 | 225.14 | 1.52 | 0.35 |
| msi (tmpfs) | 0.48 | 2.94 | 1.03 | 0.82 | 2.80 | 0.66 |
| asus (tmpfs) | 0.14 | 1.36 | 0.34 | 0.28 | 1.36 | 0.34 |
| hp (tmpfs) | 0.13 | 1.27 | 0.32 | 0.27 | 1.27 | 0.32 |
| lenovo (tmpfs) | 0.38 | 2.76 | 0.73 | 0.62 | 2.75 | 0.72 |

On tmpfs FULL equals NORMAL, which is why the earlier timings misled. On disk, FULL costs about
one fsync per window even at long intervals, because the input log commits every window: a
longer checkpoint interval removes only the second commit. NORMAL avoids the per-window fsync
at 0.3 to 2.9 ms but weakens durability on power loss, which is untested. A longer interval
also lengthens the worst-case replay (up to the interval minus one windows): on the hard disk
under FULL a 64-window interval recovered in 438 ms against 26 ms for every-window checkpoints.
`msi` is an interactive workstation and `hp`'s storage varied about sixfold between runs.

## Limits

- SMD server metrics only. The mechanism probe uses one machine and one restart position; the
  sweep covers 28 machines; the SIGKILL and real-library campaigns use three.
- The alarm records come from a fixed threshold, so their operational value is not evaluated.
- The stand-in vendor store is not Lite; only the real-library runs use the real backend.
- SIGKILL and in-process reopening are not power loss. The crash-image campaign is a
  whole-syscall model; sub-write tearing, block-device reordering, flash behavior, hardware
  power cuts, and the durability of `synchronous=NORMAL` on real hardware remain untested.
- The hosts are x86 desktop and laptop machines, not embedded targets. Timings are host-local
  and say nothing about embedded latency, RAM, or flash endurance.
- `gigabyte-worker` was unreachable and is not covered. The first four-host runs and the
  on-disk runs cover the same four hosts (`msi`, `hp`, `asus`, `lenovo`).
- `msi` ran Python 3.12 with NumPy 2.5.4; the other hosts ran Python 3.14 with NumPy 2.5.3.
  All shared one input digest and gave the same correctness outcome.
- This is a systems-recovery result, not anomaly-detection accuracy.
