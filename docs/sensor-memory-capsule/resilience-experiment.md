# Capsule Resilience Experiment

## Claim under test

The capsule preserves uEngram recovery semantics across a process loss. It is not an
experiment of anomaly-detection accuracy, fault lead time, field value, target-device
latency, or flash endurance.

## Fixed conditions

| Condition | Durable detector state | Event delivery | Historical retrieval |
|---|---:|---|---|
| `lite` | No | Direct Lite-like idempotent store | Latest keyword candidate only |
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

The restart is injected at the midpoint of the replay prefix by closing and reopening
the capsule inside one process. It exercises logical recovery ordering. It is not a
`kill -9`, power-cut, or torn-write test, and this experiment makes no claim about those.

## Endpoints and interpretation

| Endpoint | Required capsule outcome | Interpretation |
|---|---|---|
| `state_equivalent_to_uninterrupted` | `true` at every crash boundary | Engine arrays exactly match an uninterrupted replay |
| `event_keys_match_reference` | `true` at every crash boundary | The set of lifecycle idempotency keys sent to the vendor store equals the uninterrupted run's set |
| `missing_event_count`, `extra_event_count` | 0 and 0 | Reference records never emitted, and records emitted that the reference never produced |
| `ranked_history_count` | Report only | Candidates returned by that arm's own retrieval path; judge it against the reference record set, not against another arm's count |
| `recovery_ns` | Report only | Restart-to-completion time over the remaining windows. For the capsule it includes opening the database, restoring the checkpoint and a durable commit per window, so it measures the cost of durability and is not a pure recovery latency |
| `local_storage_bytes` | Report only | SQLite file sizes are page-granular; do not extrapolate per window |

Event **count** is deliberately not an endpoint. After losing detector state, a restarted
arm can emit records the uninterrupted run never produced and omit records it did, so equal
or near-equal counts can hide both errors. Compare idempotency-key sets.

## Representative-stream and host artifacts

Every v3 JSON result records the stream name, input-window count, feature dimension,
SHA-256 digest of the ordered feature bytes, the uninterrupted reference's lifecycle-record
count, and host metadata (hostname, platform, CPU model, CPU count, memory, Python and
NumPy versions). `--target-label` is an operator-supplied identifier, not a hardware
certification. Use `--require-target-label` for target-device artifacts so unnamed runs
fail closed.

Run an SMD replay prefix on a named target:

```bash
python -m bench.capsule_resilience \
  --stream smd --machine machine-1-1 --windows 64 \
  --crash after_ingest \
  --target-label msi --require-target-label \
  --out ~/.local/state/uengram/capsule-smd-machine-1-1.json
```

`scripts/capsule_target_run.sh` runs all three boundaries five times on each fleet host
(`msi` locally; other hosts receive a minimal tree and use system Python with NumPy) and
collects `<label>/<crash>-<n>.json`. Copy the artifacts to `results/capsule_resilience/` and
aggregate them:

```bash
scripts/capsule_target_run.sh msi hp asus-worker lenovo-worker
rsync -a ~/.local/state/uengram/capsule-targets/ results/capsule_resilience/
python -m bench.capsule_resilience_summary
```

The aggregator rejects artifacts that do not share one input digest, disagree on the
reference, or carry a label that differs from their directory. Correctness failures are
counted in the summary, never dropped. Compare `recovery_ns` only between artifacts with a
matching stream digest and declared host metadata.

Run one boundary:

```bash
python -m bench.capsule_resilience \
  --crash after_ingest \
  --out ~/.local/state/uengram/capsule-resilience.json
```

Run all three boundaries by invoking the command once per `--crash` value.

## Results (2026-10-31)

Input: SMD `machine-1-1`, first 64 test windows, 38 features, digest
`47257349ba07964c08e0dd6e529cb9efd80116021e153970697eed7236de67fd`. The restart occurs at
window 32. Four hosts, three boundaries, five repeats: 60 runs per arm. Raw artifacts are in
`results/capsule_resilience/`; the aggregate is `results/capsule_resilience_summary.json`.

The uninterrupted reference emits three lifecycle records: `new_candidate` (step 0),
`promote` (step 4) and `alarm_open` (step 39).

| Arm | State equals uninterrupted | Event keys equal reference | Missing records | Extra records | Ranked history |
|---|---:|---:|---:|---:|---:|
| `lite` | 0 / 60 | 0 / 60 | 1 | 2 | 1 |
| `sidecar` | 0 / 60 | 0 / 60 | 1 | 2 | 4 |
| `capsule` | 60 / 60 | 60 / 60 | 0 | 0 | 3 |

The outcome was identical on every host and boundary. Without detector state, the restarted
engine re-emitted `new_candidate` (step 32) and `promote` (step 36) as if the machine were
new, and never emitted the `alarm_open` that the uninterrupted run raised at step 39. The
sidecar's four ranked candidates therefore include two spurious records and lack the alarm.
The capsule's three match the reference exactly.

Restart-to-completion time, median of five runs at the `after_ingest` boundary (ms):

| Host | CPU | `lite` | `sidecar` | `capsule` | `capsule` / `lite` |
|---|---|---:|---:|---:|---:|
| hp | i7-6700, 8 CPUs | 7.9 | 9.2 | 50.7 | 6.4 |
| asus | i3-6100, 4 CPUs | 7.8 | 9.0 | 50.2 | 6.4 |
| lenovo | i3-3110M, 4 CPUs | 23.6 | 27.3 | 135.2 | 5.7 |
| msi | i3-7100, 4 CPUs | 18.6 | 32.0 | 102.2 | 5.5 |

The capsule costs about five to six times the memory-only arms over roughly 32 replayed
windows, which is about 1.3 ms per window on `hp` if the whole difference is attributed to
durable commits (an upper bound on that attribution: it also includes opening the database
and restoring the checkpoint). Timing is noisy: `capsule` medians across the three boundaries
ranged 104.5 to 135.2 ms on `lenovo` and 96.0 to 103.1 ms on `msi`, and `msi` is an
interactive workstation. Read the ratios as roughly 5 to 6 times, not as point estimates.
Local storage was 131,072 bytes for the capsule (SQLite page granularity) and 3,039 bytes for
the sidecar.

## Limits

- One SMD machine, one 64-window prefix, one restart position (the midpoint). The result
  shows the mechanism works on that stream; it does not show it generalizes to other
  streams, restart positions or longer histories.
- The vendor store is a Lite-like in-process stand-in, not the real EdgeMemory Lite.
- Restarts are simulated by reopening the database in one process. Power loss, torn writes
  and filesystem behaviour are untested.
- The hosts are x86 desktop and laptop machines, not embedded targets. Timings are host-local
  and say nothing about embedded latency, RAM, or flash endurance.
- `gigabyte-worker` was unreachable ("No route to host") and is not covered.
- `msi` ran Python 3.12 with NumPy 2.5.4; the other hosts ran Python 3.14 with NumPy 2.5.3.
  All shared one input digest and gave the same correctness outcome.
- This is a systems-recovery result, not anomaly-detection accuracy. The SMD inputs are used
  only as an ordered replay source; labels are never read.
