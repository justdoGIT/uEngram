# Sensor Memory Capsule

## Decision

The Sensor Memory Capsule is the host-owned system of record for a uEngram deployment. It keeps detector recovery, lifecycle events, raw-evidence pin intent, exact historical vectors, and EdgeMemory Lite delivery state together. EdgeMemory Lite is an asynchronous, idempotent metadata projection, never the sole source of sensor state.

## Block diagram

```mermaid
flowchart LR
    W[Sensor window] --> I[SQLite input WAL]
    I --> U[uEngram ingest]
    U --> T[SQLite transaction]
    T --> C[Versioned engine + alarm checkpoint]
    T --> E[Lifecycle event + exact vector]
    T --> P[Raw-evidence pin intent]
    T --> O[Lite delivery outbox]
    O --> D[Delivery worker]
    D --> L[EdgeMemory Lite]
    L -->|idempotency acknowledgement| O
    Q[Identity-filtered query] --> U
    Q --> E
    Q --> L
    U --> R[Live state]
    E --> H[Exact historical matches]
    L --> K[Keyword metadata matches]
    R --> X[Auditable response]
    H --> X
    K --> X
```

## Invariants

1. A window is written to the input WAL before uEngram consumes it.
2. Each committed checkpoint records the resulting engine state, adapter state, last accepted step, and alarm state.
3. A lifecycle event has one stable idempotency key across local recovery and Lite delivery retries.
4. Raw-evidence operations are durable intents. Pin/unpin implementations must be idempotent because they run outside SQLite.
5. Exact historical ranking filters by identity before scoring. Lite keyword recall is supplementary and remains visibly distinct.
6. Recovery restores the latest validated checkpoint, then exposes WAL windows after that checkpoint for deterministic replay.

## Storage model

The capsule uses SQLite in WAL mode with `synchronous=FULL` by default for its local transaction log; `SensorMemoryCapsule(path, synchronous="NORMAL")` relaxes it. Its schema contains input windows, checkpoints, lifecycle events, compact event vectors, raw pin intents, and Lite outbox rows. Arrays are stored in an explicit NumPy archive with no pickle payload; checkpoints refuse a mismatched engine configuration or latent dimension.

Two options trade durability cost against replay length:

- `synchronous`: `FULL` (default) syncs the WAL on every commit; `NORMAL` syncs only when SQLite checkpoints its WAL. `NORMAL` is far cheaper on slow disks but can lose the most recent commits on power loss. That durability is untested here; only process death (SIGKILL) is.
- `CapsuleSensorStateMemory(..., checkpoint_every=N)` writes the engine checkpoint every `N` ingested windows (default 1). Recovery then replays up to `N - 1` logged windows. The input log still commits every window, so under `FULL` this removes the second fsync per window, not the first.

## Integration

```python
capsule = SensorMemoryCapsule("/var/lib/pump-17/capsule.db")
pipeline = CapsuleSensorStateMemory(capsule, engine, edge_memory)

pipeline.recover()  # restores the checkpoint, then replays a WAL suffix
receipt = pipeline.ingest(window)
```

`CapsuleSensorStateMemory` records each input before uEngram consumes it, uses the
capsule for Lite delivery and exact-vector indexing, then checkpoints the resulting
engine state. Lite writes remain idempotent projections. A raw-evidence adapter must
reconcile the capsule's pin intents with idempotent `pin` and `unpin` operations.

## Research protocol and result

The deterministic A/B/C recovery experiment, crash boundaries, fixed endpoints, and
claim limits are in [the resilience experiment protocol](resilience-experiment.md).

Measured results (SMD server metrics, systems recovery only):

- **Detector state is lost on restart.** After a restart the Lite-only and file-sidecar arms
  emitted spurious lifecycle records and lost alarm-open records: in 120 / 120 runs on four
  hosts, and in 406 / 448 machine-and-restart-position cells across all 28 SMD machines where
  the uninterrupted run alarms. A replay experiment indicates why: a cold restart re-baselines
  on already-drifted data.
- **The capsule matches an uninterrupted run** in all 60 runs, all 448 sweep cells, and all
  780 SIGKILL trials, including against the real EdgeMemory Lite library.
- **Durability costs real time on real disks.** Full synchronization costs 8 to 490 ms per
  window on an SSD, a hard disk and a slow overlay filesystem, against about 1.5 ms with
  `synchronous=NORMAL`. An earlier "five to six times" figure was measured on tmpfs, where
  fsync is nearly free, and was CPU cost only; it was wrong as a cost of durable commits.
- **Not tested:** power loss, torn writes, the durability of `NORMAL`, embedded targets.

## Deliberate limits

This is an exact host index, not ANN. It improves candidate completeness when Lite keyword recall is truncated, but does not prove detector accuracy, target-hardware latency, flash endurance, or field diagnostic value. Native vectors, binary payloads, graph relationships, and cross-store transactions still require a versioned vendor ABI and backend implementation. Process death is tested; power loss is not.
