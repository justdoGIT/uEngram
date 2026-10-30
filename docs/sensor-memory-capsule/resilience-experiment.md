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

## Endpoints and interpretation

| Endpoint | Required capsule outcome | Interpretation |
|---|---|---|
| `state_equivalent_to_uninterrupted` | `true` at every crash boundary | Detector state exactly matches uninterrupted replay |
| `vendor_event_count` | Matches uninterrupted replay | Idempotent replay did not create duplicate projections or lose lifecycle records |
| `ranked_history_count` | At least Lite-only count | Exact local index retains candidates beyond Lite recall truncation |
| `recovery_ns` | Report only | Local-process timing, not target-device latency |
| `local_storage_bytes` | Report only | Workload-specific local persistence cost |

## Representative-stream and host artifacts

Every v2 JSON result records the stream name, input-window count, feature dimension,
SHA-256 digest of the ordered feature bytes, and Python/NumPy/host/CPU metadata.
`--target-label` is an operator-supplied identifier, not a hardware certification.
Use `--require-target-label` for target-device artifacts so unnamed runs fail closed.

Run an SMD replay prefix on a named target:

```bash
python -m bench.capsule_resilience \
  --stream smd --machine machine-1-1 --windows 64 \
  --crash after_ingest \
  --target-label msi --require-target-label \
  --out ~/.local/state/uengram/capsule-smd-machine-1-1.json
```

Repeat each crash boundary on each target host and preserve the generated JSON files.
The recorded `recovery_ns` values are host-local timing observations; compare them only
between artifacts with a matching stream digest and declared host metadata.

Run one boundary:

```bash
python -m bench.capsule_resilience \
  --crash after_ingest \
  --out ~/.local/state/uengram/capsule-resilience.json
```

Run all three boundaries by invoking the command once per `--crash` value. Result JSON
is evidence for systems recovery only. Publish no accuracy, latency, endurance, or
production-readiness claim from this experiment without independent target-hardware and
representative-stream measurements.
