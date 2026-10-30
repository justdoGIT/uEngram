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

All arms consume the same four deterministic 2-D sensor windows. They use the same
uEngram configuration, idempotent Lite-like store, event identifiers, and query.

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
| `vendor_event_count` | 4 | Idempotent replay did not create duplicate projections or lose records |
| `ranked_history_count` | At least Lite-only count | Exact local index retains candidates beyond Lite recall truncation |
| `recovery_ns` | Report only | Local-process timing, not target-device latency |
| `local_storage_bytes` | Report only | Workload-specific local persistence cost |

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
