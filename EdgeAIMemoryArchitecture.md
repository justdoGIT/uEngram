# Edge sensor-state memory architecture

This document describes the architecture implemented and evaluated by this repository. It is a repository design note, not a product specification or a claim about an external implementation.

## Purpose

uEngram is a deterministic, fixed-budget memory for continuous sensor streams. It turns causal sensor windows into compact representations, absorbs routine operating states into bounded statistics, retains rare excursions with raw-window provenance, and emits alarms through a frozen policy.

The design targets edge workloads where the write path must remain bounded and predictable. It does not put a language model in the ingest transaction and it does not treat physical telemetry as natural-language memory.

## System boundary

```text
sensor stream
    │ causal window X ∈ R^(C×K)
    ▼
validation and feature encoder
    │ DSP features, optional CNN, PCA projection
    ▼
noise whitening and fixed-dimensional vector z
    │ novelty, health, operating-state gates
    ├───────────────► alarm persistence policy
    │
    ▼
bounded slot memory
    │ semantic states · candidate states · episodic states
    ├───────────────► event store and raw-window pin
    │
    ▼
operator/research readout
```

The online path is deterministic. Model fitting, PCA, noise-covariance estimation, threshold calibration, and teacher distillation belong to offline commissioning or research experiments, not to the normal ingest loop.

## Per-window lifecycle

For each validated causal window:

1. The selected encoder produces a fixed-dimensional feature vector.
2. A frozen projection and within-state noise model produce the whitened vector `z`.
3. The engine compares `z` with semantic slots and evaluates novelty and health scores.
4. A window inside a semantic gate updates the relevant running state.
5. A new or unconfirmed operating state enters a candidate slot and remains in probation until its promotion rule is met.
6. A rare excursion enters an episodic slot only when its competitive admission rule defeats the weakest stored episode.
7. A lifecycle event is persisted only after the raw window is pinned. If persistence fails, the pin is released and the pending record is returned for retry.
8. Alarm counters apply the frozen persistence rule; ordinary windows do not become flash writes merely because they were observed.

The default reference policy uses five hits in eight steps for candidate promotion, twenty hits in thirty steps for re-confirmation, and a minimum semantic adaptation rate of `0.01`. Dataset-specific configuration is frozen before its reported evaluation.

## Memory roles

### Semantic slots

Semantic slots summarize confirmed normal operating states. They maintain pooled counts, means, and variances and adapt only under the configured safe-admission rule. A state change can supersede a dormant semantic slot while retaining the previous state for history.

### Candidate slots

Candidate slots quarantine an unconfirmed regime. They prevent a new load, repair state, or transient shift from immediately rewriting the healthy reference. Promotion requires repeated evidence under the frozen gate.

### Episodic slots

Episodic slots retain rare excursions and a pointer to the raw sensor window. Admission is competitive and bounded: an incoming episode is kept only when its importance exceeds the weakest stored episode. This preserves diagnostic examples without allowing the episodic budget to grow without limit.

## Representation and noise model

The reference DSP encoder computes per-channel time and frequency features, robustly scales them, and projects them with PCA. CNN encoders are evaluated as research variants; their weights are fitted offline and frozen for evaluation.

Within an operating state, the reference estimates noise from lagged window differences. A robust covariance estimate is regularized before whitening. The resulting distance is measured in the whitened coordinate system so correlated sensors do not count as independent evidence multiple times.

For a state with mean `μ` and covariance `Σ`, the conceptual score is:

```text
d(z, μ) = sqrt((z - μ)ᵀ Σ⁻¹ (z - μ))
```

The implementation uses fail-soft covariance handling and records the conditioning diagnostics needed to distinguish a measured result from an unsupported numerical assumption.

## Alarm policy

The reference policy separates:

- **Novelty:** distance from confirmed semantic states.
- **Health:** distance from the frozen healthy commissioning reference.
- **Persistence:** consecutive threshold exceedances required to open an alarm.
- **State coverage:** explicit handling for windows that cannot be safely compared with a commissioned state.

Thresholds are calibrated on fault-free commissioning data. Reported event recall and false-alarm results are criterion-dependent; ranking metrics such as AUROC are not substitutes for an operational alarm evaluation.

## Persistence and recovery

The capsule adapter combines an input write-ahead record with a versioned engine checkpoint in one SQLite-backed store. It records lifecycle events, detector state, checkpoint position, and raw-window pin references. Restart, process-death, and modeled whole-syscall crash-image experiments measure recovery behavior and synchronization cost.

Those experiments do **not** establish hardware power-loss behavior. Torn writes, block-device reordering, flash-controller behavior, embedded-device timing, and real power cuts remain deployment gates.

## Offline commissioning

Commissioning fits and freezes the artifacts used by the online path:

1. Select a training-only healthy or development split.
2. Fit the encoder and PCA projection.
3. Estimate the within-state noise model and record conditioning diagnostics.
4. Build the healthy reference and calibrate alarm thresholds.
5. Freeze configuration, model weights, split, and analysis before evaluating a holdout.

Teacher models may be distilled into compact students for research comparisons. Distillation is optional and must be judged with operational alarm criteria, not ranking metrics alone.

## Implementation status

| Capability | Repository status |
|---|---|
| Causal sensor-window processing | Implemented in the Python reference engine |
| DSP plus PCA encoder | Implemented and used in benchmark protocols |
| CNN encoder variants | Implemented as research baselines and ablations |
| Robust noise estimation and whitening | Implemented and tested |
| Semantic, candidate, and episodic slot lifecycle | Implemented and tested |
| Persistence counter and frozen alarm policy | Implemented and evaluated |
| Raw-window pin-before-write adapter | Implemented and tested |
| Restart and process-death studies | Implemented; results are committed under `results/` |
| Production C/C++ database integration | Not implemented in this repository |
| Hardware power-cut validation | Not performed |
| Independent field-validation cohort | Not available; protocol is frozen for future acquisition |

## Evidence and limits

The synthetic lifecycle benchmark demonstrates bounded-memory behavior under its stated generator assumptions, including a reported 86.2% write-operation reduction over ten seeds while retaining injected events. That is a modeled operation count, not measured flash endurance or write amplification.

Public benchmark results show the distinction between representation ranking and deployable warning behavior. The paper reports the exact splits, thresholds, event criteria, negative transfer results, domain-specialist results, and durability experiments. Most results are single-workload studies; they are not field-validation or revenue claims.

Before safety-relevant deployment, the remaining gates include:

- a preregistered multi-asset field cohort with independently adjudicated fault-onset times;
- adversarial poisoning and shadow-baseline tests;
- explicit abstention for unknown operating states;
- transactional cross-store power-cut and torn-write tests; and
- target-hardware latency, RAM, flash-write-amplification, and retrieval measurements.

## Repository map

- `src/uengram/engine.py`, `lifecycle.py`, and `sensor_memory.py`: bounded state and slot lifecycle.
- `src/uengram/encoders.py` and CNN encoder modules: feature representations.
- `src/uengram/persistence.py`: capsule and event-store adapter.
- `docs/EXPLAINED.md`: implementation explanation and mathematical details.
- `docs/reproducing-the-paper.md`: dataset layout, experiment commands, and artifact mapping.
- `docs/independent-cohort-preregistration.md`: frozen independent-validation protocol.
- `docs/research-paper/From_Sensor_Streams_to_Bounded_Machine_Memory.md`: manuscript source.
- `docs/research-paper/From_Sensor_Streams_to_Bounded_Machine_Memory.pdf`: generated publication PDF.

## Reproduction

Run from the repository root:

```bash
uv sync --extra cnn --extra baselines
uv run pytest -q
```

The paper reproduction guide is the source of truth for dataset placement, experiment commands, committed artifacts, and publication-PDF generation. Raw datasets remain outside version control.
