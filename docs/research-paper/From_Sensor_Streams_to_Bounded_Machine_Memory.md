# From Sensor Streams to Bounded Machine Memory: Architecture, Negative Transfer Results, and Domain Specialists in uEngram

**Kamal Kishor Pandey** · Independent Researcher · kamal.pandey@kepr.dev

**Keywords:** Bounded edge memory · Streaming anomaly detection · Noise whitening · Time-series foundation models · Knowledge distillation · Telemetry databases

## Abstract

Industrial and edge equipment generate continuous telemetry across vibration, temperature, pressure, and system load. Retaining every raw window exhausts flash endurance and storage capacity, while naive first-in-first-out (FIFO) buffers overwrite rare fault episodes with routine operational data. This paper evaluates **uEngram**, a bounded stateful memory runtime that compresses short sensor windows into low-dimensional vectors, tracks operating regimes through noise-whitened statistical summaries, quarantines unverified shifts, and pins raw waveform slices for high-rarity episodes.

While synthetic benchmarks confirm that uEngram cuts storage writes by 86.2% on average over 10 seeds (`results/synth_runs.jsonl`, DSP $d=16$) at 1.00 modeled event recall, evaluations on public datasets and benchmark suites expose a sharp divide between point-wise anomaly ranking (AUROC) and operational early warning. Frozen detectors miss uncommissioned load states on MetroPT-3 (AUROC 0.07 and 0.35 for clean windows against the 24 h before a report, within sustained load), detect 0 of 5 events on MetroPT-2, and catch only 5 of 34 SKAB faults—all after physical onset. Furthermore, distilling foundation models (MOMENT-1-small and Chronos-2) into a 46 KB edge student improves window ranking on MFPT and CMAPSS (CWRU saturates at 1.0000 AUROC for all models), yet flags the one held-out healthy recording per bearing dataset in every model (normal-file false-alarm rate 1.0; `results/cwru_two_teacher_gate.json`, `results/mfpt_two_teacher_gate.json`), and degrades timely maintenance recall at the 20-cycle remaining-useful-life (RUL) horizon. We show that overcoming channel dilution and covariance singularity requires pre-fit numerical diagnostics paired with a **role-separated, domain-specialist architecture**, which improves strict event recall on OS-Metrics to 69.0% (29/42), reduces false alarms to 19.6 per week, achieves 100% subsystem root-cause attribution (30/30 test-split events), and executes in $6.27\,\mu\text{s}$ per window.

> **Principal findings**
>
> 1. **Bounded memory works where its assumptions hold:** 86.2% fewer writes at 1.00 modeled event recall over 10 synthetic seeds (`results/synth_runs.jsonl`).
> 2. **Ranking is not warning:** every CWRU model reaches 1.0000 AUROC, yet every model flags the held-out healthy file (normal-file false-alarm rate 1.0).
> 3. **Proxy screens mislead:** the label-free screen's winner (middle layer, 0.610) scored a lower event median than the bottleneck in the full pipeline on both SKAB and MetroPT-3.
> 4. **Distillation can cost lead time:** on CMAPSS FD001 the ordinary CNN caught 7/10 failing engines by RUL 20; MOMENT and two-teacher students caught 2/10.
> 5. **Partitioning beats pooling:** domain specialists reach 29/42 strict recall at 19.6 false alarms per week on OS-Metrics, against 27/42 at 26.6 for the DSP baseline.
> 6. **Retraction:** the NASA IMS "63.8-hour early warning" claim is withdrawn (§3.2).

---

## 1. Introduction and the Flawed "Edge Mem0" Premise

Embedded telemetry systems face a structural retention dilemma: at one 16-float vector per second (86,400 vectors/day), raw vector payloads alone consume ~5.3 MiB daily before accounting for timestamps, metadata, index structures, and high-rate source waveforms. Updating approximate nearest-neighbor (ANN) graph indexes (such as HNSW) in place also introduces random-write amplification on raw NAND flash and eMMC storage. Conversely, ring buffers that evict strictly by age discard historical fault signatures whenever a machine runs normally for days after an incident.

This investigation originated from an embedded database proposal to port cloud LLM memory frameworks—specifically Mem0 [1]—directly to edge devices using ANN vector search. Testing that concept against physical telemetry exposed three architectural fallacies:

1. **Physical State vs. Conversational Memory:** Mining trucks, wind turbines, and compressors do not need natural-language entity extraction to remember user preferences; they require bounded tracking of wear baselines, operating regimes, and acute fault excursions.
2. **The Ingestion-Path Compute Bottleneck:** Cloud Mem0 invokes an LLM on every write to deduplicate facts and resolve contradictions. On an embedded gateway ingesting 100 Hz–1 kHz sensor streams, placing a language model in the write loop violates real-time latency and RAM bounds. Ingestion and consolidation must remain deterministic and mathematical.
3. **RAM Starvation and Flash Wear:** Continuous mutation of in-memory HNSW graphs starves constrained RAM and accelerates flash wear. Edge state memory requires tiered, density-aware compaction and strict admission gates.

To evaluate what works in practice, we analyze the `uEngram` Python reference implementation alongside its `EdgeMemory Lite` persistence adapter, synthesizing mechanism proofs, negative transfer findings, and methodological corrections across synthetic and field datasets.

![Figure 1. Text RAG versus the proposed sensor-state memory pipeline: two kinds of memory, one lifecycle idea.](figures/rag-sensor-memory.svg)

---

## 2. Core Architecture and Memory Lifecycle

![Figure 2. uEngram edge path (per window, deterministic) and offline commissioning path.](figures/system-architecture.svg)

### 2.1 Feature Encoding and Noise-Whitened Projection

Rather than storing individual samples, uEngram processes time-aligned causal windows $X \in \mathbb{R}^{C \times K}$ across $C$ channels and $K$ samples. The baseline digital signal processing (DSP) encoder extracts per-channel mean, spread, and energy across four frequency bands. Alternative encoders include a compact two-stage 1D convolutional neural network (`CNNEncoder`) trained via reconstruction and partitioned domain-specialist encoders.

Because raw sensor features covary under load changes, Euclidean distance overweights correlated channels. uEngram projects features via principal component analysis (PCA) to dimension $d$ and applies within-state noise whitening [3] estimated from lagged differences between adjacent windows using the median absolute deviation (MAD). For a representation $r(x)$ with mean $\bar{r}$, PCA eigenvectors $V$, eigenvalues $\Lambda$, and within-state noise covariance $N$, the whitened state vector is:

$$z(x) = N^{-1/2}\Lambda^{-1/2}V^\top(r(x) - \bar{r})$$

This whitening step suppresses routine co-movement and outlier spikes during commissioning, though setting $d$ too wide introduces collinear frequency bands that make $N$ ill-conditioned or singular.

Algorithm 1 gives the noise estimate. Lag differences inside one state contain only noise, so a trimmed, reweighted covariance of those differences ignores the jumps caused by regime switches and faults. Listing 1 shows the DSP features and the symmetric whitening $N^{-1/2}$ built from it; eigenvalues below `rcond` times the largest are floored before inversion.

```algorithm | Algorithm 1. Robust within-state noise covariance (`robust_noise_cov`, `src/uengram/stats.py:82-89, 128-153`).
Input: projected training vectors z₁ … zₙ, lag L, trim quantile q_t = 0.975, iterations I = 5
Output: noise covariance N
D ← { (z_{t+L} − z_t) / √2 : t and t+L in the same group }
q ← χ²_d quantile at q_t ;  c ← q_t / P(χ²_{d+2} ≤ q)            ▷ consistency factor for trimming
N ← diag(σ̂²),  σ̂_j ← 1.4826 · MAD(D_j)                            ▷ robust per-dimension start
repeat I times
    keep ← { Δ ∈ D : Δᵀ N⁻¹ Δ ≤ q }                                ▷ drops regime jumps and steps
    if |keep| < max(10, d + 1) then raise ValueError
    N ← c · Cov(keep)
return N
```

```python | Listing 1. DSP band features and symmetric noise whitening (`src/uengram/encoders.py:41-48, 59-69`; type hints and docstrings trimmed).
def band_features(windows):
    x = np.asarray(windows, dtype=np.float64)
    N, C, K = x.shape
    mean = x.mean(axis=2)
    xc = x - mean[:, :, None]
    log_std = np.log(xc.std(axis=2) + 1e-6)
    power = np.abs(np.fft.rfft(xc, axis=2)[:, :, 1 : K // 2 + 1]) ** 2 / K
    bands = [np.log(b.sum(axis=2) + 1e-8) for b in np.array_split(power, 4, axis=2)]
    feats = np.stack([mean, log_std, *bands], axis=2)  # [N,C,6]
    return feats.reshape(N, C * 6)


class NoiseWhitener:
    def fit(self, z, noise_lag, groups):
        lam, V = np.linalg.eigh(robust_noise_cov(z, noise_lag, groups))
        lam = np.maximum(lam, self.rcond * max(float(lam.max()), 1e-12))
        self.W = (V / np.sqrt(lam)) @ V.T
        return self

    def transform(self, z):
        return np.asarray(z, dtype=np.float64) @ self.W
```

### 2.2 Bounded State Slots and Competitive Admission

uEngram divides a fixed memory budget into **semantic slots** (compact summaries of confirmed routine operating modes), **candidate slots** (unfamiliar patterns undergoing persistence probation), and **episodic slots** (rare excursions paired with pointers to raw time-series slices).

Early versions of uEngram evicted the weakest mature episodic slot whenever any out-of-gate sample arrived, allowing mild outliers to overwrite severe historical faults. Informed by thresholded streaming memory in MemStream [8], diversity-aware replay in Rainbow Memory [9], constrained buffer allocation [10], and prototype replay [11], uEngram implements **competitive rarity admission**. An incoming out-of-gate sample is admitted to a full episodic bank only if its rarity-derived importance exceeds the weakest stored episode; otherwise, it is discarded while probation candidates remain replaceable.

![Figure 3. Slot state machine of `Engram.ingest`: gate routing, candidate probation, promotion, step supersession, and competitive episodic admission. A window inside an episodic slot's gate is absorbed by that slot; only the candidate path is drawn.](figures/slot-state-machine.svg)

On a five-seed synthetic benchmark with a configured total budget of 16 slots (13 episodic, 3 semantic; `docs/EXPLAINED.md:375`), competitive admission raised mean event recall from 0.760 to 0.851 and write reduction from 85.78% to 87.56%; type recall stayed at 1.0 and current-regime retrieval rose from 0.998 to 1.0. Seizure-event recall fell from 0.667 to 0.600, so the rule is not a universal improvement and must be validated on independent real streams.

Algorithm 2 is the per-window update in `Engram.ingest`. Novelty is scored against semantic (normal) slots only, so a recurring fault held in an episodic slot cannot mask itself; rarity, the distance to any stored slot, ranks what is worth keeping. When the episodic bank is full, a new window must beat the weakest episode's importance, surprisal × (1 + log n) × $2^{-\text{age}/h}$ with $h$ = `half_life_epi` (default 20,000 steps), or it is discarded (Listing 2).

```algorithm | Algorithm 2. Per-window slot lifecycle (`Engram.ingest`, `_absorb`, `_step_detected`; `src/uengram/engine.py:143-245`).
Input: whitened vector z, step t, raw-window index r
G ← active candidate, episodic and semantic slots
for each k in G: d²_k ← Σ_j (z_j − μ_kj)² / v_kj                  ▷ v: predictive variance
score ← min over semantic k of d²_k ;  rarity ← min over G of d²_k
alarm ← score > τ_alarm
if some d²_k ≤ τ then                                               ▷ window is inside a gate
    P ← semantic slots inside τ if any, else all slots inside τ
    k* ← argmin over P of d²_k + Σ_j log v_kj
    ABSORB(k*, z, t)
else
    if not MAKEEPISODICROOM(t, surprisal(rarity)) then return discard
    open a candidate slot at z with raw pointer r                   ▷ action new_candidate
candidates older than n_promote steps become episodic
every consolidate_every steps: merge duplicate semantic slots, checkpoint drift, supersede dormant ones
Procedure ABSORB(k, z, t)
α ← max(1/(n_k+1), β_min) if k is semantic, else 1/(n_k+1)
e ← (z − μ_k) / √v_k ;  update (μ_k, S_k) with rate α ;  n_k ← n_k + 1
if k is semantic then
    E_k ← λ·e / s_step + (1 − λ)·E_k                               ▷ MEWMA, λ = 0.1
    if n_k ≥ n_min and c·‖E_k‖² > h then split k; old slot superseded   ▷ step_change
    return
record hit t ;  (m, N) ← (5, 8) for a candidate, (20, 30) for an episodic slot
if the m-th latest hit is within N steps and semantic room exists then promote k   ▷ promote
```

```python | Listing 2. Competitive episodic admission (`src/uengram/lifecycle.py:112-142`; docstring trimmed).
def _episodic_importance(eng, idx, t: int) -> np.ndarray:
    age = t - eng.t_last[idx]
    return (eng.surpr[idx] * (1.0 + np.log(eng.n[idx]))
            * 2.0 ** (-age / eng.cfg.half_life_epi))


def make_episodic_room(eng, t: int, incoming_surprisal: float | None = None) -> bool:
    group = np.flatnonzero((eng.kind == CAND) | (eng.kind == EPI))
    if len(group) < eng.cfg.B_epi:
        return True
    epi = np.flatnonzero(eng.kind == EPI)
    if len(epi):
        importance = _episodic_importance(eng, epi, t)
        victim_pos = int(np.argmin(importance))
        threshold = eng.cfg.episodic_admission_ratio * importance[victim_pos]
        if incoming_surprisal <= threshold:
            return False
        victim = epi[victim_pos]
    else:
        cand = np.flatnonzero(eng.kind == CAND)
        victim = cand[np.argmin(eng.t_created[cand])]
    evict_slot(eng, int(victim), t, "evict_episodic")
    return True
```

### 2.3 Layered Persistence and Target Database Primitives

In the implemented `src/uengram/sensor_memory.py` adapter, uEngram manages live in-memory states while recording new candidates, state promotions, step changes, and alarm open/clear transitions into `EdgeMemory Lite` as versioned `EVENT` records. The persisted action set is `_PERSIST_ACTIONS = {new_candidate, promote, step_change}` plus alarm transitions (`src/uengram/sensor_memory.py:33`, `:112-113`).

![Figure 4. Sensor-state persistence pipeline in the implemented adapter. The raw window is pinned before the event write; on failure the pin is released and the error carries the pending record for retry.](figures/sensor-persistence-pipeline.svg)

**Table 1.** Implemented adapter stages: owning module, interface, and behavior when the stage fails.

| Stage | Owner | Interface | Failure behavior |
|---|---|---|---|
| Window schema and step order | `sensor_memory.py` (`SensorStateMemory.ingest`) with `validate_sensor_window` in `sensor_memory_records.py` | `SensorWindow` | `ValueError` raised before the engine consumes the window |
| Encoding and whitening | Host-selected encoder (`encoders.py` and `encoder_*.py` variants) | `encode(windows)` | Outside the adapter; not covered by the adapter's error handling |
| Live bounded memory and lifecycle events | `engine.py` (`ingest`, `recall`, `drain_events`) | Engine API | Window already consumed; recovery must retry storage, not re-ingest |
| Persist gate | `SensorStateMemory._should_persist` | Internal to `ingest` | Returns a receipt with `memory_id=None` and `persisted=False`; no storage write |
| Raw-window pin | Host `RawEvidencePins` implementation (`sensor_memory_types.py`) | `pin(ref, key)`, `unpin(ref, owner)` | Pin released if the event write fails, then the error is raised |
| Event storage | EdgeMemory Lite via `EdgeMemoryPort` (external) | `remember`, `recall`, `forget` | `SensorMemoryPersistenceError` carries `pending`; retry with `persist_pending` without re-ingesting |
| Retrieval bundle | Engine `recall` (exact state) and EdgeMemory `recall` (keyword history) combined in `SensorStateMemory.recall` | `recall(query_vector, mode=…)` | `ValueError` for an invalid mode or non-finite or mis-shaped query |
| Alarm combination policy | `alarm.py` (`AlarmLogic`) | `combine`, `error_k` | Not on the adapter's write path; the adapter persists only `decision.flagged` transitions |

Listing 3 is the write path behind the last two failure columns. The pin is taken before `remember`, released if `remember` raises, and the caller receives the pending record, so a retry never re-feeds the window into the engine.

```python | Listing 3. Pin-before-write persistence (`src/uengram/sensor_memory.py:120-123, 212-239`; asserts and the source, trust, privacy and retention arguments trimmed).
# in SensorStateMemory.ingest
try:
    return self._persist(pending, decision, window.step_index)
except Exception as exc:
    raise SensorMemoryPersistenceError(pending, decision, exc) from exc


def _persist(self, pending, decision, step_index):
    pinned = False
    if pending.raw_window_ref is not None:
        self.raw_evidence.pin(pending.raw_window_ref, pending.idempotency_key)
        pinned = True
    try:
        result = self.memory.remember(
            pending.content,
            kind=_EDGE_EVENT,
            structured_value=pending.structured_value,
            observed_at_ms=pending.observed_at_ms,
            idempotency_key=pending.idempotency_key,
            keywords=pending.keywords,
        )
    except Exception:
        if pinned:
            self.raw_evidence.unpin(pending.raw_window_ref, pending.idempotency_key)
        raise
```

Re-ingesting a window after a persistence failure is not allowed: the adapter returns the unpersisted records in `SensorMemoryPersistenceError.pending` for retry with `persist_pending` (`src/uengram/sensor_memory.py:37-50`), and pins raw waveform slices before the event write, releasing the pin if that write fails. Scaling this pipeline into a complete embedded "Sensor-State Memory" DBMS would require five **proposed** primitives. Generic slot merge, fixed capacities, and adapter pins are only precursors; none of the five is implemented as specified:

1. **Hybrid Temporal–Vector–Severity Retrieval:** Ranking stored memories $m$ for query $q$ via $S(q,m)=w_s\,\mathrm{sim}(q,v_m)+w_t\,e^{-\lambda\Delta t}+w_r\,g(r_m)$, returning individual score components alongside strict asset/mode metadata filters rather than an opaque scalar.
2. **Verified Centroid Compaction:** Merging confirmed routine windows within identical `(asset, mode, schema)` keys using exact pooled counts, means $\mu = (n_1\mu_1 + n_2\mu_2)/(n_1 + n_2)$, and covariances prior to source deletion.
3. **Keyed State Superseding:** Updating active baselines via exponential moving averages $\mu_t = (1-\beta)\mu_{t-1} + \beta x_t$ gated by quarantine rules so gradual wear is not absorbed as healthy behavior.
4. **Pinned Cross-Modal Raw Pointers:** Linking compact episodes to `(stream_id, start_offset, length, checksum)` slices in circular buffers with explicit expiration tracking.
5. **Quota-Bounded Eviction with Protected Quotas:** Enforcing deterministic high-water marks while shielding pinned critical episodes and active investigations from deletion.

An optional LLM belongs strictly downstream of retrieval to summarize structured evidence bundles; it must never sit in the real-time write path or control baseline admission.

Alarm logic in `src/uengram/alarm.py` can OR the gated statistical channel with an evidence-driven tripwire:

$$\text{Alarm} = \text{Novelty}_{\text{gated}} \lor \text{Tripwire}_{\text{evidence}}$$

With `AlarmLogic(combine="or", error_k=3)`, the external evidence stream opens a persisted tripwire over `error_k` windows. The tripwire is active only when a run receives a real evidence array, so an uncorroborated run cannot alarm on an implicit placeholder. The default `combine="and"` keeps the original gated behavior.

---

## 3. Baseline Benchmarks and Methodological Pitfalls

### 3.1 Summary of Core Benchmark Evidence

Across development and holdout studies [6], empirical performance diverges sharply between controlled synthetic streams and physical equipment.

**Table 2.** Core benchmark evidence across synthetic and field datasets.

| Dataset / Study | Configuration | Primary Result | Operational & Methodological Verdict |
|---|---|---|---|
| **Synthetic Stream** (10 seeds) | DSP ($d=16$) | 1.00 event recall; AP 0.9955; 86.2% write reduction (2,762 / 20,000 windows) | Validates bounded memory mechanics under known generative assumptions; CNN variant achieved only 0.262 recall despite high AUROC. |
| **CMAPSS FD001** | DSP ($d=16$) | Health AUROC 0.999, AP 0.995; RUL RMSE 42.1 (vs. 25.8 naive storage) | Separating early/late life (AUROC) does not yield accurate remaining-life regression (RMSE); $d=32$ failed due to singular frequency covariance. |
| **MetroPT-3** | Frozen CNN | Mean 3.6/4 events detected; 10.2 FA/week; negative mean lead time | 492 of 495 event windows fall in sustained load, which commissioning never contained. Clean windows vs. the 24 h before a report, within sustained load, give AUROC 0.07 (health) and 0.35 (novelty); this is a docs analysis (`docs/research-alarm-components.md:54-66`), not a result JSON. |
| **MetroPT-2** | Frozen v2 policy | 0/5 events detected across April (2) and January (3) subsets | Low false-alarm rate achieved at the cost of missing every target failure. |
| **SKAB** | Frozen DSP ($d=16$) | 5/34 events detected; AUROC 0.546; AP 0.419; detection delays 28–114 s post-onset | 5 false alarms over 6.21 clean hours makes weekly extrapolation (135 FA/wk) unstable. |
| **EngineAD** [2] | CNN ($d \in \{8,16,32\}$, 10 seeds) | Held-out truck pooled 5-minute segment AUROC near chance (d=16 median 0.556; seed-level JSON at `~/.local/state/uengram/enginead/cnn_seed_stability.json`, outside the repo) | Pre-fitted fleet-wide PCA channels leak preprocessing across splits, and 5-minute segment labels mix normal and faulty seconds. |
| **SCANIA Component X** | Frozen AND policy | 22.8% recall on classes 1–4 at final row; 19.3% alarm rate on class 0; AUROC 0.541 | Coarse repair-proximity classes on 5,046 vehicles cannot measure lead time or early warning. |

![Figure 5. EngineAD pooled held-out segment AUROC for ten training seeds at each latent width; each dot is one seed, not one test vehicle.](figures/enginead-seed-histogram.svg)

### 3.2 Critical Error Post-Mortems

Several benchmark failures exposed traps that routinely distort published anomaly-detection claims:

* **Retraction of the NASA IMS "63.8-Hour Early Warning" Claim [7]:** A prior project-session recap and draft section reported that an "optimal strategy" on the NASA/IMS bearing dataset achieved 63.8 hours of early warning and zero false alarms over four healthy weeks. Inspection of the external script invalidated both claims: (1) the script hardcoded assumed degradation row indices in Python and evaluated lead time against its own hardcoded indices using the final sample as failure reference, without adjudicated physical fault-onset timestamps; and (2) "zero alarms" came from the assumed healthy prefix of a single campaign. Those claims are retracted in full (`docs/sensor-state-memory-and-anamoly-detection.md:266-270`).
* **Encoder Width Regressions:** Sweeping DSP width from $d=16$ to $d=36$ on synthetic streams dropped mean event recall from 1.00 to 0.358 and mean AP from 0.9955 to 0.565 (10 seeds each; $d=36$ from `results/synth_d36_runs.jsonl`, median AP 0.558; $d=16$ from `results/synth_runs.jsonl`). On SKAB, an exploratory $d=48$ sweep (`results/skab_d_sweep.json`) detected 9/34 events, against 5/34 for the frozen $d=16$ detector. The sweeps do not identify one embedding dimension that is best across datasets.
* **Threshold Tuning Cannot Fix Missing State Coverage:** A parameter sweep over MetroPT-3 alarm policies failed to find any setting meeting all three operational criteria (4/4 events detected, $\le 2\%$ clean alarm time, $\le 1$ FA/week). Forcing 4/4 detection required enabling a health path for unbaselined states, which inflated clean alarm time to $\ge 4.6\%$.

---

## 4. Representation Depth and Foundation-Model Distillation

### 4.1 Intermediate CNN Readouts: Proxy Screen vs. Full Pipeline

Recent literature suggests that intermediate network layers and patch prototypes preserve localized anomaly signals that final reconstruction bottlenecks smooth over, including VGG middle-layer game-theoretic importance (LayerIM) [12], layer-16 foundation representations in TimeRep [13], and patch memory gates in MOMEMTO [15].

To test this without test-label leakage, we extended `CNNEncoder` with a `representation="multiscale"` option that pools and concatenates first- and second-stage convolutional feature maps ($r(x) = [\operatorname{pool}(h_1(x)); \operatorname{pool}(h_2(x))]$) before PCA and noise whitening. We first ran a label-free synthetic screen across 5 neural seeds (384 training windows, 96 selection windows injected with 5 fault families: level shift, drift, impulse, oscillation, freeze). On that proxy ruler, the middle-layer readout ($d=32$, 30 epochs) ranked first with a median worst-family displacement score of **0.610** (vs. **0.373** for multi-level and **0.356** for the final bottleneck), while training for 90 epochs degraded every configuration by over-specializing on waveform reconstruction.

However, running those exact five seeds through the full calibrated alarm pipeline (`bench/cnn_stability.py`, $d=32$, 30 epochs, frozen v2 policy) overturned the proxy screen:

**Table 3.** Intermediate CNN readouts through the full calibrated alarm pipeline (5 seeds, $d=32$, 30 epochs; `results/cnn_intermediate_pipeline.json`).

| Dataset | Readout | Events Detected (med, min–max) | AUROC (med, min–max) | AP (med, min–max) | False Alarms/Wk (med) | Clean Alarm Share (med) |
|---|---|---:|---:|---:|---:|---:|
| **SKAB** | Bottleneck (plain) | **23.0** (18–34) | **0.623** (0.480–0.675) | **0.511** (0.441–0.585) | 189.25 | 0.480 |
| **SKAB** | Middle layer | 20.0 (18–29) | 0.570 (0.540–0.660) | 0.505 (0.448–0.538) | 243.32 | 0.440 |
| **SKAB** | Multi-level | 20.0 (13–33) | 0.571 (0.553–0.687) | 0.504 (0.467–0.518) | 243.32 | **0.310** |
| **MetroPT-3** | Bottleneck (plain) | 4.0 (3–4) | 0.989 (0.987–0.996) | 0.676 (0.485–0.959) | 0.69 | 0.022 |
| **MetroPT-3** | Middle layer | 3.0 (3–4) | 0.990 (0.987–0.992) | 0.617 (0.517–0.671) | **0.58** | **0.019** |
| **MetroPT-3** | Multi-level | **4.0** (4–4) | **0.992** (0.988–0.996) | **0.773** (0.541–0.956) | **0.58** | 0.025 |

In the full pipeline the middle layer had the lower event median on both datasets (SKAB 20 vs. 23; MetroPT-3 3 vs. 4), though on MetroPT-3 its median AUROC was marginally higher (0.990 vs. 0.989) and its false-alarm rate lower (0.58 vs. 0.69 per week). Seed spread was wide on SKAB (13–34 events across all readouts; 18–34 for the bottleneck alone), and no readout dominated across modalities. Accordingly, `representation="bottleneck"` remains the default encoder.

### 4.2 Offline Foundation-Model Distillation

Because deploying 100M+ parameter time-series foundation models directly on microcontrollers or gateways violates memory and latency constraints [16], `src/uengram/teacher_transfer.py` implements **offline teacher-to-student distillation**. During commissioning, causal right-aligned windows $X_t$ are passed to one or $M$ frozen teachers (such as MOMENT-1-small [18], 141 MB, and Chronos-2 [22], 478 MB) while a 46,144-byte student CNN ($S_\theta$) minimizes reconstruction error alongside dimension-normalized feature matching:

$$\mathcal{L}(\theta) = \operatorname{MSE}\big(D(S_\theta(X_t)), X_t\big) + \lambda \frac{1}{B}\sum_{b=1}^{B}\operatorname{MSE}_b\big(H(S_\theta(X_t)), T(X_t)\big) + \lambda_{\mathrm{orth}}\operatorname{mean}\big((W W^\top - I)^2\big)$$

The loss is implemented in the `CNNEncoder` training loop (`src/uengram/encoder_cnn.py`). $\operatorname{MSE}_b$ averages over the $b$-th teacher feature block, and $B$ is the number of feature blocks.

Listing 4 shows the training step. Teacher targets are standardised per feature before matching (`src/uengram/encoder_cnn.py:162-168`), and when several teachers are stacked each feature block is averaged separately, so a 514-dimensional MOMENT block cannot outweigh a 32-dimensional Chronos-2 block.

```python | Listing 4. Reconstruction, orthogonality and block-averaged teacher loss in one training step (`src/uengram/encoder_cnn.py:201-229`; reflowed).
z = self._forward_enc(xb)
recon = torch.mean((dec(z).reshape(-1, C, K) - xb) ** 2)
loss = recon
if self.ortho_beta > 0.0:
    W = self.fc.weight
    loss = loss + self.ortho_beta * torch.mean((W @ W.T - I) ** 2)
if target is not None:
    student = self.teacher_head(self._readout(xb))
    if teacher_features is not None and teacher_features.block_slices:
        teacher_loss = torch.stack([
            torch.mean((student[:, start:end] - target[indices, start:end]) ** 2)
            for start, end in teacher_features.block_slices
        ]).mean()
    else:
        teacher_loss = torch.mean((student - target[indices]) ** 2)
    loss = loss + teacher_weight * teacher_loss
opt.zero_grad()
loss.backward()
opt.step()
```

Because foundation models are often applied after per-window z-normalization, which erases absolute pressure or vibration levels, we instead appended raw per-channel statistics to the teacher embedding: $T^+(X) = [T(X), \mu_1..\mu_C, \sigma_1..\sigma_C]$ (per-window mean and standard deviation; `src/uengram/moment_teacher.py`). The teacher input is not z-normalized. We also tested left-edge padding versus linear time resampling when the edge window $K$ is shorter than the teacher's native context $L$.

### 4.3 Cross-Dataset Distillation Results

Evaluating single- and dual-teacher distillation across six benchmark gates revealed that higher foundation-model AUROC frequently masks severe operational alarm failures:

1. **MetroPT-3 vs. Synthetic Context Mismatch:** Distilling MOMENT-1-small into the student improved MetroPT-3 test AP from 0.6225 to 0.7942 (AUROC 0.9756 to 0.9775), but degraded synthetic 4-family probe AUROC from 0.6872 (ordinary CNN) to 0.5035 (edge-padded) and 0.5362 (resampled + instance statistics).
2. **TSB-AD-U External Holdout (6 series [21]):** On the six TSB-AD-U series, resampled MOMENT + statistics edged out the ordinary CNN on pooled AUROC (0.9157 vs. 0.9111), where long series dominate, but lost on equal-series AUROC (0.6316 vs. 0.6476).
3. **MFPT [23] and CWRU [24] Bearing Gates (Ranking vs. Calibration):** Combining MOMENT (514 dims) and Chronos-2 (32 quantile/median dims) into a 46 KB student raised MFPT window AUROC from 0.7948 to 0.8253 (file AUROC 0.85 to 0.90) and achieved 1.0000 AUROC across all models on CWRU. Per-window encode time for the two-teacher student was 0.0606 ms on MFPT and 0.0635 ms on CWRU (`encode_test_ms_per_window` in `results/mfpt_two_teacher_gate.json` and `results/cwru_two_teacher_gate.json`). However, every model flagged the single held-out normal file on both datasets (normal-file false-alarm rate 1.0000; one normal file per dataset in the same two JSONs), because training-normal distance thresholds treated unseen healthy load regimes as faults.
4. **CMAPSS FD001 Run-to-Failure Gate [25] (AUROC vs. Timely Maintenance Lead):** Training on engines 1–40 (first 30 windows) and evaluating through failure on engines 41–50 (requiring 3 consecutive anomalous windows), teacher-distilled students achieved higher window AUROC (0.9890–0.9902 vs. 0.9823 for the ordinary CNN). Yet by the RUL 20 maintenance horizon, the ordinary CNN caught **70% (7/10)** of failing engines on time, whereas MOMENT and Two-Teacher students caught only **20% (2/10)**.
5. **SKAB Wall-Clock Gate (Single-Channel vs. 8-Channel Fusion):** Under a 30-minute CPU cap, single-channel teacher runs caught **0/34** events strictly within fault intervals (AUROC 0.463–0.471, 5.47–8.75 FA/hr). Removing the teachers to run all 8 channels through the multiscale CNN cut healthy false alarms by 58.3% (from 6.57 to 2.74 FA/hr) and raised AUROC to 0.5270, but strict in-interval event recall remained **0/34**.

![Figure 6. TSB-AD-U pooled versus equal-series AUROC and average precision for six frozen series (`results/moment_tsb_ad_holdout.json`).](figures/tsb-holdout-comparison.svg)

---

## 5. Role-Separated Encoders and Domain Specialists

### 5.1 Role Separation: Mode, Level, Dynamic, and Health

The failure of monolithic encoders on SKAB and multi-channel telemetry stems from mixing incompatible physical variations into a single distance metric. In `src/uengram/role_architecture.py`, we decompose telemetry into four explicit branches:

* **Mode:** Operating context (idle, startup, load, transition), used solely for conditioning rather than voting on alarms.
* **Level:** Per-channel absolute levels, robust spread, and short-term trends.
* **Dynamic:** Deterministic multiscale frequency and transient shifts.
* **Health:** Distance from a frozen commissioning reference.

The Level, Dynamic, and Health branches are calibrated independently on healthy data and fused via empirical tail probabilities. Only operator-confirmed normal windows enter semantic memory; unconfirmed normal windows route to quarantine, and alarmed windows route to episodic memory. On the 8-channel SKAB split, where all three CNN representations scored 0/34 strict recall, the role-separated architecture achieved **10/34 strict recall** at 6.19 FA/hr (`at_least_2` policy) and **8/34 strict recall** at **1.59 FA/hr** (`dynamic_health` policy).

Each branch score $s$ becomes an upper-tail p-value against $n$ scores $c_1 \dots c_n$ from a separate healthy calibration stream (`EmpiricalCalibrator`, `src/uengram/role_architecture.py:47-55`):

$$p(s) = \frac{1 + \#\{\, i : c_i \ge s \,\}}{n + 1}$$

A branch fires when $p \le \alpha$. Listing 5 shows how the fusion policies combine the three alarm branches and how a window is routed; the mode branch never votes.

```python | Listing 5. Branch fusion policies and memory routing (`src/uengram/role_architecture.py:168-188`; signature reflowed).
flags = np.stack([np.asarray(p_values[name]) <= alpha
                  for name in cls._ALARM_BRANCHES], axis=1)
if policy == "any":
    return flags.any(axis=1)
if policy == "at_least_2":
    return flags.sum(axis=1) >= 2
if policy == "dynamic_health":
    return flags[:, 1] & flags[:, 2]
return flags.all(axis=1)


def memory_route(p_values, alpha, *, alarm_open, data_quality_ok,
                 operator_confirmed_normal=False) -> str:
    """Route evidence without allowing unconfirmed data to poison normal memory."""
    if not data_quality_ok:
        return "QUARANTINE"
    if alarm_open or any(float(p_values.get(name, np.nan)) <= alpha
                         for name in RoleBasedArchitecture._ALARM_BRANCHES):
        return "EPISODIC"
    if operator_confirmed_normal:
        return "SEMANTIC"
    return "QUARANTINE"
```

### 5.2 Pre-Fit Numerical Diagnostics and Domain Specialists on OS-Metrics

To test role separation on multi-subsystem operational telemetry, we evaluated the **OS-Metrics** dataset (Zenodo 17981352; 40,320 15-second samples over 7 days across CPU, memory, and network channels with 42 annotated disturbances).

Initial 6-channel ingestion crashed with `LinAlgError: Singular matrix` during noise whitening. To prevent silent numerical instability or crashes, we built a pre-fit diagnostic module (`src/uengram/diagnostics.py`) that inspects covariance rank, condition number, sentinel rates, and null-space collinearity before fitting. It automatically flagged the 6-channel matrix as `singular` (rank 5/6, condition number $7.15 \times 10^{15}$) because `network_response_time_diff` was an exact linear combination of streams `a` and `b`, while certifying the 5 primitive channels as `moderately_conditioned` (rank 5/5, condition number $4.76 \times 10^8$).

```python | Listing 6. Rank, conditioning and null-space checks in `diagnose_channels` (`src/uengram/diagnostics.py:118-166`; empty-input guards and the perfect-correlation scan trimmed).
evals, evecs = np.linalg.eigh(cov)
evals = np.maximum(evals, 0.0)
order = np.argsort(evals)[::-1]
evals, evecs = evals[order], evecs[:, order]
max_ev, min_ev = float(evals[0]), float(evals[-1])
rcond_eps = np.finfo(np.float64).eps * max(max_ev, 1e-12) * n_channels
effective_rank = int(np.sum(evals > rcond_eps))
cond = float(max_ev / max(min_ev, 1e-18))

redundant_groups = []
if effective_rank < n_channels:
    null_vecs = evecs[:, effective_rank:]
    for j in range(null_vecs.shape[1]):
        v = null_vecs[:, j]
        group = tuple(names[i] for i in np.where(np.abs(v) > 0.1)[0])
        if group and group not in redundant_groups:
            redundant_groups.append(group)

if effective_rank < n_channels or cond > 1e12:
    status = "singular" if effective_rank < n_channels else "ill_conditioned"
elif cond > 1e6:
    status = "moderately_conditioned"
else:
    status = "well_conditioned"
```

Even on the 5 primitive channels, pooling all metrics into one global role encoder underperformed the DSP baseline (24/42 recall at 42.0 FA/wk vs. DSP's 27/42 at 26.6 FA/wk) because localized subsystem spikes were diluted by quiet channels. Partitioning the telemetry into dedicated **Domain Specialists** (`src/uengram/domain_specialists.py`—independent CPU, Memory, and Network role encoders fused via a calibrated `or_dynamic_health` rule) resolved the dilution:

```algorithm | Algorithm 3. Domain-specialist scoring, fusion and routing (`DomainSpecialistArchitecture.score`, `fuse`, `memory_route`; `src/uengram/domain_specialists.py:128-219`).
Input: window x; specialists s with channel sets C_s (CPU, Memory, Network); level α (default 0.01)
Output: alarm, triggering specialists A, memory route
for each specialist s do
    p_s,b ← calibrated p-value of branch b ∈ {level, dynamic, health} on x[C_s]
    flag_s ← (p_s,dynamic ≤ α) and (p_s,health ≤ α)                ▷ dynamic_health branch policy
alarm ← flag_CPU or flag_Memory or flag_Network                    ▷ or specialist rule
A ← { s : flag_s }                                                  ▷ root-cause attribution
score ← Σ_s −log min_b p_s,b                                        ▷ continuous fused score
if data quality fails then route ← QUARANTINE
else if alarm is open or A ≠ ∅ then route ← EPISODIC (with A)
else if operator confirmed normal then route ← SEMANTIC
else route ← QUARANTINE
```

**Table 4.** OS-Metrics chronological split: DSP baseline, all-channel role encoder, and domain specialists (`results/osmetrics_specialist_gate.json`; all-channel latency from `results/osmetrics_role_gate.json`).

| Architecture (OS-Metrics Chronological Split) | Strict Event Recall | 2-Hour Recall | False Alarms / Week | Window AUROC | Root-Cause Attribution | Latency / Window |
|---|---:|---:|---:|---:|---:|---:|
| DSP ($d=16$) Baseline | 27/42 (64.3%) | 30/42 (71.4%) | 26.6 | 0.8845 | N/A | — |
| All-Channel Role (`dynamic_health`) | 24/42 (57.1%) | 30/42 (71.4%) | 42.0 | 0.8731 | N/A | $6.80\,\mu\text{s}$ |
| **Domain Specialists (`or_dynamic_health`)** | **29/42 (69.0%)** | **29/42 (69.0%)** | **19.6** | **0.8919** | **30/30 (100.0%)** | **$6.27\,\mu\text{s}$** |

Domain Specialists achieved the highest strict recall (69.0%), the lowest false-alarm rate (19.6/week), and 100% subsystem attribution accuracy (30/30 test-split events, `results/osmetrics_specialist_gate.json`) while maintaining a $6.27\,\mu\text{s}$ per-window execution time.

![Figure 7. Strict event recall of domain specialists and role separation against monolithic encoders on OS-Metrics (A) and SKAB (B).](figures/domain-specialist-comparison.svg)

Per-subsystem results (from `results/osmetrics_specialist_gate.json`) show where each detection comes from. The per-subsystem event counts sum to the 42 disturbances (13 + 12 + 17):

**Table 5.** Per-specialist results on OS-Metrics (`results/osmetrics_specialist_gate.json`).

| Specialist | Assigned channels | Strict event recall | False-alarm episodes | Window AUROC |
|---|---|---:|---:|---:|
| CPU | `cpu_usage` | 7/13 (53.8%) | 6 | 0.9589 |
| Memory | `memory_usage`, `memory_used` | 8/12 (66.7%) | 8 | 0.9641 |
| Network | `network_response_time_a`, `network_response_time_b` | 14/17 (82.4%) | 12 | 0.8924 |
| Fused (`or_dynamic_health`) | all 5 primitive channels | 29/42 (69.0%) | 14 (19.6/week) | 0.8919 |

In isolated incidents, such as `mem_1765646427688_2393`, only the memory specialist fired. In cascading disruptions, such as `mem_1765654018485_7936`, all three specialists fired. Per-specialist firing for these two events is not reproduced in the archived result files, so both are illustrative case notes, not verified results.

---

## 6. Discussion, Limitations, and Deployment Gate

This study makes no "first-ever" algorithmic claim; it builds on established analytical redundancy [4, 5], noise whitening [3], streaming memory [8–11], and recent time-series benchmarks [18–26]. Its primary contribution is architectural and methodological: demonstrating how bounded sensor-state memory must separate mode, level, and dynamic roles per subsystem while guarding against baseline poisoning and deceptive benchmark metrics.

Before transitioning uEngram from a Python reference design to a safety-relevant embedded C/Rust DBMS deployment, five engineering gates remain mandatory:

1. **Preregistered Multi-Asset Field Cohort:** Evaluate on held-out physical assets with independently adjudicated fault-onset timestamps and complete operating-mode coverage (startup, idle, full load, post-repair), preventing preprocessing leakage across vehicles or machines.
2. **Adversarial Poisoning and Shadow Baselines:** Replay slow degradation ramps against a frozen shadow reference to verify that quarantine gates prevent gradual wear from updating semantic healthy centroids.
3. **Abstention as a First-Class Output:** Explicitly emit `unknown_operating_mode` or `insufficient_calibration_support` rather than forcing out-of-distribution loads into false alarms or healthy summaries.
4. **Transactional Cross-Store Crash Recovery:** Implement an atomic write-ahead outbox unifying uEngram checkpoint state, `EdgeMemory` event records, and circular-buffer pin reference counts across power cuts.
5. **Hardware Storage Profiling:** Measure p99 latency, peak RAM, flash write amplification, and ANN/exact recall under high-water-mark eviction on target embedded flash hardware.

---

## References

1. Chhikara, P., Khant, D., Aryan, S., Singh, T., and Yadav, D. (2025). “Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory.” arXiv:2504.19413.
2. Hojjati, H., Roth, C., Woods, R., Sills, K., and Armanfard, N. (2026). “EngineAD: A Real-World Vehicle Engine Anomaly Detection Dataset.” arXiv:2603.25955.
3. Green, A. A., Berman, M., Switzer, P., and Craig, M. D. (1988). “A Transformation for Ordering Multispectral Data in Terms of Image Quality with Implications for Noise Removal.” *IEEE Trans. Geosci. Remote Sens.*, 26(1), 65–74.
4. Willsky, A. S. (1976). “A Survey of Design Methods for Failure Detection in Dynamic Systems.” *Automatica*, 12(6), 601–611.
5. Chow, E. Y., and Willsky, A. S. (1984). “Analytical Redundancy and the Design of Robust Failure Detection Systems.” *IEEE Trans. Autom. Control*, 29(7), 603–614.
6. uEngram repository documentation and benchmark artifacts (`docs/RESULTS.md`, `docs/sensor-state-memory-pipeline.md`, `docs/research-teacher-distillation.md`, `docs/osmetrics-specialist-gate.md`).
7. NASA IMS Bearings Dataset, NASA Open Data Portal.
8. Bhatia, S., Jain, A., Srivastava, S., Kawaguchi, K., and Hooi, B. (2022). “MemStream: Memory-Based Streaming Anomaly Detection.” *Proc. ACM Web Conf. 2022*.
9. Bang, J., Kim, H., Yoo, Y., Ha, J.-W., and Choi, J. (2021). “Rainbow Memory: Continual Learning With a Memory of Diverse Samples.” *CVPR 2021*, 8218–8227.
10. Elenter, J., NaderiAlizadeh, N., Javidi, T., and Ribeiro, A. (2024). “Primal Dual Continual Learning: Balancing Stability and Plasticity through Adaptive Memory Allocation.” arXiv:2310.00154v2.
11. Aghasanli, A., Li, Y., and Angelov, P. (2025). “Prototype-Based Continual Learning with Label-free Replay Buffer and Cluster Preservation Loss.” arXiv:2504.07240.
12. Yang, L., Lu, L., Liu, C., et al. (2025). “Interactive exploration of CNN interpretability via coalitional game theory.” *Scientific Reports*, 15, 9261.
13. Han, C. S., and Lee, K. M. (2025). “Leveraging Intermediate Representations of Time Series Foundation Models for Anomaly Detection.” arXiv:2509.12650.
14. Park, J., and Kang, S. (2026). “PaAno: Patch-Based Representation Learning for Time-Series Anomaly Detection.” arXiv:2602.01359v2.
15. Yoon, S., Kim, J., Ha, J., and Ko, Y. M. (2026). “Patch-based Memory Gate Model in Time Series Foundation Model.” arXiv:2509.18751v4.
16. dos Santos Oliveira, A., da Silva, F. S. T., Ferreira, D. L., et al. (2026). “Looking at anomaly detection in EdgeAI through the up-to-date lens of academic and market perspectives.” *Cluster Computing*, 29, 251.
17. Kim, M., Bhaumik, K. K., Ali, A. A., and Woo, S. S. (2024). “MIXAD: Memory-Induced Explainable Time Series Anomaly Detection.” arXiv:2410.22735.
18. Goswami, M., et al. (2024). “MOMENT: A Family of Open Time-series Foundation Models.” *ICML 2024*.
19. Kim, J., Yoon, S., Ko, Y. M., Kim, Y., Kim, S. I., and Tae, J. (2026). “Complexity- and Statistics-Guided Anomaly Detection in Time Series Foundation Models.” *ICLR 2026*.
20. Khan, U., Capogrosso, L., Biondani, F., Magno, M., and Fummi, F. (2026). “ChronosAD: Leveraging Time Series Foundation Models for Accurate Anomaly Detection.” arXiv:2606.01300.
21. TSB-AD Project. “A Comprehensive Benchmark for Time-Series Anomaly Detection.” [https://thedatumorg.github.io/TSB-AD/](https://thedatumorg.github.io/TSB-AD/).
22. Ansari, A. F., et al. (2025). “Chronos-2: From univariate to multivariate probabilistic time series forecasting.” arXiv:2510.15821.
23. Bechhoefer, E. (2013). “A Data-Driven Approach to Predictive Maintenance of Industrial Equipment.” MFPT Fault Data Sets.
24. Case Western Reserve University Bearing Data Center.
25. NASA. “Turbofan Engine Degradation Simulation Data Set (CMAPSS FD001).” NASA Open Data Portal.
26. Si, H., Li, J., Pei, C., et al. (2024). “TimeSeriesBench: An Industrial-Grade Benchmark for Time Series Anomaly Detection Models.” arXiv:2402.10802v3.

