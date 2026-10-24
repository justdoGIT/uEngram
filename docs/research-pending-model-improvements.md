# Remaining model-improvement research after the CNN stability study

**Research date:** 2026-10-25  
**Status:** exploratory development evidence, not independent confirmation

## 1. Decision

There is still no single "perfect" encoder for uEngram. The new evidence is
strong enough to reject one tempting promotion: the current
`CNNEncoder(d=32, epochs=90, pool="maxavg", ortho_beta=0.05)` should remain an
experimental option, not replace the deterministic encoder.

The main findings are:

1. **The earlier seed-stability claim was too strong.** Across ten SKAB seeds,
   the max-average, orthogonally regularized CNN detects 7--16 of 34 faults.
   Its median is 10. Clean alarm time ranges from 2.45% to 8.82%.
2. **Activation decorrelation does not solve the problem.** It improves some
   ranking medians on SKAB, but its detection range remains 7--17 and its
   worst clean alarm time rises to 11.30%.
3. **Late weight averaging is not a clear win.** It narrows the five-seed SKAB
   false-episode range, but lowers median detection and still has a wide AP
   range. It also fails on MetroPT-3.
4. **The SKAB-specific CNN changes do not transfer to MetroPT-3.** Every tested
   CNN variant detects 0 of 4 MetroPT-3 events under both the native frozen
   alarm policy and a score-only comparison policy.
5. **Seed-score ensembles remove the need to pick one seed, but not the model
   weakness.** A five-seed median ensemble gives 13/34 on SKAB at five times
   the encoder work, yet still gives 0/4 on MetroPT-3.
6. **Naive DSP--CNN fusion is rejected.** Max and mean fusion do not produce a
   consistent cross-dataset gain and can increase the false-alarm burden.

No production model was changed. The highest-value next work is not another
isolated CNN loss term. It is a representation split that keeps operating
level, fast dynamics, slow health, and physical residual evidence separate,
followed by a frozen asset-separated evaluation.

## 2. What was still unresolved in the repository

The repository already contains enough negative evidence to narrow the search.
These are model problems, not threshold-tuning problems.

| Unresolved problem | Repository evidence | Consequence |
|---|---|---|
| A fault state is absent from commissioning | MetroPT-3 has almost no sustained-load reference data before test | A statistical distance cannot distinguish an unseen healthy load state from a faulty one |
| No within-state pre-failure separation | Within sustained load, prior work found novelty and health ranking at or below chance for the pre-report split | A better threshold cannot create sensor information that is absent |
| Seasonal calibration drift | MetroPT channel error rates change sharply by month | One frozen calibration week is not a stable year-round reference |
| Health is a good ranking signal but not a validated alarm | Earlier MetroPT and C-MAPSS studies found strong health AUROC, but the health alarm was not independently tested | Novelty and degradation should not be forced into one score |
| Feature dimension is domain dependent | Full DSP dimension helps vibration streams but fails on low-rank or tabular streams | One global latent dimension is not defensible |
| Neural results depend on initialization | Earlier three-seed SKAB work and the new ten-seed run disagree on the apparent stability | A single seed is model selection by accident |
| All current benchmark streams are consumed | Their labels or outcomes have influenced design choices | Further replay is development evidence, not confirmation |
| Alarm actions change score semantics | uEngram can absorb, promote, or create a candidate instead of emitting an ordinary score | Encoder ranking and deployed alarm behavior must both be reported |

The practical implication is important: a higher AUROC can coexist with zero
detected events. The operating threshold, persistence, state action, and event
shape decide whether the ranking gain reaches an operator.

## 3. New experiment

### 3.1 Questions

The experiment asked four narrow questions:

1. Does penalizing correlation between **actual bottleneck activations** improve
   seed robustness beyond the existing orthogonal weight penalty?
2. Does averaging late encoder weights give a stable single model?
3. Does combining calibrated scores from several seeds remove seed-selection
   risk?
4. Does a simple DSP--CNN late fusion preserve the strengths of both encoders?

### 3.2 Data boundary

| Dataset | Encoder fit | Calibration | Test | Role |
|---|---:|---:|---:|---|
| SKAB | 282 windows | 658 windows | 3,722 windows | Consumed development data |
| MetroPT-3 | 2,658 windows | 911 windows | 21,556 windows | Consumed development data |

SKAB has only about 6.21 hours of observed clean test time. Its per-week false
alarm rate is therefore very unstable. The report emphasizes false-alarm
episodes and clean-time share for SKAB instead of treating the extrapolated
weekly rate as precise.

### 3.3 Models

All neural models use `d=32`, 90 epochs, max-average temporal pooling, and the
existing `ortho_beta=0.05` output-weight penalty.

| Name | Additional change | Seeds |
|---|---|---:|
| Base | No additional change | SKAB 0--9; MetroPT-3 0--2 |
| Activation correlation | Add `0.05` times mean squared off-diagonal activation correlation | SKAB 0--4; MetroPT-3 0--2 |
| Late weight average | Arithmetic mean of encoder weights from epochs 60--90 | SKAB 0--4; MetroPT-3 0--2 |

The late-weight-average variant is **SWA-inspired**, not canonical SWA.
Canonical SWA uses a deliberate constant or cyclical learning-rate trajectory.
This experiment retained the existing AdamW schedule so that only the late
weight average changed.

The activation penalty is also not full VICReg. VICReg includes paired-view
invariance and per-dimension variance terms in addition to covariance control.
Here only the off-diagonal activation-correlation term was tested.

### 3.4 Evaluation contracts

Two alarm views were retained:

- **Native:** the frozen uEngram policy, including memory actions such as new
  candidate, absorb, promotion, and score.
- **Score-only:** every action is replaced with `score` and
  `new_state_alert=False`. This view exists only to compare fused scores under
  one common action contract. It is not a drop-in result for the deployed
  engine.

Each component's novelty score is converted to empirical upper-tail surprisal
using the existing calibration split. Median, Fisher-style sum, maximum, and
mean combinations are then calibrated again on that same split. The seed
scores are correlated, so no Fisher independence theorem or formal combined
p-value guarantee is claimed.

No coefficient or fusion was selected using test labels. All tested variants,
including negative results, are retained in
`results/cnn_stability_ablation.json`.

### 3.5 Reproduction

The permanent harness is `bench/cnn_stability.py`; the deterministic merger is
`bench/cnn_stability_merge.py`.

```bash
uv sync --extra cnn

PYTHONPATH=src:. .venv/bin/python -m bench.cnn_stability \
  --datasets skab --variants base,actcorr,swa --seeds 0,1,2,3,4 \
  --out ~/.local/state/uengram/cnn_pending_skab.json

PYTHONPATH=src:. .venv/bin/python -m bench.cnn_stability \
  --datasets skab --variants base --seeds 5,6,7,8,9 \
  --out ~/.local/state/uengram/cnn_pending_skab_extra.json

PYTHONPATH=src:. .venv/bin/python -m bench.cnn_stability \
  --datasets metropt3 --variants base,actcorr,swa --seeds 0,1,2 \
  --out ~/.local/state/uengram/cnn_pending_metropt3.json

PYTHONPATH=src:. .venv/bin/python -m bench.cnn_stability_merge \
  --skab-primary ~/.local/state/uengram/cnn_pending_skab.json \
  --skab-extra ~/.local/state/uengram/cnn_pending_skab_extra.json \
  --metropt3 ~/.local/state/uengram/cnn_pending_metropt3.json \
  --out results/cnn_stability_ablation.json
```

## 4. Results

### 4.1 Ten-seed base CNN on SKAB

| Seed | Events / 34 | AUROC | AP | False episodes | Clean alarm time |
|---:|---:|---:|---:|---:|---:|
| 0 | 14 | 0.615 | 0.505 | 9 | 8.59% |
| 1 | 13 | 0.643 | 0.544 | 4 | 2.45% |
| 2 | 16 | 0.628 | 0.532 | 9 | 4.69% |
| 3 | 10 | 0.620 | 0.540 | 4 | 2.87% |
| 4 | 7 | 0.619 | 0.499 | 5 | 3.17% |
| 5 | 7 | 0.610 | 0.506 | 4 | 2.49% |
| 6 | 9 | 0.631 | 0.533 | 5 | 3.59% |
| 7 | 9 | 0.663 | 0.568 | 4 | 5.03% |
| 8 | 15 | 0.608 | 0.534 | 7 | 8.82% |
| 9 | 10 | 0.676 | 0.592 | 4 | 2.45% |
| **Median [range]** | **10 [7, 16]** | **0.624 [0.608, 0.676]** | **0.534 [0.499, 0.592]** | **4.5 [4, 9]** | **3.38% [2.45%, 8.82%]** |

Only 7 fault files are detected by every seed; the union is 17 files. This is
not small numerical jitter. The initialization changes which fault families
reach the operational threshold.

The original three-seed subset happened to be optimistic: it gave 13--16
events. Extending to ten seeds changes the median to 10 and exposes two 7-event
runs. Three seeds were enough to find variance but not enough to characterize
it.

### 4.2 Stabilization ablations on SKAB

Values are median `[minimum, maximum]` across seeds.

| Variant | Seeds | Events / 34 | AUROC | AP | False episodes | Clean alarm time |
|---|---:|---:|---:|---:|---:|---:|
| Base | 10 | 10 `[7, 16]` | 0.624 `[0.608, 0.676]` | 0.534 `[0.499, 0.592]` | 4.5 `[4, 9]` | 3.38% `[2.45%, 8.82%]` |
| Activation correlation | 5 | 12 `[7, 17]` | 0.635 `[0.608, 0.640]` | 0.534 `[0.527, 0.561]` | 5 `[4, 9]` | 2.50% `[1.78%, 11.30%]` |
| Late weight average | 5 | 11 `[7, 12]` | 0.640 `[0.602, 0.676]` | 0.537 `[0.477, 0.607]` | 4 `[4, 6]` | 2.74% `[2.16%, 6.31%]` |

Activation correlation has no robust operational advantage. Its median AUROC
is higher than the ten-seed base median, but its event range is still wide and
its worst clean-time share is worse.

Late weight averaging makes the detected event sets somewhat more alike in the
five-seed screen: median pairwise Jaccard similarity is 0.690, versus 0.634 for
the same five base seeds. That stability is bought by reducing the event union
and median detection. It stabilizes a weaker operating point rather than
solving the task.

### 4.3 Cross-dataset result on MetroPT-3

Values are median `[minimum, maximum]` over three seeds under the native frozen
policy.

| Variant | Events / 4 | AUROC | AP | False episodes | Clean alarm time |
|---|---:|---:|---:|---:|---:|
| Base maxavg + ortho | 0 `[0, 0]` | 0.611 `[0.595, 0.669]` | 0.029 `[0.029, 0.039]` | 0 `[0, 0]` | 0% |
| Activation correlation | 0 `[0, 0]` | 0.649 `[0.542, 0.806]` | 0.034 `[0.028, 0.216]` | 0 `[0, 0]` | 0% |
| Late weight average | 0 `[0, 0]` | 0.672 `[0.568, 0.683]` | 0.034 `[0.026, 0.035]` | 0 `[0, 0]` | 0% |

The zero false-episode result is threshold silence, not evidence of useful
specificity: no CNN run crosses the operational threshold for a real event.

For comparison, a freshly encoded `DSPEncoder(d=42)` in this harness has AUROC
0.991 and AP 0.726. This is not the immutable cached v2 confirmation baseline
reported elsewhere. Its result also exposes the action-policy coupling:

| DSP evaluation | Events / 4 | False episodes | Clean alarm time |
|---|---:|---:|---:|
| Native memory actions | 0 | 3 | 0.079% |
| Score-only comparison | 3 | 15 | 2.10% |

This does not mean score-only is better. It means an encoder cannot be selected
from AUROC alone, and a score-fusion result cannot silently replace uEngram's
memory actions. The earlier strong MetroPT-3 CNN result used the plain
average-pooling CNN. The max-average plus orthogonal configuration selected for
SKAB does not transfer.

### 4.4 Score ensembles and late fusion

These are score-only experiments. The CNN ensembles use seeds 0--4 on SKAB and
0--2 on MetroPT-3.

| SKAB score | Events / 34 | AUROC | AP | False episodes | Clean alarm time |
|---|---:|---:|---:|---:|---:|
| Base CNN seed median | 13 | 0.635 | 0.531 | 4 | 2.40% |
| Base CNN Fisher-style sum | 15 | 0.634 | 0.529 | 6 | 2.88% |
| Activation-correlation median | 9 | 0.638 | 0.525 | 4 | 2.41% |
| Late-weight-average median | 9 | 0.649 | 0.553 | 4 | 2.28% |
| DSP + base median, maximum | 13 | 0.616 | 0.505 | 6 | 5.51% |
| DSP + base median, mean | 11 | 0.618 | 0.527 | 5 | 3.76% |

The base median ensemble is the only result worth keeping as a reference. It
removes arbitrary single-seed selection and improves the five-seed worst case,
but it requires five 62,848-byte neural encoders and about five times the
encoder computation. It does not transfer: every CNN-only ensemble detects
0/4 on MetroPT-3.

On MetroPT-3, DSP plus base-CNN maximum fusion detects 3/4 with 15 false
episodes and 2.10% clean alarm time. That is effectively the DSP score-only
trade-off; the weak CNN component adds no defensible benefit. Mean fusion falls
to 1/4. Naive late fusion is rejected.

## 5. What the literature says about the remaining options

### 5.1 Seed variance should be measured, not hidden

Bouthillier et al. treat a benchmark score as a random variable affected by
initialization, data sampling, augmentation, and hyperparameter choice. Their
main lesson applies directly here: fixing one seed makes a run reproducible,
but it does not show that the method is robust.

SWA and deep ensembles address different parts of this problem. SWA seeks a
wider single optimum by averaging a deliberate training trajectory. Deep
ensembles average independently trained predictors and can improve uncertainty
estimates, but multiply training and inference cost. The new uEngram experiment
matches that trade-off: a seed ensemble is more stable, while the cheap
single-model weight average is not reliably better.

**Decision:** every neural candidate needs at least ten seeds on the small SKAB
screen, plus asset or time resampling on the confirmation cohort. Report
median, range, worst case, and the full event set. "Seed variance eliminated"
is reserved for a deterministic transform or a preregistered tolerance that
all varied stochastic sources meet.

### 5.2 Distribution shift needs guarded adaptation

Kim, Park, and Choo's "new normals" method estimates a trend and updates on
samples predicted to be normal. CANDI goes further: it mines likely false
positives and updates a small structure-aware adapter while freezing the main
model. CANDI's authors also report that some selected adaptation samples are
real anomalies in their examples. That is exactly the memory-poisoning risk
uEngram must treat as a first-class failure mode.

A safe uEngram adaptation design would therefore:

1. keep the deployed encoder and semantic memory frozen;
2. update only a small, zero-initialized input/output adapter or a calibration
   layer;
3. accept samples only after a quiet period, latent-neighbor agreement, and,
   where available, operator confirmation;
4. quarantine data during and after any open alarm;
5. retain the frozen score in parallel and roll back when disagreement grows;
6. test a long, low-amplitude fault specifically to prove that adaptation does
   not learn the fault as normal.

This is safer than full test-time fine-tuning. It is still research, not a
license to adapt on an unlabeled production stream.

### 5.3 Label-free model selection is more valuable than one more architecture

Goswami et al. study prediction error, model centrality, synthetic anomaly
performance, and robust rank aggregation for choosing a detector without real
labels. AutoTSAD similarly creates synthetic anomaly families, optimizes a
heterogeneous pool, and ensembles selected scorers. AutoTSAD is primarily a
univariate system, so its implementation cannot be copied directly, but its
selection principle is relevant.

For uEngram, synthetic anomalies should be tied to known physical failure
families:

- persistent level shift;
- slow leak or monotone drift;
- impulsive shock;
- periodic amplitude or frequency change;
- sensor freeze/dropout;
- channel relationship break;
- operating-state transition with no fault.

A candidate should be selected from these synthetic tasks **before** looking at
the real test labels. Real labels then remain an evaluation, not a tuning
signal.

### 5.4 A better CNN objective must preserve fault evidence

AnomalyBERT learns from synthetic degradation, and TS2Vec learns hierarchical
temporal representations from contextual views. CODiT uses temporal
equivariance as a nonconformity signal. These methods support a better direction
than raw reconstruction alone: teach the encoder which temporal changes should
remain visible.

The augmentation must be physically constrained. Generic invariance can make a
model invariant to the fault itself. In particular:

- amplitude and DC level must not be normalized away;
- time shifts may be valid for steady vibration but not for a transition alarm;
- channel permutation is invalid when channels have physical identities;
- frequency warping can erase a bearing or rotor signature;
- synthetic faults must not be injected into the held-out real test stream.

A practical CNN experiment is a multi-task objective with normal-window
reconstruction, masked-segment localization, and a small temporal-equivariance
term. It should retain a raw level bypass and score each branch separately.

### 5.5 Frequency and channel structure are promising, but the full model is too large

CATCH patches the frequency domain and learns channel relationships separately
for each band. This addresses a real weakness of the current CNN: its two
convolutions mix all channels and then collapse time globally. CATCH also uses
instance normalization, which is dangerous here because uEngram's AC-coupled
ablation lost DC valve and leak faults.

The appropriate experiment is therefore not to import CATCH wholesale. Use a
small depthwise/dilated temporal branch, an explicit raw-level branch, and a
low-rank channel mixer. Frequency patches can be tested only in the dynamic
branch. The level branch must bypass instance normalization.

### 5.6 Deterministic multiscale front ends can remove initialization variance

Wavelet scattering is deterministic, nonexpansive, and stable to small time
deformations under its mathematical assumptions. MiniRocket is almost
deterministic and can be made fully deterministic, although its published task
is classification rather than anomaly detection. Neither paper proves that the
method will improve uEngram.

They provide a valuable falsifiable baseline: fixed multiscale filters followed
by the existing fitted projection and state model. This keeps high-frequency
transients without training a random convolutional front end. A small trainable
residual can be added only if the fixed front end is insufficient.

### 5.7 Adaptive calibration comes after representation

Adaptive conformal methods such as AgACI address dependent, shifted time
series. They can help a valid score maintain coverage as conditions change, but
they do not repair a score with no fault separation. The repository's existing
ACI novelty experiments did not recover recall.

Calibration work should therefore follow representation work. The test should
condition on operating state and season, extend adaptation to the health score,
and choose persistence from an explicit average-run-length target. An ongoing
fault must never enter the calibration buffer merely because the threshold has
adapted upward.

## 6. Ranked research program

### Priority 0 — Freeze the next evaluation before model work

Use a fresh asset-separated or later-time cohort. Freeze the split, fault
windows, metrics, alarm budget, seed count, and selection rule first. Keep event
detection, false episodes, clean alarm time, total notifications, lead time,
model bytes, and latency. Do not lower thresholds after seeing outcomes.

**Gate:** no claim of improvement without a fresh test. Replays of SKAB,
MetroPT, SCANIA, CARE, or C-MAPSS remain development evidence.

### Priority 1 — Split the representation by physical role

Use four explicit blocks:

\[
z = [z_{\text{mode}}, z_{\text{level}}, z_{\text{dynamic}}, z_{\text{health}}].
\]

- `mode`: load, idle, and transition context;
- `level`: robust per-channel level, quantiles, and short slope;
- `dynamic`: deterministic multiscale or small dilated CNN features;
- `health`: slow, state-conditioned deviation from an early healthy reference.

Calibrate each block separately. Combine alarms at the evidence level, not by
concatenating everything before one covariance inverse.

**Gate:** improve the preregistered event metric without worsening the
benchmark's false-alarm and clean-time budgets; preserve both DC and transient
fault families in ablations.

### Priority 2 — Add physical residuals and a separate health head

Where load and pressure physics are known, predict an expected relationship and
score the residual. Keep this output separate from novelty. Where run-to-failure
labels exist, train and evaluate a degradation or RUL head with unit-separated
splits and uncertainty. Do not infer RUL from one scalar health distance on
datasets that lack a degradation target.

**Gate:** a residual or health branch must add event detections beyond the raw
threshold signal and maintain calibration on a held-out asset or season.

### Priority 3 — Test guarded, adapter-only shift correction

Freeze the encoder and memory. Adapt only a small adapter using confirmed or
strongly curated normal samples. Run the frozen model in parallel as a safety
control and add rollback.

**Gate:** reduce month-to-month false-alarm drift while preserving detection of
an injected slow fault and a held-out real event. Any learned-fault failure
rejects the method.

### Priority 4 — Select or ensemble models without real test labels

Build a small heterogeneous pool: deterministic DSP, deterministic multiscale,
level/slope, and one compact CNN. Rank candidates on physically motivated
synthetic faults and centrality. Convert scores to empirical calibration ranks
before any ensemble.

**Gate:** an ensemble must beat its best selected component on the frozen
worst-case operating metric, not only mean AUROC. It must justify every extra
encoder pass and byte.

### Priority 5 — Test task-aligned self-supervision

Pretrain the compact CNN to localize masked or degraded regions and preserve
valid temporal equivariance. Keep reconstruction as an auxiliary loss rather
than the only objective.

**Gate:** the learned model must beat the deterministic multiscale baseline
across seed range and fault-family recall. A peak single-seed AUROC is not
sufficient.

### Priority 6 — Add state- and season-aware calibration

After a score passes the representation gate, test state-conditional and
seasonal calibration, then guarded ACI/AgACI-style updates. Select persistence
from measured temporal dependence and the false-alarm budget.

**Gate:** predicted and observed false-alarm rates must agree on a held-out time
block, without absorbing a persistent fault.

## 7. Experiments that should not be prioritized

1. **More width or epochs alone.** They produced good peaks but did not solve
   seed or domain transfer.
2. **Per-window mean removal as the main normalization.** It suppresses false
   alarms by deleting DC fault evidence.
3. **Full-model test-time adaptation.** It risks latent distortion and memory
   poisoning.
4. **Another generic Transformer.** Capacity is not the diagnosed bottleneck,
   and the embedded cost is harder to justify.
5. **Naive score maximum or mean.** The new DSP--CNN experiment is already a
   negative result.
6. **Threshold and persistence sweeps on consumed labels.** They move along the
   same recall--false-alarm trade-off and create post-hoc selection bias.
7. **Reporting only seed 0 or the best seed.** This hides the operational model
   selection risk.
8. **Calling fixed-seed reproducibility stability.** A fixed bad solution is
   reproducible, not robust.

## 8. Novelty boundary for a paper

A targeted search through arXiv and primary AAAI, ICLR, NeurIPS, PMLR, and
PVLDB sources did **not** find a paper with uEngram's exact combination of
bounded state statistics, candidate-state persistence, episodic raw-window
retention, repair/supersession handling, and operator-level alarm accounting.
That is not proof that no such paper exists.

Several close precedents limit any novelty claim:

- MemStream is a bounded streaming anomaly detector with a learned
  representation and drift-aware memory.
- MEMTO is a memory-guided Transformer with learned memory items and latent plus
  input-space anomaly scoring.
- DenStream and CluStream already establish bounded online summaries,
  candidate/outlier handling, and temporal evolution.
- Test-time adaptation, synthetic anomaly selection, neural ensembles,
  conformal calibration, and frequency-patched reconstruction all have prior
  art.

A defensible paper contribution is therefore the **specific state-structured,
fixed-budget system and its rigorous event/operator evaluation**, not a claim
that memory, CNN autoencoders, conformal scores, or adaptation are new. Before
submission, run a formal systematic search with explicit databases, date range,
queries, inclusion criteria, and backward/forward citation screening. It is not
honest to guarantee global nonexistence from a targeted search.

## 9. Primary sources

- Bouthillier et al., [Accounting for Variance in Machine Learning Benchmarks](https://arxiv.org/abs/2103.03098), 2021.
- Izmailov et al., [Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407), 2018.
- Lakshminarayanan et al., [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html), NeurIPS 2017.
- Bardes, Ponce, and LeCun, [VICReg](https://arxiv.org/abs/2105.04906), 2021; Lee and Aune, [VIbCReg for time series](https://arxiv.org/abs/2109.00783), 2021.
- Kim, Park, and Choo, [When Model Meets New Normals](https://arxiv.org/abs/2312.11976), AAAI 2024.
- Kim et al., [CANDI: Curated Test-Time Adaptation](https://ojs.aaai.org/index.php/AAAI/article/view/38524), AAAI 2026.
- Goswami et al., [Unsupervised Model Selection for Time-series Anomaly Detection](https://arxiv.org/abs/2210.01078), 2022.
- Schmidl, Naumann, and Papenbrock, [AutoTSAD](https://www.vldb.org/pvldb/vol17/p2987-schmidl.pdf), PVLDB 2024.
- Jeong et al., [AnomalyBERT](https://arxiv.org/abs/2305.04468), 2023.
- Yue et al., [TS2Vec](https://arxiv.org/abs/2106.10466), AAAI 2022.
- Kaur et al., [CODiT](https://arxiv.org/abs/2207.11769), 2022.
- Wu et al., [CATCH](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2b25c39788e5cf11d3541de433ebf4c0-Abstract-Conference.html), ICLR 2025.
- Mallat, [Group Invariant Scattering](https://arxiv.org/abs/1101.2286), 2011.
- Dempster, Schmidt, and Webb, [MiniRocket](https://arxiv.org/abs/2012.08791), 2020.
- Kim et al., [Reversible Instance Normalization](https://openreview.net/forum?id=cGDAkQo1C0p), ICLR 2022.
- Zaffran et al., [Adaptive Conformal Predictions for Time Series](https://proceedings.mlr.press/v162/zaffran22a.html), ICML 2022.
- Bhatia et al., [MemStream](https://arxiv.org/abs/2106.03837), WWW 2022.
- Song et al., [MEMTO](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b4c898eb1fb556b8d871fbe9ead92256-Abstract-Conference.html), NeurIPS 2023.
- Aggarwal et al., [A Framework for Clustering Evolving Data Streams](https://www.cs.albany.edu/~jhh/courses/readings/aggarwal.vldb03.clustering.pdf), VLDB 2003.
- Cao et al., [Density-Based Clustering over an Evolving Data Stream with Noise](https://epubs.siam.org/doi/10.1137/1.9781611972764.29), SDM 2006.
- Kim et al., [Towards a Rigorous Evaluation of Time-Series Anomaly Detection](https://ojs.aaai.org/index.php/AAAI/article/view/20680), AAAI 2022.
