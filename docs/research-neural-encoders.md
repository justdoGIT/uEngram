# Neural Encoders: Dimensional Capacity, Representation Trade-offs, and Latent Stabilization

## 1. Overview: The Falsification of a Universal "Perfect" Encoder

Across all seven benchmarks (Synthetic, CMAPSS FD001, MetroPT-3, MetroPT2, SKAB,
CARE, and SCANIA Component X), empirical data directly refutes the premise that
any single encoder or latent dimensionality $d$ can serve as a universal optimum.

Two mathematical tradeoffs govern why different physical domains require
fundamentally different representations:

### Tradeoff A: Diffuse Vibrations vs. Sparse Excursions (Curse of Dimensionality)
- Under nominal operation, whitened latent Mahalanobis distance satisfies
  $\mathbb{E}[D^2] = d$, and the conformal novelty threshold scales as $\tau \propto d$.
- **Diffuse Multi-Frequency Faults** (rotor imbalance, cavitation, bearing wear):
  Perturb all frequency bands and latent dimensions simultaneously. Signal scales as
  $\mathcal{O}(d)$. Preserving full dimension ($d=42..48$) maximizes signal-to-noise
  ratio by avoiding destructive PCA projections.
- **Sparse Channel Excursions** (valve throttling, localized pressure loss, single
  thermal runaway): Perturb only 1 or 2 channels. The signal magnitude is $\mathcal{O}(1)$,
  while the background threshold grows with $\mathcal{O}(d)$.
- **Empirical Proof**: On SKAB, raw valve-throttling excursions reach $1.7\times \tau$
  at $d=16$, but only $0.8\times \tau$ at $d=48$, completely submerged by the
  higher-dimensional noise volume.

### Tradeoff B: Physical Intrinsic Rank vs. Covariance Degeneracy
- Continuous physical machinery (compressors, multi-sensor pump loops) has high
  effective rank ($\ge 42$).
- Tabular histogram readouts (SCANIA) and collinear sensor suites (CMAPSS) have low
  physical rank ($\approx 2 \dots 4$ independent components).
- Forcing high dimension ($d \ge 32$) on low-rank data forces `NoiseWhitener` to
  invert near-zero eigenvalues ($\lambda \approx 10^{-12}$), inflating dimension
  variance to 30 billion and driving calibration thresholds up by $565\times$.

---

## 2. Per-Domain Optimum Encoder Matrix

| Domain / Benchmark | Optimal Encoder | Why It Wins | Failure Mode of Alternatives |
|---|---|---|---|
| **High-Bandwidth Vibrations**<br>*(MetroPT-3, MetroPT2, SKAB)* | **`DSPEncoder(d=full)`**<br>($d=42$ for 7ch, $d=48$ for 8ch) | Retains high-frequency harmonics without PCA truncation. Rescues 4 gradual faults on SKAB (9/34 vs 5/34) and halves false alarms on MetroPT-3. | Truncation to $d=16$ discards real high-frequency vibration signals; manual windowed slopes (`DSPSlopeEncoder`) destroy synthetic recall ($1.00 \to 0.23$). |
| **Coarse Tabular Fleet Telemetry**<br>*(SCANIA Component X)* | **`TabularEncoder(d=16, rcond=1e-4)`** | Achieves highest near-failure recall across all experiments (**30.1%**, up from 22.8%; Class 2 recall **42.9%**). Condition number bounded to $10^4$. | Raising to $d=32, 48$ improves raw AUROC ($0.566$ vs $0.540$), but calibration quantiles shift into the tail, collapsing recall to $6.6\%$ under $k=3$ persistence. |
| **Collinear / Low-Rank Systems**<br>*(CMAPSS FD001, Synthetic)* | **`DSPEncoder(d=16)`** | Matches the true physical rank of the system without overfitting within-state noise. | $d \ge 32$ fails numerically on CMAPSS; $d=36$ on synthetic regresses recall ($1.00 \to 0.36$). |

---

## 3. CNN Encoders: Deep Dive into Capacity and Training Regimes

The plain `CNNEncoder` (`src/uengram/encoder_cnn.py`) is a 1D-convolutional autoencoder
with MSE reconstruction loss on channel-standardized inputs:
$$\mathcal{L}_{\text{recon}} = \frac{1}{N C K} \sum_{i=1}^N \| \text{Dec}(\text{Enc}(X_i)) - X_i \|_F^2$$
Followed by linear PCA whitening and within-state noise whitening via `fit_projection`.

### Breakthrough Potential on Continuous Waveforms (MetroPT-3)
Widening to $d=32$ and tripling training duration to 90 epochs surpassed the frozen
`DSPEncoder` on MetroPT-3 (seed 0):

| Model | Latent $d$ | Epochs | AUROC | Average Precision | Events Detected | False Alarms / Week |
|---|---:|---:|---:|---:|---:|---:|
| `DSPEncoder` (frozen baseline) | 16 | — | 0.979 | 0.502 | 3/4 | 14.12 |
| `DSPEncoder` (full bandwidth) | 42 | — | 0.985 | 0.531 | 4/4 | 6.88 |
| `CNNEncoder` (default) | 16 | 30 | 0.786 | 0.095 | 4/4 | 12.60 |
| `CNNEncoder` (wide) | 32 | 30 | 0.988 | 0.512 | 4/4 | 11.50 |
| `CNNEncoder` (long) | 16 | 90 | 0.986 | 0.516 | 3/4 | 4.45 |
| `CNNEncoder` (wide + long) | **32** | **90** | **0.991** | **0.643** | **4/4** | **2.56** |

### Breakthrough Detection of Non-Vibration Faults on SKAB
On SKAB's 34 pointwise-labelled testbed runs, `CNNEncoder(d=32, epochs=90)` learned
representations that captured faults structurally invisible to `DSPEncoder`:
- Specifically detected `valve1/0.csv` (valve throttling), `other/1.csv` (water
  circulation shutoff), and `other/2.csv` (water level decrease).
- Novelty AUROC reached **0.615–0.656** (higher than `DSPEncoder(d=48)` at 0.584).

---

## 4. The Two Critical Failure Modes of CNN Encoders

Despite these peaks, `CNNEncoder` cannot currently be deployed universally:

### Failure Mode 1: The Seed Variance Trap
In multi-seed evaluations on SKAB (`d=32, epochs=90`), identical training data and
architectures converged to wildly divergent operating points under different AdamW seeds:
- **Seed 0**: 9/34 detected, AUROC 0.615, **4 FA episodes, 3.42% clean-time share**.
- **Seed 1**: 19/34 detected, AUROC 0.656, 13 FA episodes, 6.33% clean-time share.
- **Seed 2**: 21/34 detected, AUROC 0.653, 10 FA episodes, **30.00% clean-time share**.

While Seed 0 meets the false-alarm budget, Seed 2 floods the operator with false
alarms for 30% of healthy operation. Because AdamW solves a non-convex optimization
landscape with unconstrained latent geometry, random weight initializations align
arbitrary latent axes with nominal within-state drift.

### Failure Mode 2: Non-Stationary Operating Transients (Synthetic Benchmark)
On the synthetic stream (where operating regimes switch and coolant temperature has
first-order thermal lag), `CNNEncoder(d=32, epochs=90)` achieves norm AUROC = **0.823**
and AP = **0.337** (compared to `DSPEncoder(d=16)`: AUROC **0.999**, AP **0.995**).

**Root Cause**: Raw MSE reconstruction loss forces the autoencoder to allocate
capacity to tracking high-amplitude, low-frequency dynamics (thermal ramps and RPM
sinusoids). Normal regime transitions produce large reconstruction residuals, causing
84% of transition windows to flag and evicting real anomaly memories.

### Failure Mode 3: Contrastive Self-Supervision Collapse (`CNN2`/`CNN3`)
Contrastive learning (`CNN2`) was formally rejected because pulling augmented views
together while repelling random window pairs shatters local Euclidean metric continuity.
The resulting clustered latent manifold breaks the temporal smoothness assumption of
`robust_noise_cov`, causing singular noise covariance matrices where iterative trimming
rejects almost all sample differences.

---

## 5. Neural Latent Stabilization: Measuring Seed Variance

### 5.1 Root Cause Diagnosis

Profiling the latent representations and test-score distributions of `CNNEncoder`
across seeds 0, 1, and 2 revealed the exact mechanism behind the 30% false-alarm blowout:

1. **Inter-Session DC Baseline Shifts**: Real-world machinery data exhibits substantial
   inter-session environmental drift between commissioning and testing. On SKAB:
   - `Accelerometer2RMS`: -57.71 SD drift
   - `Volume Flow RateRMS`: -49.73 SD drift
   - `Temperature`: -27.43 SD drift
   - `Pressure` and `Voltage`: < 0.20 SD drift (stationary)
2. **Unconstrained Bottleneck Alignment**: In the standard CNN autoencoder, the input
   $X$ is standardized globally by training channel statistics, and the pooled features
   pass through an unconstrained linear projection $W \in \mathbb{R}^{d \times 64}$.
   Depending on the random initialization seed, $W$ may align its principal projection
   axes with the drifting channels:
   - **Seed 0**: Random weights happen to project orthogonal to the drift directions;
     clean-test median novelty is 66.1 (95% = 783.6), yielding 4 FA episodes and 3.42% clean time.
   - **Seed 2**: Random weights align with the -50 SD flow/accelerometer drift;
     clean-test median novelty explodes to 147.6 (95% = 6,701.0), triggering 731 clean spikes
     and 30.00% clean alarm time.

### 5.2 Empirical Evaluation of Stabilization Architectures

We initially evaluated three stabilization mechanisms on SKAB ($K=10$, seeds 0, 1, 2):

| Architecture | Mechanism | Detections | Novelty AUROC | FA Episodes | Clean-Time Share | Seed Variance ($\Delta$ clean%) |
|---|---|---:|---:|---:|---:|---:|
| **Unregularized CNN** | Global standardisation, raw MSE | 9 – 21 | 0.615 – 0.656 | 4 – 10 | 3.42% – 30.00% | **26.58%** (extreme) |
| **AC-Coupled CNN** | In-window zero-mean $\tilde{x} = x - \bar{x}$ | 4 | 0.522 – 0.542 | **1 – 1** | **1.20% – 1.28%** | **0.08%** (narrow; three seeds) |
| **OrthoCNN ($\beta=0.05$)** | Bottleneck penalty $\|W W^T - I\|_F^2$ | **10 – 22** | **0.614 – 0.669** | 4 – 14 | 2.74% – 22.97% | 20.23% (partial) |
| **Two-Stream CNN ($d=48$)** | Decoupled AC Conv ($z_{\text{AC}}$) + Robust DC ($z_{\text{DC}}$) | 9 – 12 | 0.570 – 0.577 | 7 – 8 | 12.12% – 14.37% | **2.25%** (narrow; weak operating point) |

### 5.3 Key Architectural Findings

1. **AC coupling narrows false-alarm spread but is not an acceptable fix**:
   Subtracting the in-window mean reduces the clean-time spread to 1.20%–1.28%
   across the three tested seeds, but it also removes the DC level signal used
   by valve and leak faults. Detection falls to 4/34.
2. **Orthogonal Latent Regularization reaches highest peak AUROC (0.669)**:
   Penalizing $\|W W^T - I\|_F^2$ prevents latent dimensions from collapsing onto collinear
   drift directions, lifting Seed 0 AUROC to 0.669 and AP to 0.574 with 2.74% clean time,
   and detecting up to 22 of 34 faults.
3. **Two-stream decoupling stabilizes ranking but is not optimal**:
   Separating dynamic waveform convolution from robust DC tracking gives
   $$z = [z_{\text{AC}}, z_{\text{DC}}] \in \mathbb{R}^d.$$
   Its AUROC range is narrow (0.570–0.577), but it detects only 9–12 of 34
   faults while occupying 12.1%–14.4% of clean time. Stability around a weak,
   high-false-alarm operating point is not a deployment win.

---

## 6. Multi-Scale Pooling, Spectral Loss Ablations, and Production Implementation

### 6.1 Multi-Scale Max-Average Pooling (`pool="maxavg"`)

A major limitation of the standard CNN autoencoder is `AdaptiveAvgPool1d(1)`: averaging
activations over $K$ time samples dilutes isolated impulsive shock waves (knock impulses,
Dirac rotor strikes, cavitation bursts) by $\frac{1}{K}$.

Concatenating global average pooling (steady-state harmonics) with global maximum
pooling (peak transient shock):
$$h_{\text{pool}} = [\text{mean}_t(h), \max_t(h)] \in \mathbb{R}^{128}$$
- **Synthetic Benchmark**: Average Precision jumped from **0.337 $\to$ 0.504** (+49.6% relative gain), with AUROC rising to 0.848.
- **SKAB**: Peak novelty AUROC reached **0.671** (with 18/34 faults detected on Seed 2).

### 6.2 Combined Multi-Scale Orthogonal Model (`pool="maxavg", ortho_beta=0.05`)

Combining max-average pooling with orthogonal bottleneck regularization ($\beta=0.05$)
was tested as a transient-sensitive candidate rather than assumed to be stable:

| Seed | Events Detected | Novelty AUROC | Novelty AP | FA Episodes | Clean-Time Share |
|---:|---:|---:|---:|---:|---:|
| **0** | **14 / 34** | 0.615 | 0.505 | 9 | **8.59%** |
| **1** | **13 / 34** | 0.643 | 0.544 | 4 | **2.45%** |
| **2** | **16 / 34** | 0.628 | 0.532 | 9 | **4.69%** |
| **3** | **10 / 34** | 0.620 | 0.540 | 4 | **2.87%** |
| **4** | **7 / 34** | 0.619 | 0.499 | 5 | **3.17%** |
| **5** | **7 / 34** | 0.610 | 0.506 | 4 | **2.49%** |
| **6** | **9 / 34** | 0.631 | 0.533 | 5 | **3.59%** |
| **7** | **9 / 34** | 0.663 | 0.568 | 4 | **5.03%** |
| **8** | **15 / 34** | 0.608 | 0.534 | 7 | **8.82%** |
| **9** | **10 / 34** | 0.676 | 0.592 | 4 | **2.45%** |

- **Result**: The ten-seed follow-up did not reproduce the unregularized CNN's
  30.0% clean-time blowout, but it disproves the earlier claim that seed
  variance was eliminated. Detection spans 7–16 faults (median 10) and
  clean-time share spans 2.45%–8.82% (median 3.38%).
- **Detection**: The original three seeds caught 13–16 faults; later seeds
  include two 7-event runs. Three seeds were not enough to establish
  robustness.
- **Synthetic Benchmark**: AUROC reached **0.820–0.887** and AP reached
  **0.475–0.680** across the original three seeds.

### 6.3 Negative Results and Falsified Approaches

1. **Time + Spectral Loss ($\mathcal{L}_{\text{MSE}} + \gamma \mathcal{L}_{\text{FFT}}$)**:
   - *Result*: Synthetic AUROC fell from **0.797 $\to$ 0.720** ($\gamma=1.0$).
   - *Cause*: Without logarithmic power compression, the low-frequency RPM sinusoid carries $1000\times$ more energy than high-frequency knock impulses, dominating the gradient updates and blinding the network to small transient bursts.
2. **Log-Spectral Distance Loss ($\mathcal{L}_{\text{log-spec}}$)**:
   - *Result*: Synthetic AUROC collapsed further to **0.589** ($\gamma=1.0$).
   - *Cause*: Reconstructing exact phase and frequency spectra through a 2-layer MLP decoder is an ill-posed phase retrieval problem that produces severe gradient instability and high-frequency ringing artifacts.
3. **LayerNorm / GroupNorm on Latent Bottleneck**:
   - *Result*: Novelty AUROC collapsed to **0.484** (worse than random guessing).
   - *Cause*: In Mahalanobis state math ($D^2 = z^T \Sigma^{-1} z$), the vector magnitude $\|z\|$ carries the anomaly surprisal. Normalizing it across dimensions removes the core metric uEngram relies on for detection.

### 6.4 Production API in `src/uengram/encoder_cnn.py`

The production encoder supports the original reconstruction path plus the
intermediate-readout options:
```python
class CNNEncoder:
    def __init__(self, seed=0, d=16, epochs=30, batch=128,
                 pool="avg", ortho_beta=0.0,
                 representation="bottleneck"):
        ...
```
- **`pool="avg", ortho_beta=0.0` (default)**: Keeps the ordinary bottleneck path and existing behavior.
- **`representation="middle"` or `"multiscale"`**: Reads first-stage or multi-stage convolution features before PCA/noise whitening; these are experimental options, not universal defaults.
- **`pool="maxavg", ortho_beta=0.05`**: Activates multi-scale transient pooling and orthogonal bottleneck regularization. Ten-seed SKAB variance and 0/4 MetroPT-3 detection prevent treating it as a universal or deployment-default configuration.

The stabilization ablations, cross-dataset replay, fusion results, literature
review, and ranked follow-up plan are documented in
[`research-pending-model-improvements.md`](research-pending-model-improvements.md).

## 7. Teacher-to-student transfer

Large-model transfer is now implemented as an offline feature-distillation
option, not direct weight copying. Transformer weights cannot be inserted into
the two-stage CNN because their tensor shapes and learned operations differ.
Instead, a permitted teacher adapter emits an intermediate representation, and
the student learns both reconstruction and teacher-feature similarity.

```python
from uengram import TeacherSpec, WindowSchema

spec = TeacherSpec(
    model_name="vendor-time-series-model",
    model_version="1.0",
    layer="encoder.layer.16",
    feature_dim=256,
    input_schema=WindowSchema(
        ("vibration_x", "vibration_y", "temperature"),
        sample_rate_hz=100.0,
        window_size=256,
    ),
    causal=True,
    alignment="right",
)
student.fit_distilled(
    commissioning_windows,
    teacher_adapter,
    spec,
    noise_lag=1,
    groups=commissioning_segments,
    teacher_weight=0.25,
)
```

The adapter must expose `encode(windows)` and return one finite feature row per
input window. The contract rejects non-causal or centered teachers, wrong
channel order/count, wrong window length, wrong feature dimension, and missing
model/layer/schema versions. Teacher target scaling is fitted on commissioning
windows only. The ordinary `fit()` path and `bottleneck` default are unchanged.

The MetroPT-3 smoke harness uses a local multiscale CNN as a teacher stand-in:

| Student | AUROC | AP |
|---|---:|---:|
| Ordinary CNN | 0.9879 | 0.4781 |
| Distilled CNN | **0.9909** | **0.5633** |

This is one consumed development split, five epochs, and a local CNN teacher;
it is an API/integration demonstration, not evidence that a large LLM or
foundation model improves field accuracy. Full guidance, pseudocode, licensing
warnings, and the acceptance gate are in
[`research-teacher-distillation.md`](research-teacher-distillation.md).

The first public foundation-model run uses the MIT-licensed MOMENT-1-small
checkpoint at revision
`411e288267f82cce86296dbe4d6c8bc533cc162f`. With native 512-sample MetroPT-3
windows, the full bounded-memory pipeline changed AUROC/AP from
`0.9756/0.6225` for the ordinary CNN to `0.9775/0.7942` after distillation,
while retaining all four reported events. The same teacher reduced the
four-family synthetic probe from `0.6872/0.8007` to `0.5035/0.6602`. The
negative cross-domain result is the stronger engineering lesson: teacher
transfer must be gated by window compatibility and held-out fault families.

### 7.1 Context repair and instance-statistics ablation

The adapter now makes the 512-sample context mismatch explicit. `edge` repeats
the first sample on the left; `linear` resamples the known short window across
the full teacher context. A second option appends each channel's original mean
and standard deviation, because instance normalization in a foundation model
can hide absolute level and volatility.

| Variant on the fixed synthetic probe | AUROC | AP |
|---|---:|---:|
| Ordinary CNN baseline | 0.6872 | 0.8007 |
| MOMENT + edge repeat | 0.5035 | 0.6602 |
| MOMENT + linear resampling | 0.5250 | 0.6956 |
| MOMENT + linear + mean/std | **0.5362** | **0.7136** |

The repair helps relative to edge padding but does not beat the ordinary
domain-specific CNN. On MetroPT-3's native 512 context, mean/std augmentation
gave AUROC/AP `0.9751/0.5846`; the prior no-statistics transfer gave
`0.9775/0.7942`. It is therefore an optional research branch, not a new
default.

This matches two external findings. Kim et al. (ICLR 2026) report
overgeneralization and overstationarization in time-series foundation-model
anomaly detection and recover useful statistics without retraining the
foundation model. Khan et al.'s ChronosAD (2026) adds temporal refinement
after foundation-model features. Both support testing a small, causal
refinement stage rather than assuming a frozen vector is complete. The TSB-AD
benchmark is a suitable source for broader preregistered comparisons, while
its warning that simple statistical methods can win some series argues
against replacing uEngram's statistical safeguards with a neural teacher.

### 7.2 New TSB-AD-U public holdout

The first new public holdout used six exact TSB-AD-U series from six
application/domain labels. Healthy-prefix normalization, a 50% healthy
training split, 64-sample windows, stride 32, `d=16`, five epochs, and
teacher weight 0.25 were fixed before evaluation.

| Model | Pooled AUROC | Pooled AP | Macro-series AUROC | Macro-series AP |
|---|---:|---:|---:|---:|
| Ordinary CNN | 0.9111 | 0.8397 | **0.6476** | **0.4290** |
| MOMENT + edge repeat | 0.9091 | **0.8397** | 0.6336 | **0.4397** |
| MOMENT + linear + mean/std | **0.9157** | 0.8351 | 0.6316 | 0.4249 |

The pooled and macro rankings disagree. Exathlon contains many more
evaluation windows and dominates the pooled score; the macro score gives each
series equal weight. This is evidence that one aggregate number is unsafe for
heterogeneous anomaly benchmarks. Two selected series had only 2 and 4
positive windows, so their individual estimates are high variance.

The result is a genuine new public holdout for this repository, not a new
industrial validation cohort. The exact archive hash, members, splits, and
teacher revision are recorded in
`results/moment_tsb_ad_holdout.json`. The correct conclusion is that
MOMENT transfer remained competitive in aggregate but did not produce a
consistent cross-domain gain over the local CNN.

### 7.3 Multi-teacher transfer: the CNN idea, adapted safely

The image-CNN technique remembered from transfer learning has two meanings.
One is literal feature stacking: keep several large networks in the inference
graph and concatenate their hidden outputs. The other is multi-teacher
distillation: run frozen teachers offline, concatenate their declared
representations, and train one small student to reproduce them. The second
meaning is now implemented in `CNNEncoder.fit_distilled_multi` and
`extract_multi_teacher_features`.

The edge device receives only the student CNN and its compact latent. Large
teacher weights are never loaded into the edge process. Each teacher must use
the same causal window schema. Its feature block is standardized on the
commissioning set, and the training loss averages the mean-squared error of
the blocks so a wide teacher cannot automatically dominate a narrow one:

$$
\mathcal{L}=\mathcal{L}_{\mathrm{reconstruction}}+
\lambda\frac{1}{M}\sum_{m=1}^{M}
\frac{1}{d_m}\lVert h_m(X)-\hat h_m(X)\rVert_2^2.
$$

Here $M$ is the number of teachers, $d_m$ is teacher $m$'s feature width,
$h_m$ is its frozen output, and $\hat h_m$ is the corresponding student
block. In simple words, several specialists teach during training, but the
small student does the work after deployment.

This is not automatically better. Correlated teachers can repeat the same
mistake, mismatched preprocessing can create a false training target, and
feature concatenation can increase fitting cost. The contract tests now cover
block order, schema mismatch rejection, equal block weighting, and the
combined student head. A real multi-model accuracy claim still requires two
independent, licensed teachers and an untouched asset-disjoint test set; the
current public holdout used one MOMENT teacher.

### 7.4 Stronger frozen gate

`bench/stronger_gate.py` writes `results/stronger_gate.json`. It combines the
new TSB-AD-U teacher-transfer holdout with the five-seed CMAPSS lifecycle/RUL
results and a fresh replay of the frozen SKAB event protocol. This separation
matters: CMAPSS and SKAB are not silently presented as MOMENT-transfer
results.

The fresh SKAB replay detected 5 of 34 events, with AUROC 0.5458 and AP
0.4192. All five detections were post-onset under the frozen policy, and the
observed clean exposure remains too short for a stable weekly false-alarm
estimate. The CMAPSS report retains five-seed health AUROC, RUL RMSE, and
end-of-life retention together. The gate therefore passes protocol coverage,
not universal performance: MOMENT transfer remains an explicit CMAPSS/SKAB
follow-up rather than an implied success.

### 7.5 Two independent teachers on an untouched MFPT gate

The next gate used two different, openly licensed time-series teachers:

- **MOMENT-1-small**, fixed to revision
  `411e288267f82cce86296dbe4d6c8bc533cc162f`, with an MIT license.
- **Chronos-2**, fixed to revision
  `29ec3766d36d6f73f0696f85560a422f50e8498c`, with an Apache-2.0 license.

They were used only during training and measurement. The edge model still
contains the compact CNN; it does not carry either teacher's large weights.
MOMENT produced 514 features for each window and Chronos produced 32
quantile-and-median forecast features. A feature is simply a number describing
what the teacher noticed. The student learned both blocks with equal
per-teacher loss, so the wider MOMENT block could not win merely by having
more numbers.

The new dataset was the MFPT Fault Data Sets archive. The archive was
downloaded from the archived MFPT source, checked with SHA-256
`cd46b9d95545c095899b2b9f6462ce60cdbfeae87a743625dd312e6a4df6e6fa`, and
never added to git. Baseline recordings 1 and 2 trained the models. Baseline
recording 3, all test-rig fault recordings, and all three field recordings
were held out. Windows were resampled to 2,048 Hz, used 512 samples, and
overlapped by 75%. Normalization and the 99th-percentile decision threshold
used training-normal data only.

| Model | Window AUROC | File AUROC | Positive-file recall | Normal-file false alarms | Edge model bytes |
|---|---:|---:|---:|---:|---:|
| Ordinary CNN | 0.7948 | 0.8500 | 1.0000 | 1.0000 | 46,144 |
| MOMENT student | 0.7926 | 0.8000 | 1.0000 | 1.0000 | 46,144 |
| Chronos student | 0.8143 | 0.8000 | 1.0000 | 1.0000 | 46,144 |
| **Two-teacher student** | **0.8253** | **0.9000** | 1.0000 | **1.0000** | 46,144 |

The two teachers improved ranking: window AUROC rose by 0.0305 and file
AUROC rose by 0.0500 against the ordinary CNN. That is useful evidence for
distillation, but it is **not a deployment pass**. Every model incorrectly
flagged the only held-out normal recording. In everyday language, the models
recognized faults but did not recognize a new healthy operating condition.
The false-alarm result has more operational meaning than the AUROC gain.
Because MFPT contains static fault experiments rather than labelled
run-to-failure streams, warning time is not measured.

The first short run also failed before producing a result: the covariance
estimator saw too few difference pairs. The safe repair was to use more
overlapping windows, not to suppress the covariance estimate or silently
lower the latent dimension. This is a reusable error path: a model can fail
because the statistical calibration sample is too small even when the neural
network itself is valid.

The teachers cost roughly 141 MB (MOMENT) and 478 MB (Chronos) of parameter
storage in this run and took about 21.0 ms and 58.9 ms per probe window,
respectively. Those costs are acceptable for offline training on a gateway,
not for an always-on microcontroller. The measured process peak was about
1.21 GB. The compact student remained 46,144 bytes, and its measured test
encoding time was about 0.066 ms per window in this benchmark environment.

This is a stronger gate than a random window split, but it is still not a
fully verified physical-asset split: the MFPT metadata does not document the
identity of every test-rig bearing. The result therefore supports the
engineering direction, not a universal industrial claim. The next gate
should use a dataset with explicit bearing IDs, multiple healthy operating
loads, onset timestamps, and repeated seeds. The source license is also
marked unspecified for commercial use, so commercial deployment requires a
separate permission review.

The implementation and exact report are `bench/mfpt_two_teacher_gate.py`
and `results/mfpt_two_teacher_gate.json`. The independent teacher adapters
are optional; installing their packages is not required for the base
uEngram package.

Useful primary sources:

1. Chronos-2 model card: <https://huggingface.co/amazon/chronos-2>.
2. Chronos-2 technical report: <https://arxiv.org/abs/2510.15821>.
3. MFPT fault-data source: <https://www.mfpt.org/fault-data-sets/>.
4. The pinned archive used by this gate:
   <https://web.archive.org/web/20250505045128id_/https://www.mfpt.org/wp-content/uploads/2020/02/MFPT-Fault-Data-Sets-20200227T131140Z-001.zip>.

### 7.6 CWRU asset-group holdout: perfect ranking, unusable calibration

To make the holdout stricter, a second public subset came from the Case
Western Reserve University (CWRU) bearing data. The source manifest records
fault groups by damaged-bearing identifier. Healthy recordings 97--99 trained
the model; healthy recording 100 was held out as a new operating load; three
fault asset groups (`IR007`, `B007`, and `OR007_6`) were held out entirely.
The downloaded files were hashed individually and kept outside git. The
manifest does not specify a commercial license, so this is a research-only
result until permission is verified.

| Model | Window AUROC | File AUROC | Positive-file recall | Normal-file false alarms | Edge model bytes |
|---|---:|---:|---:|---:|---:|
| Ordinary CNN | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 46,144 |
| MOMENT student | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 46,144 |
| Chronos student | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 46,144 |
| Two-teacher student | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 46,144 |

This apparently perfect result is a warning, not a breakthrough. The only
held-out normal recording was flagged by every model, so the threshold did
not recognize healthy variation at a new load. The fault recordings are
static and strongly separated from the healthy training recordings; AUROC
therefore answers an easier question than an operating alarm system needs.
The CWRU gate confirms the MFPT finding: high ranking scores can coexist with
zero useful calibration.

The run used 624 test windows, a 99th-percentile training-normal threshold,
and a 512-sample window at 12 kHz. The two-teacher student still measured
46,144 bytes and about 0.063 ms per test window. Offline teacher costs were
about 141 MB for MOMENT and 478 MB for Chronos, with a process peak near
1.22 GB. The exact hashes, split, and resource measurements are recorded in
`results/cwru_two_teacher_gate.json`; the raw files are intentionally not
committed.

The CWRU experiment is useful because it catches a tempting but dangerous
interpretation: “perfect AUROC” does not mean “safe anomaly detector.” The
next experiment must include healthy recordings from many loads and assets,
calibrate on a development fleet rather than only three healthy files, and
report false alarms per operating hour. Until then, the two-teacher gain is
supported on MFPT ranking but not on CWRU operational calibration.

Primary sources:

1. CWRU public file source: <https://engineering.case.edu/sites/default/files/97.mat>
   (the same path pattern was used for the pinned subset).
2. CWRU metadata manifest:
   <https://github.com/VictorBauler/BearingDatasets/blob/main/src/bearing_datasets/datasets/cwru/files.csv>.

### 7.7 Complete run-to-failure CMAPSS gate

The next step used the existing NASA CMAPSS FD001 data, which has complete
run-to-failure trajectories. Units 1--40 supplied 30 early commissioning
windows each. Units 41--50 were never shown during training and were scored
through their complete failures. The alarm required three consecutive flagged
windows. The “end-of-life horizon” was defined as RUL (remaining useful life)
of 20 cycles or less.

This gate uses one retained sensor (`sensor_indices=[0]`) because the
14-channel teacher run exceeded the bounded 15-minute execution limit on this
host. That reduction is an explicit limitation, not a hidden optimization.
It provides a reproducible latency/cost gate for the teacher pipeline, not the
best possible CMAPSS accuracy.

| Model | Window AUROC | Window AP | Recall before failure | Recall by RUL 20 | Pre-EOL alarms | Pre-EOL alarms / 100 healthy cycles |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary CNN | 0.9823 | 0.9052 | 1.0000 | **0.7000** | 7 | **0.406** |
| MOMENT student | **0.9902** | 0.9453 | 1.0000 | 0.2000 | 19 | 1.103 |
| Chronos student | 0.9896 | **0.9491** | 1.0000 | 0.3000 | 13 | 0.754 |
| Two-teacher student | 0.9890 | 0.9397 | 1.0000 | 0.2000 | 15 | 0.871 |

The result reverses the ranking-only story. The foundation teachers slightly
improved AUROC and AP, but the ordinary CNN gave the best timely event recall
by the RUL-20 horizon and produced fewer early alarm episodes. The
two-teacher student detected every eventual failure before the end of the
trajectory, but only 2 of 10 units had a sustained alarm by RUL 20. In other
words, it often detected degradation too late for the chosen maintenance
horizon.

“Pre-EOL alarms” are intentionally not called false alarms. Every held-out
CMAPSS unit eventually fails, so an alarm before RUL 20 may be a valid early
warning rather than a false alarm. The true false-alarm measurement available
here is the healthy commissioning portion: all four models produced zero
sustained calibration alarms over 1,200 healthy windows. This is not enough
to estimate field false alarms per hour. CMAPSS has cycles, not wall-clock
sampling intervals, so the report uses alarms per 100 healthy cycles.

The compact student remained 46,144 bytes. Two-teacher student encoding was
about 0.0031 ms per window, while the process peak was about 1.35 GB.
Teacher parameter storage remained approximately 141 MB for MOMENT and 478 MB
for Chronos. The exact split, threshold, sensor choice, revisions, resource
measurements, and all metrics are in
`results/cmapss_two_teacher_gate.json`.

The most important methodological correction came during this gate. An early
metric implementation treated a sustained alarm before RUL 20 as a false
alarm and also accepted a shortened two-window suffix as a three-window
persistent alarm. Both errors could make the result look worse or better for
the wrong reason. The final implementation requires the full persistence
window and separates:

1. detection before eventual failure;
2. detection by the chosen maintenance horizon;
3. early-warning episodes before that horizon; and
4. false alarms measured only on healthy calibration windows.

This CMAPSS study therefore gives a stronger lifecycle diagnosis but does not
establish industrial superiority. The next meaningful gate is a multi-asset,
wall-clock run-to-failure dataset with healthy machines that do not all fail,
so false alarms per operating hour and maintenance lead time can be measured
without ambiguity.

Primary source: NASA CMAPSS FD001 dataset and protocol notes in
[`docs/research-cmapss.md`](research-cmapss.md).
