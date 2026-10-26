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
