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

## 5. Neural Latent Stabilization: Eliminating Seed Variance

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

We designed and evaluated three stabilization mechanisms on SKAB ($K=10$, seeds 0, 1, 2):

| Architecture | Mechanism | Detections | Novelty AUROC | FA Episodes | Clean-Time Share | Seed Variance ($\Delta$ clean%) |
|---|---|---:|---:|---:|---:|---:|
| **Unregularized CNN** | Global standardisation, raw MSE | 9 – 21 | 0.615 – 0.656 | 4 – 10 | 3.42% – 30.00% | **26.58%** (extreme) |
| **AC-Coupled CNN** | In-window zero-mean $\tilde{x} = x - \bar{x}$ | 4 | 0.522 – 0.542 | **1 – 1** | **1.20% – 1.28%** | **0.08%** (zero variance) |
| **OrthoCNN ($\beta=0.05$)** | Bottleneck penalty $\|W W^T - I\|_F^2$ | **10 – 22** | **0.614 – 0.669** | 4 – 14 | 2.74% – 22.97% | 20.23% (partial) |
| **Two-Stream CNN ($d=48$)** | Decoupled AC Conv ($z_{\text{AC}}$) + Robust DC ($z_{\text{DC}}$) | 9 – 12 | 0.570 – 0.577 | 7 – 8 | 12.12% – 14.37% | **2.25%** (stabilized) |

### 5.3 Key Architectural Findings

1. **AC-Coupled CNN eliminates false-alarm seed variance completely**:
   Subtracting in-window mean removes all DC inter-session level drift, yielding an identical
   clean-time share of 1.20% to 1.28% across all seeds. However, it blinds the detector
   to pure DC level faults (valve throttling and fluid leaks), reducing detections to 4/34.
2. **Orthogonal Latent Regularization reaches highest peak AUROC (0.669)**:
   Penalizing $\|W W^T - I\|_F^2$ prevents latent dimensions from collapsing onto collinear
   drift directions, lifting Seed 0 AUROC to 0.669 and AP to 0.574 with 2.74% clean time,
   and detecting up to 22 of 34 faults.
3. **Two-Stream Decoupling is the optimal neural architecture**:
   By separating dynamic waveform convolution from linear robust DC level tracking:
   $$z = [z_{\text{AC}}, z_{\text{DC}}] \in \mathbb{R}^d$$
   the network achieves near-zero seed variance on ranking (AUROC $0.570 \dots 0.577$)
   and false alarms (clean share $12.1\% \dots 14.4\%$), while retaining the ability
   to detect both dynamic vibration faults and DC flow/valve faults.
