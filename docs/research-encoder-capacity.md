# Encoder capacity, CNN improvements, and combined encoders

Three follow-up questions after `docs/research-skab-validation.md`'s slope-
encoder rejection: can `DSPEncoder`'s latent dimension `d` be raised instead
of adding features; what improves the rejected CNN encoders; and does
combining `ContextEncoder` with `DSPSlopeEncoder` do better than either
alone. All three were run as measured, falsifiable experiments, not
speculation. **None of this changes the frozen `uengram/dsp/v2_frozen`
candidate**; it is development evidence for where a future encoder change
would need to start.

## 1. Raising `d`: a real effect, dataset-dependent, with a real statistical
   ceiling

`DSPEncoder(d=16)` keeps only the top-16 of 36-84 raw band features
(depending on channel count), by dataset: synthetic 36, MetroPT-3 42, CMAPSS
84, SKAB 48. Raising `d` reduces how much of that raw feature space PCA
discards. Two real infrastructure bugs were found and fixed while testing
this (both were hardcoded `D=16` module constants decoupled from the
encoder's actual dimension, silently correct only because every prior
confirmed result used `d=16`):

- `bench/v2_scoring.py`: `B, D = 256, 16` → `B = 256`; `d = Z.shape[1]`
  computed per call.
- `bench/run_metropt.py`: `B, D, K = 256, 16, 60` → `B, K = 256, 60`;
  `d = len(enc.sigma0)` computed per call.

`bench.run_synth.make_encoder` now accepts `"dsp_dN"` (e.g. `"dsp_d48"`) for
any even `N`, routed through the same shared pipeline used for every other
encoder, so this is testable with the existing harnesses, not a one-off
script.

### Results (full seed counts, same alarm policy, no other change)

| Benchmark | `d=16` (frozen) | Full/near-full `d` | Verdict |
|---|---:|---:|---|
| Synthetic (`d=36`, 10 seeds) | event_recall 1.00, seizure_recall 1.00, AP 0.996 | event_recall **0.358**, seizure_recall 0.967, AP 0.565 | **Regression** (same PCA-budget mechanism as the slope encoder, milder: seizure_recall survives, event_recall does not) |
| CMAPSS (`d=24`, 5 seeds) | AUROC 0.684, RUL-RMSE 42.05 | AUROC **0.520** (worse), RUL-RMSE **21.05** (halved) | Mixed; `d>24` fails outright (next section) |
| MetroPT-3 (`d=42`, 5 seeds, `uengram`) | AUROC 0.979, AP 0.502, FA/week 14.12 | AUROC **0.988**, AP **0.601**, FA/week **5.70** | **Consistent improvement**, every method (naive/fifo/reservoir too), detection count unchanged (3/4) |
| SKAB (`d=48`, seed 0) | 5/34 detected, 5 FA episodes | **9/34** detected, **5 FA episodes** (same) | **Clear win, free**: no slope/context feature engineering needed |

Raw data: `results/synth_d36_runs.jsonl`, `results/cmapss_d24_runs.jsonl`,
`results/metropt_d42_runs.jsonl`; SKAB tested via throwaway script (not
committed as a separate results file, consistent with this project's
diagnostic-script convention).

**SKAB correction to the earlier slope-encoder analysis**: plain
`DSPEncoder(d=48)` alone rescues the *same* four experiments
(`other/10,11,13,14`) that `DSPSlopeEncoder` and `CtxDSPEncoder` rescue,
at the *same* false-alarm-episode count as the `d=16` baseline (5, not
10+). The earlier diagnosis ("not a PCA-discarding problem") was correct in
the narrow sense that detected-vs-undetected retained fraction didn't
differ — but the *absolute* signal lost at `d=16` was still enough to push
several borderline faults below the alarm threshold. Feature engineering
(slope, context) was solving a problem that was actually just information
loss from aggressive compression. For SKAB specifically, plain `d=48` is
strictly better than every encoder variant tested so far: more events,
same false-alarm cost, no new feature code.

### Why `d` cannot simply be raised everywhere: a real numerical ceiling

CMAPSS fails outright above `d=24`:

```
d=16 OK   inflation range=[3.17,26.38]
d=24 OK   inflation range=[0.39,16.17]
d=32 FAILED: fewer than 10 difference pairs for noise estimate
```

This is not a sample-size shortage (9,159 commissioning windows). CMAPSS's
48 raw features come from a 4-way FFT-bin split of only `K//2=8` bins (`K=16`
cycles), so several band-energy columns are near-duplicates at higher `d`.
`_segments()` (`uengram/stats.py`) computes a Mahalanobis distance against
`robust_noise_cov`, a `d×d` matrix; near-degenerate columns make that matrix
ill-conditioned, its inverse unstable, and the chi-square change-point test
fires on nearly every window — fragmenting the 50-unit grouping into tiny
segments that then fail the outer lag-38 pair-count check. SKAB and MetroPT
tolerate full dimension because each is one long, genuinely continuous
commissioning stream (no per-unit fragmentation to begin with); CMAPSS's
per-unit structure interacts badly with near-duplicate features at high `d`.
**The practical ceiling is dataset-specific and must be found per dataset,
not assumed from channel count.**

### Recommendation

Not adopted as a global default (synthetic regression blocks that), but
worth a per-dataset decision, same way `K` and the commissioning split
already are:
- MetroPT-3: `d=42` (full) is a strict improvement on every metric tested
  and every method, at no detection cost. Worth promoting past this
  development stage with proper confirmation-style validation before
  changing the frozen candidate.
- SKAB: `d=48` (full) is strictly better than the earlier slope ablation.
- CMAPSS: `d=24` trades novelty AUROC for RUL-RMSE; not a clean win,
  needs a decision about which objective matters more.
- Synthetic: stay at `d=16`; this is the benchmark that would catch a
  regression, and it did.

## 2. Improving the CNN encoders: undertrained and under-capacity, not
   fundamentally wrong

The plain `CNNEncoder` (CPU-trainable 1D-conv autoencoder, `d=16`,
`epochs=30` by default) was never the formally rejected encoder — `CNN2`/
`CNN3` (contrastive, GPU-trained, ensemble) were, on MetroPT-3 development.
Testing `CNNEncoder` itself on MetroPT-3 (seed 0, same engine/alarm
pipeline) by varying `d` and `epochs`:

| `d` | `epochs` | AUROC | AP | detected | FA/week |
|---:|---:|---:|---:|---:|---:|
| 16 (default) | 30 (default) | 0.786 | 0.095 | 4/4 | 12.60 |
| 32 | 30 | 0.988 | 0.512 | 4/4 | 11.50 |
| 16 | 90 | 0.986 | 0.516 | 3/4 | 4.45 |
| **32** | **90** | **0.991** | **0.643** | **4/4** | **2.56** |

The default configuration (`d=16, epochs=30`) was simultaneously
undertrained and bottlenecked too narrowly. Tripling training epochs and
doubling the latent width together give AUROC/AP that are at or above the
frozen DSP encoder's own committed MetroPT numbers (DSP: AUROC up to 0.979,
AP up to 0.575, FA/week 4.9-14.1 across methods), on this single seed.

**This does not vindicate CNN2/CNN3**, the encoders actually rejected.
Testing the same widening on `CNN2Encoder` (`d=32`, same 24 epochs) hit a
*different* numerical wall:

```
ValueError: noise covariance retained 3 inliers; need at least 33 at iteration 4
```

CNN2's contrastive-self-supervised latent at `d=32` produces lag-1
differences whose iterative robust trimming (`robust_noise_cov`, 97.5%
keep-rate per iteration, 5 iterations) rejects almost everything — a
different fragility than CMAPSS's near-duplicate-feature problem, but the
same family of issue: **widening a latent changes its noise structure in
ways the whitening/step-detection machinery assumes are well-behaved, and
that assumption is not free to violate.**

### Recommendation

- The plain `CNNEncoder` is worth a proper multi-seed revalidation at
  `d=32, epochs=90` before any adoption decision — this single-seed result
  is promising, not confirmed.
- CNN2/CNN3 need an architecture- or training-side fix for why their
  specific latent distribution degenerates the noise-covariance estimate
  at higher `d`, before width alone can be tested the same way.
- Neither is wired into `bench.run_v2.FROZEN` or any confirmed result; both
  are development-stage findings.

## 3. `CtxDSPEncoder`: a real combination, plus a real bug fixed first

`ContextEncoder` (the first-order-lag-residual encoder) had never been
applied to any multi-file/multi-unit real dataset — only to the synthetic
benchmark, which is one continuous stream. Its `_residuals`/`fit` had no
segment-boundary handling at all: applying it directly to SKAB's 34
concatenated experiment files would have let the lag-residual's internal
state leak across unrelated files, producing a spurious residual spike at
every file boundary that has nothing to do with the labelled fault.

Fixed in `src/uengram/encoder_ctx.py`: `fit`/`encode`/`_residuals` now take
the same `groups`/`segment` convention `DSPSlopeEncoder` uses. A window
immediately after a segment break falls back to the raw ramp (the
`sequential=False` convention) instead of chaining the previous segment's
state; `fit()` drops segment-first rows from the lag-model regression.
Covered by `tests/test_encoders.py::test_context_encoder_resets_at_a_
segment_boundary`, `::test_context_encoder_same_segment_mask`, and
`::test_context_encoder_fit_excludes_segment_boundary_rows_from_the_
regression`.

`CtxDSPEncoder` (same file) combines `ContextEncoder`'s lag-residual
channels with `DSPSlopeEncoder`'s windowed-mean slope for every channel, by
inheritance (`_raw()` override; `fit`/`encode` unchanged, dynamic dispatch
picks up the combined features automatically).

### SKAB results (seed 0, `Current` driving `Temperature`/`Thermocouple`,
slope lags 3 and 12 windows)

| Encoder | `d` | Detected | FA episodes | Rescued vs. baseline |
|---|---:|---:|---:|---|
| Frozen `dsp` (baseline) | 16 | 5/34 | 5 | — |
| `ContextEncoder` alone | 16 | 5/34 | 4 | none (AUROC 0.550→0.597, no rescue) |
| `DSPSlopeEncoder` alone | 16 | 8/34 | 10 | `other/10,13,14` |
| `CtxDSPEncoder` (combined) | 24 | **9/34** | 6 | `other/10,11,13,14` |
| Plain `DSPEncoder(d=48)` | 48 | **9/34** | **5** | `other/10,11,13,14` |

`CtxDSPEncoder` does rescue one more experiment (`other/11`) than
`DSPSlopeEncoder` alone, and at a lower false-alarm cost (6 vs 10 episodes)
— combining the two mechanisms is better than either alone. **But it does
not beat plain `DSPEncoder(d=48)`**, which rescues the identical four
experiments at the baseline's exact false-alarm count, with no new feature
code at all. `ContextEncoder` alone learns a physically plausible lag
(`a=0.784`, a ~13s electrical-heating time constant for `Current`→
`Temperature`) but does not rescue anything by itself on SKAB; combining it
with slope features is what adds the fourth rescue (`other/11`), at the
cost of the extra false alarms slope features already carried.

### Recommendation

For SKAB specifically, prefer plain `DSPEncoder(d=48)` over every
feature-engineered alternative tested: it is simpler, detects the same
events, and costs nothing in false alarms. `CtxDSPEncoder` remains useful
evidence that the two mechanisms compose (not just stack their costs
additively) and that `ContextEncoder` generalizes correctly once
segment-aware, but it is not the best SKAB-specific answer found.

## 4. Project-wide per-dataset matrix (complete)

Section 1 covered the four windowed DSP benchmarks (synthetic, CMAPSS,
MetroPT-3, SKAB). This section completes the matrix with the two remaining
DSP streams (MetroPT2) and the two tabular-encoder streams (CARE, SCANIA),
all run through each dataset's existing protocol with the frozen alarm
policy unchanged. `d` here means `DSPEncoder(d=…)` latent width on the
windowed streams and `TabularEncoder(d=…)` on the two row-summary streams;
"full" is the raw band-feature count (6 per channel × 7 analog channels =
42 for both MetroPT streams, 48 for SKAB's 8 channels).

| Dataset | Encoder | Swept `d` | Frozen `d=16` | Best `d` | Verdict |
|---|---|---:|---:|---|---|
| Synthetic | DSP | 16, 36 | event/seizure recall 1.00/1.00, AP 0.996 | **16** | `d=36` regresses event_recall to 0.36 |
| CMAPSS | DSP | 16, 24, 32✗ | AUROC 0.684, RUL-RMSE 42.05 | 16 or 24 | `d=24` halves RUL-RMSE but drops novelty AUROC; `d≥32` fails numerically |
| MetroPT-3 | DSP | 16, 42 | AUROC 0.979, AP 0.502, FA 14.12/wk | **42** | strict win on every metric, every method |
| SKAB | DSP | 16, 24, 32, 48 | 5/34 detected, 5 FA episodes | **48** | 9/34 detected, same 5 FA episodes |
| MetroPT2 | DSP | 16, 24, 32, 42 | jan 0/3, apr AUROC 0.812 | **42** | `d=42` best jan detection (2/3); `d=32` best apr AUROC (0.958); both beat `d=16` |
| CARE | Tab | 16, 32 | 12 events, 0 care anomalies | **32** | 15 events, 2 care anomalies; FA 0.14→0.29/wk |
| SCANIA | Tab | 16, 32, 48 | recall 0.228, AUROC 0.541 | **16** | higher `d` lifts AUROC to 0.571 but collapses recall to 0.022 via policy re-calibration |

### MetroPT2 (`dsp`, jan + apr streams, seed 0, v2 policy)

`d=16` is the worst of the four widths on both streams. `d=42` (full) is
the only width that detects the jan stream's fault events under the v1
rule (2/3 vs 0/3 at `d=16`), and `d=32` gives the best apr novelty ranking
(AUROC 0.958, AP 0.415 vs 0.812/0.167 at `d=16`). No width detects under
the frozen MetroPT-3-selected v2 policy (it is calibrated for MetroPT-3,
not MetroPT2), so the v1 rule and the policy-independent AUROC/AP are the
informative columns here. Raw data: `results/metropt2_d16_d24_d32_d42.jsonl`.

### CARE (`tab`, 95 streams, seed 0)

CARE's events are extremely hard: most streams detect nothing. `tab_d32`
recovers 15 events to `tab_d16`'s 12 and turns 2 care anomalies
criticality-detected vs 0, at the cost of a near-doubled false-alarm rate
(0.14 → 0.29/week) and marginally more alerts (37.6 → 39.0/week). This is a
small recall-for-precision trade, not a strict win; it is the only dataset
where a latent-width change produces any CARE-criticality detection at all.
Raw data: `results/care_tabular_d16_d32.jsonl`.

### SCANIA (`tab`, 5,046 vehicles, frozen point-label protocol)

Raising `d` is a trap here, not a ceiling. Novelty ranking improves
(AUROC 0.541 → 0.571 at `d=32`, 0.561 at `d=48`), but the frozen alarm
policy re-calibrates its conformal threshold on the higher-dimensional
scores and becomes drastically more conservative: near-failure recall
collapses from 0.228 to 0.022 (`d=32`) / 0.029 (`d=48`), and the healthy
alarm rate falls from 0.193 to 0.027. The encoder is *better* at ranking,
but the unchanged detector turns that into *fewer* alarms on the very
vehicles it should flag. SCANIA's coarse time-to-repair classes cannot
support a `d` change without re-deriving the alarm policy, which is outside
this study's "frozen policy" boundary. Stay at `tab_d16`.
Raw data: `results/scania_tab_d_sweep.json`.

### Bottom line

There is no single best latent width. Full dimension is a free, strict
improvement on the two long continuous MetroPT streams (MetroPT-3 `d=42`,
MetroPT2 `d=42`) and on SKAB (`d=48`). It is a regression on the synthetic
benchmark, a numerical failure on CMAPSS above `d=24`, a recall-collapse
on SCANIA through policy re-calibration, and a marginal recall trade on
CARE. The frozen `d=16` therefore stays the default; `d` is a per-dataset
decision to be made alongside `K` and the commissioning split, and this
matrix is the evidence for each dataset.

## 5. Tabular dimensional calibration shift resolved (`TabularEncoder(rcond=1e-4)`)

Section 4 identified a sharp contradiction on SCANIA: moving from `tab_d16`
to `tab_d32` improved ranking performance (AUROC $0.541 \to 0.571$, AP
$0.0321 \to 0.0333$), yet near-failure recall collapsed from $0.228 \to 0.022$.

### Mathematical Root Cause

Profiling the calibration scores revealed why:

1. **Covariance Conditioning**: On SCANIA's 105 discrete operational counters
   and histograms, minor PCA directions (dims 16–31) have tiny noise
   eigenvalues ($\lambda \approx 10^{-11} \dots 10^{-12}$).
2. **Variance Inflation**: `NoiseWhitener` inverted these eigenvalues using
   a floating-point machine floor (`1e-12 * max_lam`). This scaled minor
   components by $10^6 \dots 10^8$, inflating dimension variance to 30 billion.
3. **Calibration Tail Shift**: The $(1-\alpha)$ conformal threshold on
   calibration vehicles skyrocketed by **565×** (from 1,368,182 at `d=16`
   to 772,671,329 at `d=32`).
4. **Persistence Filter Starvation**: Consequently, single-window test
   exceedance plummeted from 13.68% to 1.59%. Under $k=3$ persistence, the
   probability of three consecutive exceedances collapsed ($(0.0159)^3 \approx 4 \times 10^{-6}$),
   reducing alarmed vehicles from 978 (19.29% healthy rate) to 138 (2.75%
   healthy rate).

### The Fix: Regularized Noise Whitening

In `src/uengram/encoders.py` and `src/uengram/encoder_tab.py`:
- Added an explicit condition-number floor `rcond: float = 1e-12` to
  `NoiseWhitener` (retaining 100% byte-identical backward compatibility for
  all DSP and CNN encoders).
- Configured `TabularEncoder(rcond=1e-4)`: capping the condition number at
  $10^4$ bounds noise variance to sane levels while preserving the +0.030
  AUROC ranking advantage of higher dimensional tabular representations.

### Measured SCANIA Results (`results/scania_tab_reg_sweep.json`, `rcond=1e-4`)

| Encoder | Whitening | Novelty AUROC | Novelty AP | Near-Failure Recall | Healthy Alarm Rate |
|---|---|---:|---:|---:|---:|
| `tab_d16` | unregularized | 0.541 | 0.0321 | 0.228 | 0.193 |
| `tab_d16` | **regularized (`1e-4`)** | 0.540 | 0.0316 | **0.301** (+7.3%) | 0.233 |
| `tab_d32` | unregularized | **0.571** | **0.0333** | 0.022 | 0.027 |
| `tab_d32` | **regularized (`1e-4`)** | 0.566 | 0.0330 | **0.066** (3× higher) | 0.044 |
| `tab_d48` | regularized (`1e-4`) | 0.562 | 0.0330 | 0.000 | 0.0004 |

On `tab_d16`, regularized whitening raises near-failure recall from 22.8% to
30.1% (+7.3% absolute gain across classes 1–4, with class 2 recall jumping
from 28.6% to 42.9%). On `tab_d32`, regularization triples recall from 2.2%
to 6.6% while preserving a superior AUROC (0.566 vs 0.540). At $d=48$, the
48-dimensional volume still pushes the $(1-\alpha)$ quantile too far into
the tail for $k=3$ persistence, confirming that $d=16$ remains the optimal
operating dimensionality for this coarse, sparsely-logged tabular stream.
