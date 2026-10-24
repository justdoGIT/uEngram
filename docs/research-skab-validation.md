# SKAB validation report

This report records the first validation of the frozen `uengram/dsp/v2_frozen`
detector against real, independently verified pointwise fault timestamps —
the specific gap [research-remediation-plan.md](research-remediation-plan.md)
identified: MetroPT-2/3 and CARE are consumed, and the SCANIA validation
(`research-scania-validation.md`) has only coarse per-vehicle repair classes.
The frozen procedure is [validation-skab-protocol.md](validation-skab-protocol.md).
Raw output: `results/skab_validation.json`.

## Data and integrity

- Source: SKAB v0.9, `https://github.com/waico/SKAB`, pinned commit
  `b2c0d46c2971dcbfe71e26087b6d231998bb91c2`, GPL-3.0.
- All 35 CSVs verified against GitHub's git blob SHA-1 before any row was
  parsed. 9,405 healthy commissioning/calibration rows plus 34 fault
  experiments, 46,806 rows total (4.3 MB).
- **Split correction, made before any alarm metric existed:** the protocol's
  first choice (`K=20`, 70% commissioning) left only ~141 calibration
  windows, too few for a finite conformal threshold at `alpha=0.003`
  (needs ~333). `apply_policy` raised `CalibrationUnavailable` before
  computing any metric. The corrected split (`K=10`, 30% commissioning) is
  recorded in the protocol; no alarm parameter (`alpha`, `k_novelty`, or any
  other field of `bench.run_v2.FROZEN["uengram"]`) changed.
- Detector, encoder and alarm policy are byte-identical to the frozen
  MetroPT-3-selected configuration; only the window length and
  commissioning/calibration split (data-engineering parameters, not alarm
  parameters) were chosen to make calibration possible on this dataset's
  size.

## Results

| Metric | Value |
|---|---:|
| Events detected | 5 / 34 |
| Row-level novelty AUROC / AP | 0.550 / 0.423 |
| Row-level health AUROC / AP | 0.539 / 0.454 |
| `alarm_time_share_clean` | 1.34% |
| `false_alarm_per_week` | 135.2 |
| Clean (non-anomalous) observed time | 6.21 hours |
| Mean lead time where detected | +40.4 s (after onset, not before) |

**`false_alarm_per_week` is not a meaningful rate here**, exactly as
predeclared in the protocol: 5 false-alarm episodes over only 6.21 hours of
clean exposure extrapolates to 135/week. `alarm_time_share_clean`
(duration-weighted, not extrapolated) is the more honest clean-time measure
and is, numerically, inside MetroPT's own 2% budget — but with 34
experiments of ~15-20 minutes each, this single number should not be read
as a rate claim either; it describes this specific 6.21-hour sample.

### Per-experiment pattern

| Fault file | Description | Detected | Lead |
|---|---|---|---:|
| `other/5.csv` | Sharp rotor imbalance | yes | +28 s |
| `other/6.csv` | Linear rotor imbalance | yes | +48 s |
| `other/7.csv` | Step rotor imbalance | yes | +49 s |
| `other/9.csv` | Exponential rotor imbalance | yes | +39 s |
| `other/12.csv` | Draining water until cavitation | yes | +114 s |
| `other/8.csv` | Dirac-delta rotor imbalance | no | — |
| `other/13.csv` | Two-phase-flow cavitation | no | — |
| `other/1-4,10,11,14.csv` | Fluid leaks/additions, water-level change, hot supply | no (7/7) | — |
| `valve1/0-15.csv` (16 files) | Inlet valve throttling | no (16/16) | — |
| `valve2/0-3.csv` (4 files) | Outlet valve throttling | no (4/4) | — |

Every detection is a vibration-forcing fault (4 of 5 rotor-imbalance
profiles, missing only the single-impulse Dirac-delta case) or the one
cavitation case reached by slowly draining the tank. Detection latency is
consistently 28-114 seconds **after** the labelled onset, not before: this
is a confirmation-type detection, not an early warning, on this dataset.
Every valve-throttling experiment (20/20, both inlet and outlet) and every
non-vibration fault (fluid level, leaks, hot supply, two-phase cavitation)
went undetected. A handful of files show an alarm earlier in the 24h search
window that had already cleared by the labelled onset (`valve1/0`,
`valve1/5`, `valve1/11`, `valve2/3`, `other/1`); these are transient false
positives before the fault, not genuine lead time, and are reported as
`detected: false` with a negative `lead_hours` in `per_experiment`.

## Interpretation and limits

- **This is not a compressor-domain result and not a multi-asset cohort.**
  All 35 files are one water-circulation testbed. It shows how the frozen
  detector's novelty/health math behaves on a different real system with
  exact pointwise labels, not whether it generalizes to other compressors.
- **The pattern is physically legible, not a black box**: the `dsp`
  encoder's band-energy features operate on all 8 channels including the
  two accelerometers, and the detections cluster exactly on the faults that
  change vibration spectral content (rotor imbalance) rather than the ones
  that change pressure/flow/temperature level alone (valve throttling,
  leaks, hot supply). This is a plausible mechanism given the feature
  design, not a proven causal account.
- **5/34 (14.7%) event recall is a low-recall result**, not a validated
  detector for this system. It narrows rather than resolves the compressor
  shortfall: the same frozen detector transfers some signal to a system
  with completely different physics and clean ground truth, but the
  transfer is partial and confirmation-only (no early warning observed).
- The 6.21-hour total clean-exposure sample is far too small to support a
  per-week false-alarm claim; only the duration-weighted
  `alarm_time_share_clean` is reported as a rate-like number, with the
  caveat that it is a single small sample.
- No retuning occurred between seeing this dataset's shape (row/window
  counts forcing the `K`/split correction) and generating its first alarm
  metric: the correction only concerns how many calibration windows exist,
  never any alarm threshold, persistence or veto parameter.

## Why the other 29 are missed (post-hoc diagnostic, not a new result)

This section reuses the already-committed SKAB scores to diagnose, not to
select a new configuration; it changes nothing about the frozen
`uengram/dsp/v2_frozen` candidate or the result above.

**It is not a PCA-discarding problem.** Projecting each experiment's
label-minus-clean mean feature-space difference onto the encoder's
retained top-16 vs. discarded 32 PCA dimensions shows the retained
fraction is similar (0.2-0.96) for detected and undetected experiments
alike; the top-16 keeps 84.3% of the standardized 48-feature variance.
The real split is magnitude: the 5 detected experiments' label-vs-clean
feature deltas are 100-1,000x larger (||delta||^2 = 6,906-92,511) than
every undetected experiment's (3-370), driven by `Accelerometer*RMS_mean`
for rotor imbalance and `Volume Flow RateRMS_mean` for the cavitation
case. Valve-throttling and leak/level/thermal faults are genuinely small
perturbations in this encoder's standardized feature space, not a
masked-but-present signal.

**Two distinct sub-causes, found by inspecting the raw novelty trace
against the calibrated threshold (1,551.6) for every undetected
experiment:**

1. **11 of 29** (`valve1/3,6,7,9,10,11,12,13`, `other/1,3,4,8`) do cross
   the threshold at least once *inside* the labelled interval, but only
   as an isolated single-window spike, never 3 consecutive/accumulated
   exceedances. `other/8` (the single Dirac-delta impulse rotor-imbalance
   case) is in this group: its physical fault is itself a single
   transient, not a sustained shift, so the frozen `k_novelty=3`
   persistence (a 30 s minimum at `K=10`) structurally cannot admit it.
   **Measured, not yet adopted:** rerunning the identical calibration
   with only `k_novelty` lowered (all else unchanged) gives
   `k=3: 5/34, 1.34% clean time, 5 false-alarm episodes` (the reported
   result) -> `k=2: 7/34, 1.85%, 8 episodes` -> `k=1: 16/34, 3.78%,
   21 episodes`. Recall roughly triples at `k=1`, and clean-time share
   and false-alarm episodes both roughly triple to quadruple. This is a
   real, quantified sensitivity/false-alarm trade, not a free win, and is
   not applied here: it would need its own calibration and validation
   pass before being proposed as a configuration change, exactly as
   `docs/research-alarm-tradeoff.md` found for the compressor data.
2. **18 of 29** never cross the threshold even once inside the labelled
   interval at any persistence setting: every valve1/valve2 file except
   the 8 listed above (12 of 20), plus the fluid-level, leak and
   hot-supply experiments (`other/2,10,11,13,14`). No persistence change
   can detect these; the raw novelty signal itself stays below threshold
   throughout.

**A separate, already-visible phenomenon**: several valve experiments
(e.g. `valve1/4`, `valve1/13`) show a large, well-sustained run of
exceedances (novelty up to ~2,290, many consecutive) immediately *after*
their labelled interval ends — the valve-reopening transient, not the
closed-valve fault itself. The detector reacts to this aftermath, not to
the labelled fault window. Counting these as "detected" would move the
evaluation's goalposts rather than fix the detector, so they are not
counted as detections above.

## RC2 slope-feature ablation: confirmed for the gradual-fault family

`src/uengram/encoder_slope.py` (`DSPSlopeEncoder`) adds, per channel, the
windowed mean minus its value `lag` windows earlier (zero across a segment
break), at lags 3 and 12 windows (30 s and 120 s at `K=10`), alongside the
unchanged DSP band features. It is a new, separately tested encoder
(`tests/test_encoder_slope.py`), not a change to the frozen `DSPEncoder`
used by every other result in this project. Run:
`python -m bench.validate_skab_slope --out results/skab_slope_ablation.json`.
The alarm policy is the same frozen `bench.run_v2.FROZEN["uengram"]`; only
the encoder differs from `results/skab_validation.json`.

| Metric | Frozen `dsp` | `dsp_slope` (lags 3,12) |
|---|---:|---:|
| Events detected | 5/34 | **8/34** |
| Row-level novelty AUROC / AP | 0.550 / 0.423 | 0.607 / 0.474 |
| `alarm_time_share_clean` | 1.34% | 5.09% |
| False-alarm episodes (6.21 h clean exposure) | 5 | 10 |

The three newly detected experiments are exactly the ones diagnosed above
as slow/gradual and non-vibration: `other/10.csv` (slow water-amount
increase), `other/13.csv` (two-phase-flow cavitation) and `other/14.csv`
(hot water supply). **Zero regressions**: all 5 baseline detections
(`other/5,6,7,9,12`) are preserved. Every valve-throttling experiment
(20/20) is still undetected — the slope feature targets gradual drift, not
a valve's throttling transient, and did not touch that group, consistent
with the isolated-spike diagnosis above being a persistence problem, not a
feature problem.

This confirms RC2's hypothesis specifically for the gradual/non-vibration
fault family, with a real cost: false-alarm episodes roughly double and
clean-time share roughly quadruples (from an already-tiny 6.21 h clean
exposure sample, so both numbers carry the same low-exposure caveat as the
confirmation (SKAB's own result now informed the choice of lags), and not
a proposal to replace the frozen `dsp` encoder project-wide. The
revalidation is in the next section, and its answer is no: this encoder
is a severe regression on the synthetic benchmark. The isolated-spike
family (11/29) and the valve-throttling family (20/20, still fully
undetected) remain open; see the persistence-sweep section below for why
neither was fixed either.

## Project-wide revalidation: a regression everywhere except its target

`DSPSlopeEncoder` was wired into the same shared encoder factory every
other benchmark uses (`bench.run_synth.make_encoder`, `"dsp_slope"`;
`bench/common.py`'s `encode_windows` threads CMAPSS's unit ids and
MetroPT's logging-segment ids through at encode time) and run through the
full existing harnesses, same alarm policy, same seeds, no other change:
`python -m bench.run_synth --seeds 0-9 --encoders dsp_slope`,
`python -m bench.run_cmapss --seeds 0-4 --encoders dsp_slope`,
`python -m bench.run_metropt --seeds 0-4 --encoders dsp_slope`. Results in
`results/synth_slope_runs.jsonl`, `results/cmapss_slope_runs.jsonl`,
`results/metropt_slope_runs.jsonl`.

**Synthetic stream (10 seeds, every `uengram*` variant): a severe
regression.**

| Metric (`uengram`, mean of 10 seeds) | Frozen `dsp` | `dsp_slope` |
|---|---:|---:|
| `event_recall` | 1.000 | **0.232** |
| `seizure_recall` | 1.000 | **0.000** |
| `auroc` | 0.9998 | 0.883 |
| `ap` | 0.9955 | **0.068** |
| `writes` | 2,762 | 13,298 (4.8x) |

Every `uengram` variant (`uengram`, `uengram_fixedvar`, `uengram_nopromo`,
`uengram_frozen`, `uengram_nostep`) collapses the same way across all 10
seeds: event recall drops to 0.20-0.25, seizure recall to exactly 0, AP to
0.07-0.08, and writes roughly quadruple. Adding 16 slope columns to 48 raw
features (a 33% size increase) changes which directions the `d=16` PCA
keeps; on synthetic data, where the anomaly signal already lives cleanly
in the spectral/level features, the slope columns' own variance (large,
since synthetic regimes switch and naturally produce big slopes) competes
for and wins PCA budget that used to carry the anomaly signal, and raises
baseline novelty enough to break promotion/supersession (`stale_at1` rises
10-100x). This is a root-cause hypothesis consistent with the measurement,
not independently verified by a further ablation.

**CMAPSS FD001 (5 seeds) and MetroPT-3 (5 seeds): mixed, not a win.**

| Metric (`uengram`, mean of 5 seeds) | CMAPSS `dsp` | CMAPSS `dsp_slope` | MetroPT `dsp` | MetroPT `dsp_slope` |
|---|---:|---:|---:|---:|
| novelty `auroc` | 0.684 | 0.651 | 0.979 | 0.968 |
| novelty `ap` | 0.246 | 0.206 | 0.502 | 0.466 |
| `health_auroc` | 0.999 | 0.9999 | n/a | n/a |
| `rul_rmse` | 42.05 | **35.25** | n/a | n/a |
| `rul_rmse_health` | 25.45 | 31.28 (worse) | n/a | n/a |
| `false_alarm_per_week` | n/a | n/a | 14.12 | **11.35** |

Health-score quality and MetroPT false-alarm rate improve slightly; raw
novelty AUROC/AP and the health-based RUL table get worse. `uengram_frozen`
on MetroPT shows the opposite false-alarm direction (4.91 -> 10.25/week,
worse). No metric improves consistently across both real datasets, let
alone all three benchmarks.

**Verdict: `DSPSlopeEncoder` is not a general-purpose replacement for
`DSPEncoder` and must not be selected for any confirmed or development
result outside this SKAB ablation.** It stays wired into the shared
factory only so this revalidation could reuse the existing harnesses
unmodified; nothing in `bench.run_v2.FROZEN`, `bench.common.ALL_METHODS`,
or any committed confirmation config references `"dsp_slope"`. The
evidence directly contradicts treating any single encoder (this one or
the frozen `DSPEncoder`) as a universal "go-to" for anomaly detection,
wear tracking and general condition monitoring: the right features are
domain- and fault-dependent, and a change that helps one real dataset's
specific fault family measurably hurts a controlled benchmark built to
catch exactly this engine's failure modes.

## Persistence/threshold sweep: no operating point dominates the baseline

`results/skab_persistence_sweep.json` searches, on the `dsp_slope`-encoded
SKAB scores, for a persistence/alpha configuration that detects more
events **and** raises fewer-or-equal false alarms than the frozen
`k_novelty=3`/`alpha=0.003` baseline (8/34, 10 episodes). None exists in
the configurations tested:

| Config | Detected | FA episodes | Clean-time share |
|---|---:|---:|---:|
| `k=3` counter (baseline) | 8/34 | 10 | 5.09% |
| `k=2` consecutive | 10/34 | 13 | 5.67% |
| `k=2` EWMA | 11/34 | 14 | 5.15% |
| `k=2` m-of-n (n=5) | 12/34 | 14 | 11.08% |
| `k=2` m-of-n (n=8) | 12/34 | 14 | 16.23% |
| `k=1` m-of-n (n=3) | 25/34 | 25 | 14.18% |
| `k=1` counter | 23/34 | 26 | 7.40% |
| `alpha=0.001`/`0.0015`, `k=1` | — | — | **calibration fails**: too few calibration windows |

Every configuration trades detections for false alarms roughly in
proportion; the "least-bad" single step is `k=2` consecutive (+2 detected,
+3 episodes) or `k=2` EWMA (+3 detected, +4 episodes, lower clean-time
share than consecutive). Tightening `alpha` to compensate for lower
persistence is not feasible: `alpha=0.001` already needs roughly 1,000
calibration windows by the conformal rank rule and SKAB's commissioning
recording supplies 657; this is not a parameter-search gap, it is a data
size limit. **The valve-throttling and isolated-spike families are not
fixed by any tested persistence or threshold change**; a genuine fix would
need either more calibration data than this testbed recorded, or a
different alarm-combination architecture (e.g. requiring two
independently-thresholded channels to agree), which is not implemented.
