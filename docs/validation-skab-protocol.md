# Frozen SKAB validation protocol

This protocol is frozen before any SKAB detector score. It answers the
blocker in [research-remediation-plan.md](research-remediation-plan.md):
MetroPT-2/3 and CARE are consumed, and SCANIA Component X has only coarse
per-vehicle repair classes, not pointwise fault timestamps. SKAB is a
different real system with genuine, independently verified pointwise
fault-onset and fault-end labels, never used to develop or select this
project's encoder or alarm policy.

## Source and integrity

- Source: Skoltech Anomaly Benchmark (SKAB) v0.9, `https://github.com/waico/SKAB`.
- Pinned commit `b2c0d46c2971dcbfe71e26087b6d231998bb91c2` (2024-08-11).
- License: GPL-3.0, as declared by the repository's top-level `LICENSE` and
  README badge. No bundled per-file checksum manifest exists; each of the 35
  downloaded CSVs is verified against the git blob SHA-1 reported by GitHub's
  tree API at the pinned commit (`sha1("blob {len}\0" + content)`), which is
  content-addressed by the git object model, not a project-authored hash.
- Stored at `data/skab/` (4.3 MB total; not committed to git, consistent with
  this project's other external datasets).
- 35 files: `anomaly-free/anomaly-free.csv` (9,405 healthy rows, no label
  columns) plus 34 fault experiments (`valve1/0..15.csv`, `valve2/0..3.csv`,
  `other/1..14.csv`), each with `anomaly` and `changepoint` columns at 1 Hz.
- **Correction to the README's "contains a single anomaly" claim:** verified
  directly on every file's label column that each of the 34 fault files has
  exactly one *contiguous* run of `anomaly=1` (confirmed by run-length
  analysis before any detector score was generated); the 2-4 `changepoint`
  marks inside that run are internal phase boundaries (e.g. valve closing,
  stabilized, valve reopening, recovered), not separate faults. This matches
  the README's description once "single anomaly" is read as "single
  contiguous labelled interval."

## Physical system and why it is used despite the mismatch

SKAB is a water-circulation testbed (centrifugal pump, inlet/outlet valves,
accelerometers, current/voltage/pressure/temperature/flow sensors) with
real injected faults: valve throttling, rotor imbalance, fluid leaks and
additions, cavitation, overheated supply water. **This is not a compressor**
and not asset-disjoint in the multi-machine sense: all 35 files come from
one testbed. It is used for one specific property MetroPT/SCANIA/CARE
cannot provide: exact, dataset-verified pointwise fault start and end
times, enabling a real lead-time measurement instead of coarse repair
classes or a single interval per published report. Any result here is
evidence about transfer to a different real system with precise timing, not
compressor-domain confirmation.

## Data split and causality

1. Parse each CSV with its semicolon delimiter; keep the 8 sensor channels
   (`Accelerometer1RMS`, `Accelerometer2RMS`, `Current`, `Pressure`,
   `Temperature`, `Thermocouple`, `Voltage`, `Volume Flow RateRMS`).
2. Commissioning (`fit`): the first 30% of `anomaly-free.csv`'s rows,
   rounded down to a whole number of windows. Calibration (`cal`): the
   remaining rows of the same file. Both splits come from one uninterrupted
   healthy recording; no fault file contributes to either.
3. Test: all 34 fault files, each kept in its own file-order block.
4. Each file's real per-row timestamps are preserved internally (so a
   file's own internal logging gaps, e.g. `other/2.csv`'s single 247 s gap,
   still produce a `monitorable=False` window exactly as in the MetroPT
   protocol). Between the commissioning, calibration and every fault file,
   a synthetic 30-day gap is inserted into the window-building clock so
   that (a) no window ever bridges two files or bridges fit into cal, and
   (b) the ±24 h search window `evaluate_alarms` uses for "first alarm
   before this event" can never pick up an alarm from a different
   experiment. This gap is bookkeeping for the shared windowing code path;
   it does not assert anything about real elapsed time between experiments.
5. Windows: non-overlapping, `K=10` samples (10 s) per window, built with
   the same gap-aware, split-respecting procedure as the MetroPT loader.
   **Correction, made before any alarm metric was produced:** an initial
   `K=20`/70%-commissioning split left only ~141 calibration windows, too
   few for a finite conformal threshold at `alpha=0.003` (`ceil((1-alpha)*
   (m+1)) <= m` needs `m >= ~333`); `score_stream` had already run (so
   novelty/health arrays existed) but `apply_policy` raised
   `CalibrationUnavailable` before any alarm metric was computed, so this
   is a data-split fix, not a result-driven retune of the frozen alpha,
   k_novelty or any other alarm parameter. `K=10` with a 30/70
   commissioning/calibration split gives ~280 commissioning windows
   (enough for the 48->16 PCA/noise fit) and ~650 calibration windows.
   Operating state is a single state (`0`): SKAB publishes no digital
   load/mode channel comparable to MetroPT's compressor state.
6. Labelled events: one `(onset, offset)` interval per fault file, the
   first and last row with `anomaly=1`. `before_h=0`, `after_h=0`: SKAB
   explicitly labels every anomalous row, so the clean-negative definition
   is exactly "not in the labelled interval," with no added recovery
   margin (unlike MetroPT's compressor, which keeps running after a
   report).

## Encoder and scores

- Encoder: `DSPEncoder(d=16)` (the compressor-confirmed `dsp` encoder),
  fitted only on commissioning windows, `noise_lag=1`, one group per
  windowing segment, exactly as `bench.v2_scoring.encode` does for every
  other `dsp` stream. No encoder retraining or hyperparameter search.
- Detector: the existing `uengram` scorer, seed `0`, budget `B=256`
  (192 semantic + 64 episodic), latent dimension `D=16`, run continuously
  over fit → cal → test in that order (`bench.v2_scoring.score_embeddings`).
- Alarm policy: the frozen MetroPT-3-selected configuration from
  `bench.run_v2.FROZEN["uengram"]` exactly: `AlarmConfig(alpha=0.003,
  k_novelty=3, k_health=0, per_state=True, min_state_windows=100,
  new_state_alert=True, aci_gamma=0.0, health_needs_baseline=True)`,
  default `AlarmLogic()` (AND rule, counter persistence). No OR rule: SKAB
  has no independent external evidence channel to corroborate with.
- Real per-window engine actions (`absorb_semantic`, `new_candidate`,
  `step_change`, ...) are passed to the alarm policy exactly as in the
  MetroPT2/MetroPT3/CMAPSS confirmation runs — unlike the SCANIA
  validation, which used a placeholder action and so never exercised the
  `new_state_alert` veto. SKAB is windowed sensor data like MetroPT, not
  flat tabular rows, so the standard pipeline (`bench.alarm_eval.apply_policy`)
  is reused unmodified.
- Baselines and thresholds are fitted from calibration rows only, using
  `bench.alarm_eval.apply_policy`'s existing calibration/evaluation code
  (the same function MetroPT2, MetroPT3 and CMAPSS confirmation use).

## Evaluation

For each of the 34 labelled fault experiments, report exactly what
`bench.alarm_eval.evaluate_alarms` already computes for every other stream:
`events_detected` (alarm active during the labelled interval),
`lead_hours`/`first_alarm` (first alarm in the 24 h before the labelled
onset, converted to seconds given SKAB's short experiments),
`false_alarm_per_week` and `alarm_time_share_clean` over the test-split
clean (non-labelled) windows, and row-level `auroc`/`ap` of the novelty and
health scores against the pointwise `event_id() >= 0` label. Also report a
per-experiment breakdown (detected, first-alarm time relative to onset) so
a result is not hidden inside an aggregate.

**Caveat predeclared here:** the 34 experiments' clean (non-anomalous)
exposure totals only a few hours (each file is ~15-20 minutes, roughly
60-65% of it outside the labelled interval), far less than MetroPT's
months of data. `false_alarm_per_week` from this little exposure is
reported but is expected to be noisy; `alarm_time_share_clean` (a
duration-weighted fraction, not a rate extrapolated to a week) is the more
stable clean-time measure here and is reported alongside it.

## Acceptance and limitations

- The archive integrity check (git blob SHA-1) must pass for all 35 files
  before any score is generated.
- Report every one of the 34 experiments, not a selected subset.
- This is a different physical system from the project's compressor focus,
  and not a multi-asset cohort. A pass here is evidence that the frozen
  detector's novelty/health math generalizes to a system with precise
  pointwise labels; it is not compressor-domain confirmation and does not
  change the MetroPT-3/MetroPT2/CARE development-vs-confirmation status.
- A fail here is equally reportable: it would show the frozen policy does
  not transfer even to a system with clean ground truth, which narrows
  (not widens) the plausible causes of the compressor-domain shortfall.
