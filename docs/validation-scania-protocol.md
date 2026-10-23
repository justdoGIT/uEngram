# Frozen SCANIA Component X validation protocol

This protocol is frozen before any SCANIA validation score. It evaluates a
real, independent fleet dataset without treating its labels as runtime input.

## Source and integrity

- Source: SCANIA Component X, SND/researchdata record `2024-34/3`.
- Archive: `/home/kk/Backup/data/uengram/scania/ComponentX.zip`.
- License: CC BY 4.0, as recorded by the dataset survey and source record.
- The archive's bundled `sha256.txt` is checked against every extracted data and
  documentation member before scoring. The archive is not committed to git.

## Data split and causality

The source provides training, validation, and test partitions. This run uses
training operational readouts only for commissioning and calibration, and the
held-out validation operational readouts for test scoring.

1. Load rows in file order, retaining `vehicle_id`, `time_step`, and the 105
   anonymized operational features.
2. Use `train_tte.csv` only to identify training vehicles with
   `in_study_repair == 0`. This is a training-set label, not a validation label.
3. Split healthy training vehicles deterministically by `vehicle_id % 5`: IDs
   not divisible by 5 are commissioning (`fit`); IDs divisible by 5 are
   calibration (`cal`). No validation row participates in either fit.
4. Use every validation operational row as `test`. Validation class labels are
   read only after streaming, for final point-level evaluation.
5. Preserve each source file's row order. A vehicle boundary is a logging
   segment boundary; its first row is unmonitorable so a prior vehicle cannot
   affect its counters. All later rows are monitorable.
6. Set operating state to one state (`0`): SCANIA publishes no compressor-like
   operating-mode or alarm channels. Use `time_step` only as an ordered feature;
   it is not converted into wall-clock lead time.

## Encoder and scores

- Encoder: `TabularEncoder(d=16)`, fitted only on commissioning rows,
  `noise_lag=1`, one group per logging segment.
- Detector: the existing `uengram` scorer, seed `0`, budget `B=256`, latent
  dimension `D=16`, in the continuous fit/calibration/test order.
- Alarm threshold and persistence: the frozen reference configuration
  `AlarmConfig(alpha=0.003, k_novelty=3, k_health=0, per_state=True,
  min_state_windows=100, new_state_alert=True, health_needs_baseline=True)`.
- Baselines and thresholds are fitted from commissioning/calibration rows only.

## AND and OR runs

Two policy runs consume identical frozen scores and the same test order:

1. **AND/control:** `AlarmLogic()` with no evidence array. This is the existing
   uncorroborated policy and is the primary SCANIA anomaly result.
2. **OR:** `AlarmLogic(combine="or", error_k=3)` with one predeclared external
   evidence source: `missing_any`, true when any raw operational feature is
   missing in that row, false otherwise. This is a data-quality tripwire, not a
   component-failure label; SCANIA provides no independent physical threshold
   or alarm channel. The raw missingness mask is computed before imputation.

The OR policy's error tripwire uses the same counter persistence model and
`error_k=3`. Validation labels never enter the evidence array. An additional
no-evidence OR control must be byte-identical to the AND/control run; otherwise
the OR implementation is rejected.

## Evaluation

SCANIA validation labels identify the class of the final observed row for each
vehicle: class 0 is healthy or more than 48 time steps from repair; classes 1–4
are progressively nearer repair. They do **not** provide exact failure
intervals or wall-clock timestamps, so MetroPT event lead-time and
false-alarms-per-week metrics are not claimed here.

For each validation vehicle, evaluate the alarm state at its final observed row:

- `recall_by_class[c]`: fraction of vehicles in classes 1–4 whose final row is
  active or opened by the alarm;
- `healthy_alarm_rate`: fraction of class-0 vehicles whose final row is active
  or opened;
- `near_failure_recall`: recall over classes 1–4;
- `alarm_vehicles`: number of validation vehicles with an alarm at the final
  row.

Also report row-level novelty AUROC/AP against the point label `(class > 0)` as
an exploratory ranking diagnostic. It is not a failure-interval metric.

## Acceptance and limitations

- Archive manifest verification must pass.
- The no-evidence OR control and AND/control must match in active/opened arrays
  and all reported metrics.
- Report the missingness-tripwire OR result separately from the control.
- Any claim about early warning, hours before failure, or compressor-domain
  transfer is out of scope: SCANIA's validation labels are coarse relative-time
  classes and its operational variables are anonymized counters/histograms.
