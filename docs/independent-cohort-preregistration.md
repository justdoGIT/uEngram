# Independent field-cohort evaluation preregistration

## Status and scope

This protocol is fixed in version control before any new cohort is staged or its labels are accessed. Candidate source is frozen at commit `8335dc52b0255c4f3dbabb5ed14762eb0110dbd6`. A change to candidate code, scoring, alarm logic, or criteria invalidates this protocol for the current test cohort; changes require a new preregistration and a new untouched cohort.

This is a repository-level preregistration, not an external registry entry. No new-cohort results are claimed here.

## Frozen reference and candidates

The reference encoder is the repository DSP path. The frozen UEngram alarm policy is `bench.run_v2.FROZEN["uengram"]`:

- per-window `alpha = 0.003`
- novelty persistence `k_novelty = 3`
- health persistence `k_health = 0` (disabled)
- per-state thresholds enabled; minimum state support `100`
- new-state alert enabled
- adaptive level `aci_gamma = 0.0` (disabled)
- `health_needs_baseline = true`
- immediate shock multiplier `shock_mult = 0.0` (disabled)

No thresholds are fit on final test data; thresholds are calibrated only on a
separate normal period using the frozen policy. Apply it consistently to DSP,
multiscale, CNN, and SSL-CNN candidates. Report multiscale level and dynamics
separately; never fuse them after viewing test results. Physical-residual
scoring is exploratory, separate, and ineligible for promotion.

The candidate set is fixed: DSP; multiscale combined, level, and dynamics; existing CNN seeds `0, 1, 2`; and SSL-CNN seeds `0, 1, 2`. CNN training uses `30` epochs. Candidate selection uses only the synthetic challenge suite, with the existing lower-tail quantile `0.10`, and unlabeled normal commissioning windows. Neither test labels nor test metrics may select a model or seed.

## Cohort eligibility and custody

The cohort must consist of newly collected real field recordings from independently identified assets not present in any previously consumed trace. It must include timestamped sensor values, operating intervals, verified asset identities, pointwise physical-fault intervals, and independent operator adjudication with reliable event timestamps. Asset identities may be pseudonymized, but a custodian must verify there is no overlap with consumed assets or time ranges.

The final test assets and labels must remain inaccessible to model developers until the candidate code, split manifest, and analysis script are frozen. Training/selection, normal calibration, and final test periods are disjoint; the final test is asset-held-out and chronologically later. Commissioning and calibration use normal windows only. Fault labels are held by an independent evaluator until one final test run is complete.

Before acquisition, a power calculation must use the fixed 5-percentage-point
event-recall non-inferiority margin, one-sided significance level `0.05`, and
power `0.80` to set the required counts of independent assets, fault events,
and normal exposure. There is no outcome-dependent stopping. If any required
count cannot be collected, the result is underpowered and cannot justify
promotion.

## Frozen execution

1. Record cohort provenance, capture interval, pseudonymous asset IDs, label-adjudication procedure, and SHA-256 hashes in a signed data manifest. The custodian verifies asset/time disjointness from all consumed traces.
2. Fit each candidate on the first 80% of chronological normal commissioning windows. Use the remaining 20% only for the predeclared label-free synthetic challenge selection. Do not inspect fault labels during fitting or selection.
3. Calibrate score thresholds only on the separate normal calibration period, using the frozen alarm policy. No threshold, encoder, seed, feature, or event-matching rule may be changed after test labels are unlocked.
4. Run each frozen candidate once on the untouched test cohort. Save raw scores, alarms, configuration, code revision, and data hash to a new result path; never overwrite prior results.
5. The independent evaluator unlocks labels and computes the metrics below. Report every candidate and all three seeds; do not report only the best test seed. Any test-driven change requires a new independent cohort.

## Metrics and pass/fail rules

Primary metrics are event-level recall and false-alarm episodes per monitored operating week. Also report event-level precision, median detection lead time, AUROC as a diagnostic only, per-seed spread, asset-level results, and p50/p95 inference latency. Use the repository's frozen event and false-alarm episode definitions; publish the exact denominators and excluded intervals.

Compare each candidate with DSP on the same test assets. Compute paired one-sided
95% confidence bounds with a 10,000-replicate cluster bootstrap over independent
assets and fixed RNG seed `0`; preserve event pairing within each resampled
asset. Aggregate false alarms by asset-week before resampling. The test is
inconclusive if the power-calculated cohort size or independent-asset support
is not met.

A candidate passes the comparative gate only if all conditions hold:

- the lower one-sided 95% confidence bound for candidate-minus-DSP event recall is at least `-0.05`;
- the upper one-sided 95% confidence bound for candidate-minus-DSP false alarms per monitored week is at most `0`;
- at least one endpoint shows a statistically supported improvement: recall difference lower bound greater than `0`, or false-alarm difference upper bound less than `0`;
- no data leakage, asset overlap, post-unblinding tuning, or protocol deviation occurred.

Passing this comparative gate alone does not authorize deployment. Deployment additionally requires the operating owner to set and approve an absolute maximum false-alarm rate and minimum event-recall target before cohort collection, and the candidate's one-sided 95% bounds must meet both limits. The repository contains no approved site-specific alarm budget, so none is invented here.

## Current data availability and blocker

The accessible project data contains existing SKAB, CMAPSS, CARE, MetroPT2, and
MetroPT3 assets and prior experiment outputs. The local UEngram state directory
contains development experiment logs and weak report-association data, not new
field recordings. None is a newly collected, independently adjudicated field
cohort. `bench/domain_dataset.py` explicitly records that independent assets,
independently adjudicated pointwise labels, and untouched evaluation data are
missing. Re-splitting, relabeling, or downloading another public benchmark
would not satisfy this cohort requirement.

Therefore the candidate source and relative evaluation criteria are frozen, but cohort acquisition and the final evaluation are blocked until a data custodian supplies eligible recordings, independent adjudication, the power-calculated sample plan, and the operating owner's absolute alarm/recall limits. No final-cohort metrics may be reported before then.
