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

## Post-preregistration external-source screening (2026-10-25)

This source screen does not change the eligibility rules above, access any test labels, or make a public benchmark equivalent to a custodian-held cohort.

- **EngineAD** ([paper](https://arxiv.org/abs/2603.25955), [Borealis record](https://doi.org/10.5683/SP3/TX13P1)): the paper describes six months of real telemetry from 25 commercial vehicles, 13 engine sensors, and technician labels informed by sensor traces and maintenance reports. The released files are five-minute segments with eight PCA channels. Borealis metadata reports 25 files totaling 6,069,110,651 bytes, a CC BY-NC 4.0 license, and file access by request. This is the strongest engine-domain candidate for a separate segment-level CNN study, but it does not provide the pointwise physical-fault intervals and independent label custody required here. No access request was submitted.
- **NGAFID Aviation Maintenance** ([paper](https://arxiv.org/abs/2210.07317), [dataset](https://doi.org/10.5281/zenodo.6624956)): the release contains 31,177 flight-hours, 28,935 flights, 23 one-second sensor channels, and labels associated with 2,111 unplanned maintenance events. Labels identify flights before or after maintenance; flight dates and aircraft serial numbers are removed. It is suitable for a separately defined flight-level prediction study, not event timing or false alarms per monitored week.
- **Hill of Towie** ([latest Zenodo version](https://doi.org/10.5281/zenodo.20204946), [analysis/data tools](https://github.com/resgroup/hill-of-towie-open-source-analysis)): the operator-published record describes 21 turbines, ten-minute SCADA and alarm logs from 2016 through April 2026, and shutdown-duration values derived from SCADA and RES operator logs. The license is CC BY 4.0. These are operational alarms/downtime, not explicitly adjudicated physical-fault intervals; the 29.7 GB full archive also exceeds the 18 GB available on the local `/home` volume at screening time.
- **Kelmarsh and Penmanshiel** ([Kelmarsh](https://doi.org/10.5281/zenodo.16807551), [Penmanshiel](https://doi.org/10.5281/zenodo.16807304)): Cubico released ten-minute SCADA and event data under CC BY 4.0 for six and fourteen turbines, respectively, through 2024. The source metadata does not define an independently adjudicated physical-fault label. A third-party parser treats one event category as a failure label, but that interpretation is not enough to establish the required label provenance. The Zenodo API and a tested file endpoint returned HTTP 403 in this environment; no files were downloaded.
- **CARE to Compare** ([Zenodo version 15846963](https://doi.org/10.5281/zenodo.15846963)) has timestamped operator-derived event labels, but the project already attempted all 95 event datasets across 36 turbines in its v2 confirmation. It is not untouched.

### Screening outcome

None of the accessible sources satisfies every preregistered condition: untouched real-field assets, time-stamped physical-fault intervals, independent operator adjudication and label custody, asset-held-out chronological testing, and a power-calculated sample plan. No external data was staged, no input adapter was changed, and no candidate was run on these sources. The original final-cohort claims remain blocked; no substitute metrics are reported.

The most promising next data path is a custodian-supplied cohort with pointwise event adjudication. EngineAD is a possible separate, non-commercial segment-level study only after access is granted and its use is approved; its labels and endpoints do not satisfy this preregistration as written. A new data request would disclose the requester's identity and research purpose, so none was submitted.
