# Dealing with the v8 root causes, and what to test them on

v8 ([research-alarm-components.md](research-alarm-components.md)) found that no combination of
alarm components reaches the MetroPT3 budget (4/4 detected, ≤2% clean alarm time, ≤1 false
alarm/week), for physical reasons, not a tuning gap. This plan maps each root cause to concrete
remediations, states what each would cost and what it cannot fix, and corrects the dataset
survey that was run to find fresh validation data.

## 1. Root causes and remediation options

### RC1 — The fault's operating state never occurred in commissioning
492 of 495 MetroPT3 event windows are sustained compressor load; commissioning has 0 such
windows, calibration has 1. Any setpoint that covers that state is therefore learned from
healthy *or* faulty load, never both.

| Option | Mechanism | Cost / new failure mode | Falsifiable test |
|---|---|---|---|
| **Longer commissioning** | Wait for the state to occur naturally before deploying its setpoint (`unbaselined="commission"`, already built) | Delays coverage of that state by however long it takes to occur; v8 showed the state's *first* natural occurrence can itself be healthy-but-unusual, so an unlucky commissioning window still mislearns the reference | Replay with longer commissioning windows (90/180 days) on a machine where sustained load occurs within that time; check whether the learned reference still flags Jul 15-like events |
| **Operator-confirmed health labels for that state** | A human confirms N windows of sustained load are healthy before they become the reference, instead of trusting whichever windows arrive first | Needs a human in the loop at commissioning; one wrong confirmation poisons the reference | Compare the operator-confirmed reference's false-alarm rate against the naive online-commissioned one on held-out healthy sustained-load windows |
| **Physical residual instead of statistical distance** | Model the *expected* reading given load (e.g. pressure recovery time, loaded-vs-unloaded pressure drop) from physics/engineering limits, not from data in that state | Needs an engineering model, not just data; the project's own `bench/physical_residuals.py` attempt (`results/v3_physical.jsonl`) got 4/4 on MetroPT3 at 4.99–5.74% clean time — closer to the budget than any statistical alarm, but still over it, and it transferred badly to MetroPT2 (3/4 at 0.9% clean time on Jan, but up to 17.1% false-alarm rate depending on config; 2/4 on Apr) | Already run; see `docs/EXPLAINED.md` section 18 and `results/v3_physical.jsonl` for exact numbers. Re-run with the v8 corroboration/persistence fixes applied to the physical score |

### RC2 — No pre-failure signal inside the fault state
Within sustained load, healthy-vs-pre-report AUROC is 0.07 (health) / 0.35 (novelty): the score
does not separate "about to fail" from "healthy and loaded" at all.

| Option | Mechanism | Cost | Falsifiable test |
|---|---|---|---|
| **Different features**: rate of change, not level | A leak develops gradually; a slope (e.g. d(reservoir pressure)/dt over the last hour) may separate "stable load" from "slowly losing pressure" where the level alone does not | New feature engineering, no guarantee it exists in this data (not tested here) | Compute the slope feature on MetroPT3's sustained-load windows and measure its AUROC against the same 24 h-before-report split used in v8, before building anything on it |
| **Admit no pre-failure signal exists in this state with these sensors** | Stop claiming early warning for this failure mode; keep the alarm as a confirmation signal only | None — it is honest, not a fix | Already the v8 finding; no further test needed, just don't oversell it |

### RC3 — Calibration drift (seasonal)
`reservoirs_min_bar` is in error 0% of healthy MetroPT2-Jan windows in January, 100% in
April–May; `oil_temp_c` is 19–76% by month on MetroPT3. A single frozen calibration week cannot
cover this.

| Option | Mechanism | Cost | Falsifiable test |
|---|---|---|---|
| **Rolling recalibration** | Refit the per-state setpoint on a trailing window (e.g. last 30 days of healthy-looking data) instead of once at deployment | Needs a rule for "healthy-looking" without labels (risk: recalibrating on an ongoing fault normalizes it away) | Replay with a rolling 30/60/90-day calibration window on MetroPT3 and MetroPT2; check whether the false-alarm rate stays flat across seasons instead of drifting |
| **Seasonal/ambient conditioning** | Model each channel's healthy range as a function of ambient temperature (or month) instead of one constant range | Needs an ambient-temperature channel or a reliable month proxy; adds a nuisance parameter to fit | Check whether oil_temp_c's monthly drift in section 6 of v8 correlates with an available ambient proxy; if yes, fit and remeasure the monthly error share |
| **Already-built ACI (adaptive conformal level)** | `aci_gamma` adapts the *novelty* level online | Only implemented for novelty, not health; v4/v8 found no recall benefit from it alone | Extend `aci_gamma`-style adaptation to the health channel and rerun the v8 factorial with it as a 9th factor |

### RC4 — The machine's own threshold signal already catches some failures, at idle/cycling
On MetroPT2, both failures are caught by `Oil_level` 3.0 h and 59.4 h before the LPS warning,
while uEngram never crosses its setpoint in the event windows.

| Option | Mechanism | Cost | Falsifiable test |
|---|---|---|---|
| **OR rule**: alarm on uEngram crossing OR a dataset error, instead of requiring both | Lets the documented threshold signal raise the alarm on its own when uEngram is silent | Not the rule the user originally specified (AND corroboration); changes what "the alarm" means; needs its own false-alarm budget check, since the dataset error alone may fire more often than uEngram | Implement as a third `AlarmLogic.combine` mode (`and`/`or`) alongside the existing AND; rerun the v8 factorial with it; check MetroPT2 Oil_level-alone false-alarm rate on calibration-period healthy windows first |
| **Two parallel alarms, reported separately** | Keep uEngram's AND-corroborated alarm and the raw threshold signal as two outputs, never merged | No model risk, but doesn't produce one alarm — it's what MetroPT2 already effectively has built in (LPS/Oil_level on the dashboard) | N/A — this is the status quo for threshold signals; the only "test" is whether uEngram adds anything beyond it, which v8 already measured (it does not, on these failures) |

### RC5 — Exceedances cluster in time
Already modeled exactly in v8 (two-state Markov chain, `bench/alarm_math.py`). Not a defect to
fix; it is why persistence-filter choice barely matters (section 2/4 of v8). The actionable
consequence: **use the measured (a, b) to pick k and the filter type for a target false-alarm
rate**, rather than guessing k.

| Option | Mechanism | Cost | Falsifiable test |
|---|---|---|---|
| **ARL-targeted k** | Given measured a, b on a machine's own calibration data, solve `arl0(model, k, a, b)` for the smallest k meeting the false-alarm budget | Needs enough calibration data to estimate b (requires exceedances to occur at all during calibration) | Already implemented (`bench.alarm_math.arl0`); apply it as the k-selection rule instead of a fixed grid, and verify predicted vs. observed rate on a validation stream |

## 2. What remediation cannot fix, on this data
- RC1 and RC2 together mean: **on MetroPT3, no alarm built from these channels and this
  commissioning period can warn before the Jul 15 report without exceeding the clean-time
  budget.** The physical residual model gets closest (4.99%, 2.5× the budget) but is not within
  it, and v3 already found it transfers inconsistently to MetroPT2.
- RC4's fix (the OR rule) only helps where a documented threshold signal exists and is itself
  reliable. It does not create an early-warning signal uEngram does not have.
- None of these can be properly validated on MetroPT3/MetroPT2/CARE/CMAPSS: they are consumed
  (used to develop and select every configuration so far). A remediation "tested" on them again
  is development, not confirmation.

## 3. Dataset survey: corrected finding

A sub-agent survey (`local://validation-datasets.md`) ranked candidate fresh datasets. Checking
its top recommendation against the Zenodo API overturned it:

**"Original MetroPT" (Zenodo record 6854240) is not a fourth recording.** It contains exactly
one file, `dataset_train.csv`, md5 `83c713c0f8a6c2f24418cca999b2a8eb`, 1646.2 MB. That file is
byte-identical (same md5) to the `dataset_train.csv` inside MetroPT2 (Zenodo 7766691), which
this project already consumed as `metropt2_jan`. **There is no fresh Metro do Porto recording
beyond what is already on disk.** The survey's #1 and #2 ranked candidates were MetroPT-original
and SCANIA; with MetroPT-original eliminated, SCANIA Component X is the strongest verified
candidate.

Verified via the Zenodo API directly (sizes, licenses, file lists — not the survey's secondary
sources):

| Dataset | Record | License | Size | Real failure/event labels | Machine type match |
|---|---|---|---|---|---|
| SCANIA Component X | researchdata.se 2024-34/3 | CC BY 4.0 | 1.54 GB (API: direct ZIP) | Workshop repair records, 23,550 vehicles | Different (truck fleet; anonymized component) |
| Kelmarsh wind farm | Zenodo 16807551 | CC BY 4.0 | 3.84 GB, 15 files | SCADA + status/event logs, 6 turbines, 2016–2024 | Different (wind turbine) |
| Penmanshiel wind farm | Zenodo 16807304 | CC BY 4.0 | 7.49 GB, 26 files | Same family, 14 turbines | Different (wind turbine) |
| Hill of Towie wind farm | Zenodo 14870023 (v1) / 20204946 (v2) | CC BY 4.0 | 12.6 GB / 29.7 GB | `Hill_of_Towie_alarms_description.csv` (schema) + `Hill_of_Towie_ShutdownDuration.zip` (19.8 MB) + per-year SCADA (~1.4 GB/year) | Different (wind turbine) |
| CARE to Compare | Zenodo 15846963 | CC BY-SA 4.0 | 5.50 GB | Already consumed (v2 confirmation) | — disqualified, not fresh |

None is a compressor or pneumatic machine: the closest physical match to MetroPT (a machine
with distinct load/idle/cycling states and documented threshold alarms) does not exist in open
data beyond what's consumed. The two realistic choices are therefore a different real fleet with
real repair records (SCANIA) or a different real machine type with its own alarm log (a wind
farm).

### Disk constraint
`/home/kk/Backup` has **16 GB free (97% full)**, already below the fleet's 35 GB auto-prune
threshold (`scripts/cluster-storage-guard.sh`). `data/uengram` already holds 11.4 GB (CARE 8 GB
+ MetroPT2 2.8 GB + MetroPT3 581 MB). Only SCANIA (1.54 GB) and Kelmarsh (3.84 GB) fit without
first freeing space; Penmanshiel and Hill of Towie do not.

## 4. Decision and execution

The selected remediation is the independently persisted **OR** rule:
`AlarmLogic(combine="or", error_k=3)`. The existing AND rule remains the
primary uEngram result; OR is an opt-in output because it changes the alarm
contract and can raise false alarms from its external evidence source.

The selected validation dataset is **SCANIA Component X**. The archive was
downloaded to the approved near-full `/home/kk/Backup` volume, its bundled
SHA-256 manifest passed for every member, and the frozen protocol is recorded
in [validation-scania-protocol.md](validation-scania-protocol.md).

The implementation and validation are complete:

1. The OR rule raises an independent `error` alarm only when a real evidence
   array is supplied. With no evidence, its active and opened arrays are
   identical to AND.
2. On the fixed MetroPT3 development replay, OR detected 2/4 events but raised
   1.217 false alarms/week and occupied 15.15% of clean time. This is
   development evidence, not fresh confirmation.
3. On the untouched SCANIA validation split, AND had 22.79% recall over
   classes 1–4 and a 19.29% class-0 final-row alarm rate. The predeclared
   missingness tripwire changed these to 23.53% and 20.71%, respectively.
   The no-evidence OR control matched AND exactly.

SCANIA has no independent physical alarm channel and no exact failure
timestamp. Its OR result is therefore a data-quality tripwire, not evidence
that the component-failure label itself independently raised the alarm. Full
metrics and the independent recomputation are in
[research-scania-validation.md](research-scania-validation.md).

## 5. SKAB: a fresh cohort with pointwise labels, different physics

SCANIA and CARE both lack what section 2 flagged as missing: exact,
independently verified pointwise fault timestamps. The Skoltech Anomaly
Benchmark (SKAB, `https://github.com/waico/SKAB`, GPL-3.0, pinned commit
`b2c0d46c2971dcbfe71e26087b6d231998bb91c2`) provides them, at the cost of
being a water-circulation testbed, not a compressor, and not a multi-asset
cohort (all 35 files are one testbed). The frozen protocol is
[validation-skab-protocol.md](validation-skab-protocol.md); full results in
[research-skab-validation.md](research-skab-validation.md).

The byte-identical frozen `uengram/dsp/v2_frozen` detector and alarm policy
detected 5 of 34 labelled faults (14.7%), all of them vibration-forcing
faults (4 of 5 rotor-imbalance profiles plus one cavitation case) with
detection 28-114 seconds **after** onset, never before. Every valve-
throttling experiment (20/20) and every non-vibration fault (leaks, water
level, hot supply, two-phase cavitation) went undetected. Row-level novelty
AUROC was 0.550, barely above chance.

This narrows rather than resolves RC1/RC2: the same frozen math carries
some real signal onto a system with totally different physics and exact
ground truth, but transfer is low-recall and confirmation-only, matching
the compressor-domain finding that no pre-failure signal was observed
ahead of a labelled event. It does not change MetroPT-3/MetroPT2/CARE's
development-vs-confirmation status, and SKAB is not itself a candidate for
further alarm-policy tuning: it was scored once, with the frozen policy,
specifically to test transfer, not to select a new configuration.

A follow-up RC2 test confirmed the diagnosis: `DSPSlopeEncoder`
(`src/uengram/encoder_slope.py`, a windowed-mean slope per channel added
to the unchanged DSP band features, tested in isolation from the frozen
`DSPEncoder`) rescued exactly the three gradual, non-vibration faults the
plain encoder structurally cannot see (`other/10,13,14`), 8/34 total, with
zero regressions, at roughly double the false-alarm episodes (5 -> 10)
and clean-time share (1.34% -> 5.09%) over the same ~6.2 h clean-exposure
sample. Valve-throttling (20/20) remains fully undetected: that family's
root cause is isolated single-window spikes failing the persistence
filter, not a missing feature, per `research-skab-validation.md`. This is
a second SKAB-specific ablation (SKAB's own result shaped the lag choice),
not a project-wide encoder change. The revalidation was run (not left
undone): `python -m bench.run_synth/run_cmapss/run_metropt --encoders
dsp_slope`, full seed counts, same alarm policy. Verdict: **reject for
project-wide adoption.** It is a severe regression on the synthetic
benchmark (event_recall 1.00 -> 0.23, seizure_recall 1.00 -> 0.00, AP
0.995 -> 0.068, ~5x writes, all 10 seeds) and a mixed, not clearly
beneficial result on CMAPSS and MetroPT-3 (some metrics better, others
worse, no consistent direction). Full tables in
`research-skab-validation.md`'s "Project-wide revalidation" section. A
follow-up persistence/alpha sweep (`results/skab_persistence_sweep.json`)
found no configuration that detects more events without a proportional
false-alarm cost; `alpha` cannot be tightened to compensate because
SKAB's 657-window commissioning sample is already near the minimum
`alpha=0.003` needs. The valve-throttling and isolated-spike families
remain unfixed by any change tested so far.
