# Step-inflation minimum support, health+age RUL, and CNN configurations on CMAPSS

**Status:** development evidence on consumed benchmarks. One deterministic seed for DSP encoders
(the encoder and the CMAPSS stream have no randomness); two seeds for the CNN sweep.

## 1. Step-inflation minimum support

The step detector divides its innovations by `sqrt(max(inflation, 1))`, where `inflation` is the EWMA
variance inflation measured on the encoder's training latents (`stats.ewma_inflation`). The strict
estimator cuts each unit at every abrupt jump. On a steadily degrading CMAPSS unit the drift trips the
jump detector, each unit fragments into short segments, and few warm lag pairs (segments longer than
the EWMA warm-up plus the lag, 19 + 38 samples at `lambda = 0.1`) remain. The estimate is then low
and noisy (median 0.87-2.75 where the unit-grouped estimate is 8.6-9.0), the clamp at 1 turns it into
"assume white noise", and the detector fires on slow within-unit motion: slots saturate and
end-of-life windows are evicted.

`noise_fallback.soft_ewma_inflation` now requires `MIN_INFLATION_PAIRS = 100` warm lag pairs (a
variance estimate from n pairs has relative standard error about sqrt(2/n), 14% at 100). Below that,
and only when the caller's own groups (here the unit ids) give more pairs, it uses them
(`step_inflation_quality = "group_segments"`, with a `RuntimeWarning`). `min_pairs=0` restores the
strict behaviour. The threshold comes from that error argument, not from tuning on these results.

`python -m bench.inflation_ab` writes `results/inflation_min_support_ab.json` (DSP, seed 0, uEngram).
Cases that changed (`strict -> rule`):

| subset / encoder | end-of-life retention | novelty AUROC | health AUROC | write reduction % |
|---|---|---|---|---|
| FD001 `dsp_d24` | 0.16 -> 1.00 | 0.520 -> 0.649 | 0.9995 -> 0.9979 | 80.3 -> 96.2 |
| FD003 `dsp` | 0.12 -> 1.00 | 0.581 -> 0.774 | 0.982 -> 0.987 | 80.0 -> 95.6 |
| FD002 `dsp`, condition norm | 0.06 -> 1.00 | 0.573 -> 0.733 | 0.986 -> 0.994 | 78.5 -> 96.1 |

All other CMAPSS cases (FD001 `dsp`/`dsp_slope`, FD003 `dsp_slope`, and every case that already used
the unit groups) are unchanged. On the synthetic stream (DSP and context encoders, seeds 0-4, both
arms) every metric is identical; at `d = 36` it is identical too, so the `d = 36` regression in
`research-encoder-capacity.md` is not an inflation problem. The CNN configurations tested below never
fall under the threshold, and forcing unit-group inflation on the CNN changed nothing material.

`FD001 dsp_d24` health AUROC falls by 0.0016; every other health AUROC rises or is unchanged.

## 2. RUL from health and unit age

`HealthRUL` maps the health score to RUL alone. Health saturates, so the same score occurs early and
late in life. `HealthAgeRUL` fits a quadratic ridge regression on (log health, age = cycles observed)
over the labelled stream windows (24 bytes of coefficients, predictions clipped to [0, 125]).
`bench.run_cmapss` now records `rul_rmse_health_age` next to the unchanged `rul_rmse_health`.
Every one of the 14 DSP cases above improves, e.g. FD003 `dsp` 29.8 -> 25.6, FD001 `dsp` 25.5 -> 23.5,
FD002 `dsp` 53.0 -> 39.6, FD004 `dsp` 53.9 -> 48.1. It remains far from the published 11-16 of trained
RUL regressors. Smoothing the health score over the last 5-40 windows was tried and did not help
(FD001 25.5 -> 25.5 at 5, 27.5 at 10, 32.9 at 40).

## 3. CNN configurations: no gain

`bench.cnn_cmapss_sweep` (`results/cnn_cmapss_sweep.jsonl`) runs CNN variants on FD003 and FD004 (the
subsets where the CNN fills all 256 slots and loses end-of-life windows), seeds 0-1, mean of two:

| config | FD003 retention | FD003 novelty AUROC | FD004 retention | FD004 novelty AUROC |
|---|---|---|---|---|
| d16 e30 avg (default) | 0.80 | 0.623 | 0.62 | 0.742 |
| d16 e30 maxavg | 0.92 | 0.644 | 0.63 | 0.745 |
| d16 e90 avg | 0.72 | 0.625 | 0.74 | 0.783 |
| d16 e90 maxavg, ortho 0.05 | 0.91 | 0.611 | 0.58 | 0.778 |
| d24 e30 avg | 0.64 | 0.552 | 0.66 | 0.731 |
| d32 e30 avg | 0.51 | 0.554 | 0.72 | 0.746 |

No configuration improves both subsets: `maxavg` helps FD003 only, 90 epochs helps FD004 only, and a
wider latent hurts. With two seeds the differences are within seed-to-seed spread, consistent with
`research-pending-model-improvements.md`. No CNN default was changed.

## 4. Regenerated results and a trade-off

`cmapss_fd003_runs.jsonl`, `cmapss_fd002_norm_runs.jsonl`, `cmapss_d24_runs.jsonl` and
`feature_group_ablation.json` were regenerated at HEAD with the rule. The non-uEngram rows (naive,
FIFO, reservoir, label-free centroid) are identical to the earlier files, so only uEngram moved. The
earlier d=24 file (retention 0.44) also predated the admission changes. Newly runnable cases, which
raised "fewer than 10 difference pairs" before `noise_fallback`, are in `cmapss_d32_runs.jsonl`,
`cmapss_fd002_raw_runs.jsonl` and `cmapss_fd004_norm_runs.jsonl`. `cmapss_runs.jsonl` (FD001, `dsp` and
`cnn`) and `cmapss_slope_runs.jsonl` were regenerated too: DSP rows are identical apart from the new
health-and-age columns, the CNN uEngram rows differ slightly (largest relative change 0.30, on the small
`mmd2_history`), and `uengram_frozen` on `dsp_slope` moved from retention 1.0 to 0.6 because the earlier
file predates the admission changes. `cmapss_d84_runs.jsonl` was committed empty; it now holds the
full-dimension run (health AUROC 0.989, retention 1.00).

The trade-off: once the memory keeps end-of-life states in few items, each item spans the whole RUL
range and the memory-lookup RUL collapses to a near-constant prediction, as it already did on FD001.
FD003 lookup RMSE goes 24.1 -> 41.9, FD002 (normalized) 38.6 -> 53.4 and FD001 d=24 21.1 -> 42.7.
Health-based RUL improves (29.8 and 36.8 for health alone, 25.6 and 34.5 with age). On FD001 d=24 the
frozen-reference variant's health AUROC falls from 0.9995 to 0.9874 and its health RUL from 26.4 to 31.2.

In the feature-group ablation the sign of the FD003 log-std effect reverses: with the rule, dropping
the log standard deviation lowers FD003 retention from 1.00 to 0.66 (it "raised" it from 0.12 before).
Cells that earlier failed now run: without band energies the synthetic stream fits at d=12 (recall
0.79, AP 0.476), and without the mean FD003 gives health AUROC 0.859.

## 5. Gate scale at wide d (`DSPEncoder(gate_lag=...)`, off by default)

At d=36 the synthetic uEngram fills all 256 slots (recall 0.801, AP 0.534). On seed-0 training latents
(diagnostic not committed), the lag-100 difference noise exceeds the lag-1 noise by more than 2x in 26
of 36 dimensions, against 4 of 16 at d=16, so lag-1 whitening treats slow within-state variation as
novelty. `gate_lag=20` widens each dimension's gate scale `sigma0` to its lag-20 difference noise when
larger. `bench.run_synth --encoders dsp_d36_g20` (10 seeds, `results/synth_d36_gate20_runs.jsonl`;
naive and reservoir rows identical to `synth_d36_runs.jsonl`):

| d=36 | event recall | AP (mean / median) | write reduction % | live items |
|---|---|---|---|---|
| default | 0.801 | 0.534 / 0.541 | 71.6 | 256 |
| `gate_lag=20` | 0.999 (0.989-1.000) | 0.591 / 0.605 | 74.7 | 194 |

It restores retention but AP stays far below d=16 (0.996). A lag of 100 on three d=16 seeds lowered
AP (0.966-0.991) and a lag of 20 did not. Exploratory seed-0 checks (not committed, so not a result)
at lag 20: d=16 synthetic, FD001, FD001 d=24 and FD003 unchanged within 0.01; FD002 (normalized)
novelty AUROC 0.733 -> 0.710; SKAB d=16 5/34 events unchanged, SKAB d=48 9 -> 7 events. The default
stays 0, so no published table changes.
