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
