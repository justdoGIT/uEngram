# Two-condition alarm: modalities in error AND holding long enough

Follow-up to [research-alarm-tradeoff.md](research-alarm-tradeoff.md). That study showed that
no `AlarmPolicy` setting on MetroPT3 jointly meets

```
events_detected == 4  AND  alarm_time_share_clean <= 0.02  AND  false_alarm_per_week <= 1
```

This study replaces the pooled novelty/health distance with a per-modality rule. It asks
whether requiring several modalities in error, held for a minimum duration, closes that gap.
It does not. The rest of this document shows exactly why.

Harness: `bench/modality_alarm.py` (rule), `bench/modality_sweep.py` (sweeps, selection,
replay, relations). Tests: `tests/test_modality_alarm.py`. Raw results:
`results/v5_modality_*`. Windows are 9.75 min, so `k_hold` windows equal `0.1625 * k_hold` h.

## 1. The rule

For test window `t` and modality `j`:

- **Physical channel error:** `e_j(t) = 1[z_j(t)^2 > tau_j]`, where
  `z_j = (x_j - median_j) / (IQR_j / 1.349)`. Reference windows come from `stream.cal` only, so
  they are label-free and precede the test split.
  - `conditioning="state"`: `x_j` is the window summary, judged against its operating-state
    reference.
  - `conditioning="loaded"`: `x_j` is the median over the window's loaded samples
    (`DV_eletric` active, at least 3 samples), judged against one pooled loaded reference.
  - A window without a value is never in error.
- **Threshold:** `tau_j` is the conformal `(1 - alpha/J)` quantile of the reference `z^2`
  under Bonferroni correction (`parametric`: the chi-square(1) quantile).
  - The conformal quantile needs `m >= J/alpha - 1` reference windows. Below that, the channel
    is reported **unjudgeable** (`tau = +inf`), not healthy.
- **Duty modality:** `e_duty(t) = 1[state(t) == sustained_load]`.
- **Condition A:** `A(t) = 1[sum_j e_j(t) >= min_modalities]`.
- **Condition B, open:** the alarm opens after `k_hold` consecutive eligible windows with `A`.
- **Release:** the alarm releases after `k_release` consecutive eligible windows without `A`.
- **Logging gaps:** a gap (unmonitorable window) resets the run count but does not release an
  open alarm.

The predeclared selection rule, fixed before any MetroPT2 computation, is the budget above.
Ties go to the smallest `alarm_time_share_clean`, then the earliest first alarm, then the
smallest `k_hold`. If no row qualifies, a fallback rule applies:

1. Keep rows with 4/4 detected and `false_alarm_per_week <= 1`.
2. Take the minimum clean share.
3. Break ties toward the simplest configuration.

The fallback result is replayed only to test transfer. In the tables below, the *offset* is
the first alarm minus the report start, averaged over detected events. A negative offset means
the alarm came before the report.

## 2. Outcome

| | value |
|---|---|
| MetroPT3 rows evaluated (OFAT + 3 interaction grids) | 159 |
| rows with 4/4 detected | 124 |
| rows with 4/4 and `false_alarm_per_week <= 1` | 36 |
| rows meeting the full budget | **0** |
| rule used | fallback |

Two rows sit on the 4/4 Pareto front between false alarms and clean time. Neither meets both
limits:

| config | detected | fa/wk | clean share | offset h |
|---|---|---|---|---|
| loaded, all 7 channels + duty, `m=1`, `k_hold=40` (**selected**) | 4 | 0.32 | 3.11% | +2.4 |
| loaded, 5 pneumatic + duty, `m=3`, `k_hold=1` | 4 | 4.44 | 2.00% | -2.0 |

The selected configuration was replayed unchanged on both MetroPT2 recordings:

| stream | detected | fa/wk | clean share | openings |
|---|---|---|---|---|
| metropt2_jan | **0/3** | 0.39 | 1.16% | 5 clean, all `dv_pressure_bar+reservoirs_min_bar` |
| metropt2_apr | **0/2** | 0.00 | 0.00% | none |

## 3. Exact relations

**R1. The duty-only alarm has a closed form.**

- Setup: duty modality only, `k_release=1`.
- A sustained-load run of `L` eligible windows is in alarm on its last `max(0, L - k + 1)`
  windows.
- A run separated from an active run only by a logging gap is active from its first window.
- Consequence: detection, clean share and offset are functions of the run-length lists alone.
- Check: `duty_only_prediction` equals the harness on 3 streams x 12 holds x 3 metrics. All
  108 checks match with an absolute difference of 0, recorded in
  `results/v5_modality_relations.json` and pinned by a parametrised test.

**R2. The duty floor.**

- Applies when `min_modalities=1`, duty is included and `k_release=1`.
- Condition A is then a superset of the duty condition, so the alarm is a superset of the
  duty-only alarm. Its clean share is at least the duty-only share at the same `k_hold`.
- Check: 106 recorded rows in this family, 0 below the floor.

**R3. Run lengths decide everything the duty floor allows.** Sustained-load runs on MetroPT3
(windows):

| | runs |
|---|---|
| event runs | 154, 41, 182, 34 |
| longest clean runs (153 total) | 309, 129, 124, 71, 70, 68, 32, 27 |

- The longest clean run (309) is longer than every event run. Five more (129 down to 68) are
  longer than the two short event runs (41, 34).
- Any hold long enough to drop those clean runs also drops events.
- Duty-only clean share by hold:

| `k_hold` | 1 | 6 | 30 | 34 | 35 | 50 | 78 |
|---|---|---|---|---|---|---|---|
| detected | 4 | 4 | 4 | 4 | 3 | 2 | 2 |
| clean share | 5.88% | 4.56% | 3.42% | 3.28% | 3.24% | 2.77% | 1.99% |

**R4. The selected channel set is infeasible in closed form.**

- With the 7 loaded channels, duty and `m=1`, 4/4 detection holds exactly for
  `k_hold <= 41` (6.7 h). R2 then gives a clean share of at least 3.05%.
- Clean share `<= 2%` first holds at `k_hold = 78` (12.7 h). There the clean share equals the
  duty-only 1.99%, but event 3 is lost (3/4).
- No hold in `[42, 77]` satisfies either requirement, so this family cannot meet the budget.

**R5. Physical channels add little to the selected alarm.**

- On MetroPT3, 1045 of its 1098 active windows have the duty modality in error.
- Physical channels contribute the one extra detection over duty-only `k=40` (event 3, 4/4
  versus 3/4) for +0.02 pp of clean time.

## 4. Sensitivity (MetroPT3, development only)

Baseline: `alpha=0.05`, state conditioning, all 7 channels + duty, `m=1`, `k_hold=6`. Each row
changes one field. "judgeable" counts channel x state cells with a finite threshold.

| change | detected | fa/wk | clean | offset h | judgeable |
|---|---|---|---|---|---|
| baseline | 4 | 17.25 | 21.39% | -13.3 | 21 |
| `k_hold` 1 / 30 / 40 / 50 | 4 / 4 / 4 / 3 | 35.29 / 2.12 / 1.53 / 1.11 | 58.2 / 9.9 / 8.2 / 6.5% | -22.0 / -10.1 / -8.7 / -2.0 | 21 |
| `k_release` 3 / 6 | 4 / 4 | 4.81 / 3.02 | 48.4 / 60.9% | -13.3 / -19.6 | 21 |
| `min_modalities` 2 / 3 | 0 / 0 | 0.05 / 0.00 | 0.22 / 0.00% | none | 21 |
| `alpha` 0.2 / 0.01 | 4 / 4 | 22.27 / 0.79 | 36.3 / 4.56% | -21.4 / +0.3 | 21 / 7 |
| `duty=False` | 1 | 15.93 | 15.45% | -17.9 | 21 |
| `correction=none` | 4 | 22.86 | 48.76% | -21.8 | 21 |
| `calibration=parametric` | 4 | 20.69 | 38.76% | -19.3 | 21 |
| `reference=fit+cal` | 4 | 11.27 | 12.95% | -9.4 | 21 |
| `conditioning=loaded` | 4 | 6.03 | 6.26% | -3.2 | 7 |
| pneumatic channels only | 4 | 2.06 | 6.74% | -4.5 | 15 |

Interactions found in the grids (`results/v5_modality_pairwise_*`):

- **`alpha` x `k_hold`.**
  - At `alpha=0.01`, all 12 rows match the closed-form duty-only alarm to within 1e-12 on
    detection, clean share and offset (k=6: 4.56%; k=30: 3.42%; k=40: 3/4, 3.09%).
  - Only 7 of 21 channel x state cells are judgeable there, because Bonferroni at 0.01/7 needs
    699 reference windows. The judgeable cells change none of the three metrics.
  - At larger `alpha`, physical channels add clean time at every hold. They add recall only
    where duty-only drops event 3 (`k_hold >= 35`).
- **`min_modalities` x `k_hold` (loaded, all channels).**
  - `m>=2` cuts false alarms sharply at short holds (k=6: 6.03 to 0.90 fa/wk).
  - Clean share converges on the duty floor once `k_hold >= 20`; `m` stops mattering there
    (k=30: 3.46 / 3.42 / 3.42%).
- **`min_modalities` x `k_hold` (loaded, pneumatic).**
  - `m=3` is the only route below 2% clean time: 1.43% at k=3, 0.76% at k=30.
  - It loses events from k=3 on (3, 2, 2, 1, 1 detected at k = 3, 6, 10, 20, 30).
  - At k=1 it detects 4/4 at exactly 2.00%, but with 4.44 fa/wk.

## 5. Failure modes

1. **Duty confound.**
   - The strongest MetroPT3 signal is "the compressor has been under sustained load for a
     long time". Every event has such a run, but so do many healthy periods.
   - The clean runs (up to 309 windows) are longer than the event runs (at most 182).
   - Recall therefore costs clean alarm time at a rate fixed by the run-length distribution
     (R3), not by any threshold.
2. **Calibration is not exchangeable with test.**
   - The selected configuration uses a nominal per-channel level of `0.05/7 = 0.71%`.
   - Share of clean eligible test windows in error: `oil_temp_c` 31.5%, `dv_pressure_bar`
     7.2%, `motor_current_a` 4.0%.
   - The conformal guarantee is marginal under exchangeability. Seasonal drift between the
     calibration period and test breaks it, most visibly for oil temperature.
3. **Unjudgeable collapse.**
   - With small `alpha`, Bonferroni and per-state references, most channels are unjudgeable.
   - The rule then silently becomes the duty alarm. The `judgeable` count is recorded per row
     so this is visible.
4. **Transfer to MetroPT2 fails through condition B, not condition A.**
   - The physical channels are in error on 214 of 217 judgeable event windows (Jan) and 233 of
     233 (Apr).
   - But MetroPT2 failures happen mostly while the compressor idles or cycles. Of the event
     windows, 138 of 355 (Jan) and 152 of 385 (Apr) have fewer than 3 loaded samples, so no
     physical channel is judged there.
   - Each such window ends the condition-A run. The longest in-event condition runs are 23, 1
     and 9 windows (Jan) and 22 and 11 (Apr), all below `k_hold=40`.
   - MetroPT2's longest clean sustained-load run is 37 windows, so the duty modality is silent
     there as well.
   - The hold length tuned on MetroPT3 describes MetroPT3's duty pattern, not the fault.
5. **Late rather than early.**
   - The selected configuration's first alarm averages 2.4 h *after* the report start.
   - Among the 36 rows with 4/4 and `false_alarm_per_week <= 1`, three alarm before the
     report on average. All are loaded/all-channel/`m=1` rows: k=25 (-0.1 h, 3.67%), k=20
     (-0.9 h, 3.97%) and k=15 (-1.7 h, 4.32%).
   - Earlier warning costs clean time on top of the duty floor.

## 6. Caveats

- One development stream with 4 events. Every count above is a point estimate from a single
  deterministic run, with no confidence intervals.
- R3 and R4 are exact for these recordings, not general laws.
- Both MetroPT2 recordings are now consumed. Any change motivated by the replay needs a fresh
  recording to evaluate. In particular, the hypothesis that unjudged windows should act as
  gaps rather than run resets (failure mode 4) is **untested** and must not be checked on
  MetroPT2.
- Event and clean definitions, eligibility and the false-alarm episode count are those of the
  `AlarmPolicy` harness (`bench/alarm_eval.py`). The budget is unchanged from the v4 study.
- This is a mechanism study, not a recommended production configuration.

## Reproduce

```bash
P="python -m bench.modality_sweep"
$P ofat
$P pairwise --pair k_hold min_modalities --fixed conditioning=loaded --tag _loaded_all
$P pairwise --pair alpha k_hold --tag _alpha_hold
$P pairwise --pair k_hold min_modalities --fixed conditioning=loaded \
  'channels=["tp2_bar","tp3_bar","h1_bar","dv_pressure_bar","reservoirs_min_bar"]' \
  --tag _loaded_pneumatic
$P select && $P replay && $P relations
```

Rerunning the full sequence with the committed code reproduces every `results/v5_modality_*`
file byte for byte.
