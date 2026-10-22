# Improvement analysis (diagnostics before the v2 changes)

This is the analysis that motivated the v2 engine changes. It was produced after the first full benchmark (`docs/RESULTS.md` at commit `b5a47f3`). The numbers come from throwaway diagnostic scripts that were kept outside the repository (`~/scripts/uengram/`, outputs in `~/.local/state/uengram/diag_*`), over 3 seeds per encoder (1 seed for the CNN encoder). Background research is in `docs/research-*.md`:

- `research-state-math.md`: hybrid-system models, change detection, Kalman/EWMA gains, few-sample predictive distributions, robust updates.
- `research-ad-eval.md`: metrics under class imbalance, conformal calibration, masking, streaming micro-clusters, drift, alarm hysteresis.
- `research-features.md`: order tracking, impulsive-fault indicators, physics residuals, condition normalisation, learned encoders, temporal context.
- `research-cmapss.md`: CMAPSS preprocessing, published FD001 RMSE/score values, health indices, evaluation protocol.

## What already worked

- Rare-event retention under a hard budget: every anomaly event and every seizure kept in about 31 slots (DSP encoder); FIFO and reservoir keep at most 4.7% of events even at 512 slots.
- Recurring faults group into one episodic slot per type (n = 24 to 42 per type); "which known fault is this" recall is 0.98 top-1.
- Supersession works when steps are large relative to the gate: in CNN latent space maintenance moves the state about 100 sigma0 and old states are superseded cleanly.

## Issues found, with evidence

### 1. Masking: the novelty score compared against stored anomalies

The prequential score was the minimum Mahalanobis distance over *all* gateable slots, including episodic slots that hold earlier anomalies. A recurring knock lands next to the stored knock and looks familiar.

| score (DSP, 3 seeds) | AUROC | AP | events caught at 1% false alarms |
|---|---|---|---|
| min distance over all slots (v1) | 0.635 | 0.100 | 0.21 |
| min distance over semantic slots only | 1.000 | 0.996 | 1.00 |

The same effect separates the point-store baselines: naive kNN stores every past anomaly (AUROC 0.935, AP 0.42); reservoir rarely does (0.985, 0.85).

### 2. Regime transitions look like faults

Coolant temperature follows a first-order lag after a regime switch. On DSP features the false-flag rate is 0.353 on nominal windows within 4 steps of a regime change and 0.008 on steady windows. The only features that move at transitions are the coolant log-std and lowest band (3.0 to 6.3 robust SDs; nothing else above 1.1). With the CNN encoder 84% of transition windows flag; they flood the 64-slot episodic budget and 81% of real anomaly events are evicted, which explains the CNN `event_recall` of 0.31.

A physics residual fixes this only when it is fitted correctly:

| thermal-lag fit | result |
|---|---|
| single window, ordinary least squares | pulled by anomaly windows (seizure rpm is a high-leverage point) |
| single window, robust trimming | discards transitions as outliers, learns "temperature never changes" |
| single window, structured T_eq(rpm) + lag | confounded by wear (a = 0.019, truth 0.5) |
| two consecutive windows: ramp_t = (1-a) ramp_{t-1} + a [T_eq(rpm_t) - T_eq(rpm_{t-1})] | a = 0.523, T_eq steps 0.234 / 0.141 (truth 0.25 / 0.15); transition windows as close to their state as steady ones (2% vs 46% beyond the gate) |

### 3. Normal regimes trapped as episodic

Promotion was only possible within 8 steps of a slot's creation. If a regime's first visit was cut short, every later visit was absorbed into that episodic slot and flagged. In one seed (context encoder) normal windows sat in episodic slots of up to 2,088 members and 36% of all windows were flagged.

### 4. Gates too wide in 16 dimensions; maintenance absorbed

The chi-square gate radius is 6.8 sigma0 while pure noise reaches about 4 sigma0 in 16 dimensions. Regimes are 6.4 to 11 sigma0 apart and slot variance inflated to 4.2x sigma0^2, so single slots held two or three regimes. The maintenance step (4.6 to 10 sigma0) was absorbed gradually rather than recognised; with `beta_min = 0.01` a slot chases a step over about 100 windows. A single-window gate cannot see a step smaller than the gate. Capping slot variance or Huber-weighted updates did not help (AP unchanged, false flags 0.077 and 0.053 vs 0.046).

### 5. CMAPSS: novelty is not health; spectral features carry noise

| CMAPSS FD001 (stream units 51 to 100) | end-of-life AUROC | RUL RMSE |
|---|---|---|
| uEngram v1, DSP features | 0.745 | 42.7 (equals predicting the stream-mean RUL) |
| naive kNN novelty, window mean/slope features | 0.552 (earlier failures are already in memory) | 29.2 (kNN) |
| distance from a healthy reference, window means | 0.9999 | n/a |
| ridge health index on mean/slope (units 1 to 50) | 1.000 | 18.5 (17.2 vs truth capped at 125) |
| published FD001 results (Lai et al. 2022 table) | n/a | 11.4 to 16.1 |

### 6. Accuracy is the wrong yardstick

At 2 to 3% anomaly prevalence a never-alarm system scores 97 to 98% accuracy. Targets are expressed as event recall at a fixed false-alarm budget, average precision, AUROC and detection delay.

## How a running machine maps to mathematics

| kind of state change | model | update rule | detection statistic |
|---|---|---|---|
| steady regime | Gaussian state per mode | exact average 1/n until mature | distance to normal states only |
| transition ramp | first-order lag x_t = x_{t-1} + a (u_t - x_{t-1}) | predict the ramp from the previous window | residual |
| slow wear | local linear trend Kalman filter | gain K = (sqrt(q^2 + 4q) - q) / 2, q = process / measurement noise | wear rate |
| step change (maintenance, new regime) | change point | sequential test on slot innovations, then new state and supersede | CUSUM / MEWMA statistic |
| transient fault | nominal dynamics plus fault | never learn from it | calibrated p-value |
| progressive degradation | health index h_{t+1} = h_t + g(h_t) + eta | Bayesian or regression update | first passage of h to failure level |

Measured wear drift on the synthetic stream is 5e-4 to 1e-3 sigma0 per window (K about 7e-4 if treated as a random walk).

## Prioritised changes

1. Score against normal (semantic) slots only; keep episodic slots for naming known faults.
2. Promote any candidate or episodic slot that gets M hits within any N windows.
3. Physics / temporal-context features for slowly responding channels, with anomalous windows excluded from the context.
4. Separate "absorb into a slot" from "raise an alarm"; calibrate the alarm threshold conformally on a normal commissioning period.
5. Sequential step detector per semantic slot; on detection create a new state and supersede the old one.
6. Mode identification from measured operating inputs, health modelled per mode.
7. Student-t predictive gate for young slots.
8. Episodic budget protection based on pattern rarity.
9. Level-plus-rate Kalman tracking of wear.
10. CMAPSS: window mean/slope features, healthy-baseline distance, supervised health index, 30-cycle windows, PHM08 score.
11. Evaluation on real labelled machinery data: SKAB (https://github.com/waico/skab), MIMII (https://zenodo.org/records/3384388), CWRU bearings (https://engineering.case.edu/bearingdatacenter).

Items 1 to 5 are implemented in v2; see `docs/EXPLAINED.md` and `docs/RESULTS.md`.

## Follow-up findings during the v2 implementation

Measured with the same throwaway scripts (seed 0 unless stated).

1. **Promotion at any time re-created masking.** Allowing an episodic slot to be promoted after any 5 hits in 8 windows promoted a pure knock slot (two knock events back to back) into a normal state; knocks were then familiar again and DSP AUROC fell to 0.917. Episodic slots are now promoted only after a sustained dwell (20 hits in 30 windows); synthetic regimes dwell about 33 windows, fault events at most 3. AUROC returned to 1.000.
2. **Correlated sensors (user question).** Noise correlation left in the PCA-whitened space averaged 0.05 (DSP), 0.04 (context) and 0.22 (CNN), with 14.1, 14.6 and 7.4 effective independent dimensions of 16. Inside one regime the per-dimension gate rejected 23 to 39% of normal windows in CNN space versus 0% with a full covariance, i.e. one fluctuation was counted several times. After whitening with a robust full noise covariance, the per-dimension gate rejects at most 0.08% (CNN, 200-window state blocks) versus 0% for a full per-slot covariance, so slots stay diagonal. On CMAPSS FD001 the 14 sensors carry about 2 independent signals (82% and 16% of standardised variance).
3. **CNN seizure retention.** With the alarm score (distance to normal states) used as the episodic importance, CNN transition candidates reached the 300-decade surprisal cap as often as seizures (37% of new candidates at the cap), and the older seizures were evicted: seizure recall 0, event recall 0.118. Ranking episodic slots by rarity (distance to the nearest stored slot of any kind) restored seizure recall 1.0 and event recall 0.299 (v1: 0.314).
