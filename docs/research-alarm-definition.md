# What causes a uEngram alarm: definition, corrections and verified results (v7)

This study answers three questions:

- What is an alarm, and which conditions must all be met for one to be raised and cleared?
- How do uEngram's steps implement each condition?
- What happens on MetroPT when the definition is applied correctly?

It builds on the [alarm tradeoff](research-alarm-tradeoff.md) (v4),
[two-condition](research-two-condition-alarm.md) (v5) and
[error-gated](research-error-gated-alarm.md) (v6) studies.

Code:
- Definition: `bench/alarm_definition.py`
- Sweep, selection, replay, relations and event trace: `bench/alarm_definition_sweep.py`
- Corroboration hook: `src/uengram/alarm.py` (`evidence`, `exceeded`)
- Tests: `tests/test_alarm_definition.py`
- Results: `results/v7_alarm_definition_*`

Scores are the frozen seed-0 DSP uEngram scores; nothing is retrained or rescored.

## 1. What an alarm is (sources)

| source | what it requires |
|---|---|
| ANSI/ISA-18.2-2016 §3.1.7, IEC 62682:2022 | An alarm is an "audible and/or visible means of indicating to the operator an equipment malfunction, process deviation, or abnormal condition **requiring a timely response**". |
| ISA-18.2 §3.1.26 | The alarm setpoint is the "threshold value of a process variable or discrete state that is used to determine if the alarm is active". |
| ISA-18.2 §3.1.20, §3.1.19, §3.1.11 | An on-delay requires the condition to persist before the alarm activates. An off-delay and a deadband govern the return to normal. Acknowledging an alarm does not clear it. |
| ISA-18.2 (bad-quality and state-based alarming) | A bad or missing measurement is handled explicitly, not taken as normal. Setpoints apply to the current operating state. |
| ISA-18.2 §9.3 (rationalization), ISO 17359 / 13374, P–F interval | Every alarm has a consequence, an action and an allowable response time. A predictive alarm is useful only if it comes early enough to act before functional failure. |
| IEC 61508/61511 (MooN voting) | Requiring M of N conditions lowers spurious activations and raises missed or late ones. The binomial arithmetic holds only for independent channels, so correlated sensors are not separate votes. |
| Adnan, Izadi & Chen, J. Process Control 21 (2011); NIST/SEMATECH SPC | Delay timers trade false-alarm rate against detection delay. For independent windows with false-exceedance probability `p`, the mean wait until `k` false exceedances in a row is `(1-p^k)/((1-p)p^k)` windows. Autocorrelated scores break this. |
| Jakobs, Veloso & Gama, arXiv:2502.07394 (2025), on MetroPT2 | "The LPS signal is a failure warning already built into the train conductor control panel. The goal ... is to find the failures **two hours before the LPS signal turns on**." |

Jakobs et al. raise their alarm in three steps:
1. **Error:** a 30-min slice is anomalous when its reconstruction error exceeds `3·q99` of the
   training errors.
2. **Persistence:** the binary series is low-pass filtered, `z_t = z_{t-1} + α(y_t - z_{t-1})`,
   with `α = 0.15`.
3. **Alarm:** a failure is declared when `z_t > 0.5`.

Step 2 is an exact on/off-delay:
- From `z = 0`, it takes 5 anomalous slices in a row to raise the alarm
  (`1 - 0.85^5 = 0.556`; 4 slices give 0.478).
- From `z = 1`, it takes 5 normal slices to clear it (`0.85^5 = 0.444`).

So the published MetroPT practice and ISA-18.2 describe the same structure:

**threshold crossing → persistence → alarm → delayed clearance**

Sources:
- Jakobs et al.: https://arxiv.org/html/2502.07394v1 (the quotes and Eq. 1 were extracted from
  this text).
- MetroPT dataset: https://www.nature.com/articles/s41597-022-01877-3 and
  https://archive.ics.uci.edu/dataset/791/metropt+3+dataset
- SPC: https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc3.htm
- Alarm KPIs: https://www.emerson.com/is/content/emerson/en/corporate/news-and-events/articles/documents/alarm-management-by-the-numbers.pdf
- IEC functional safety: https://www.iec.ch/functionalsafety

Standards texts are paraphrased except for the short definitions quoted.

## 2. The definition used here

An alarm on test window `t` is raised when **all** of these hold. Each condition maps to one
uEngram step.

| # | condition | uEngram implementation |
|---|---|---|
| 1 | **Valid input**: the window is monitorable and scored. Invalid windows neither raise nor clear. | `bench.alarm_eval.eligible`; `AlarmPolicy.step` returns early |
| 2 | **uEngram error**: novelty or health crosses its setpoint for the current state. | Window → DSP features → 16-d whitened `z` (fit on commissioning only). Novelty: min Mahalanobis² to the memory's normal slots, computed before the window is absorbed. Health: min Mahalanobis² to the reference frozen at the end of commissioning. Setpoint: split-conformal quantile `k = ceil((1-α)(m+1))` of the calibration-week scores per state (GLOBAL fallback). The new-state veto and the health-trust rule decide whether the crossing counts. Recorded per window as `AlarmPolicy.exceeded`. |
| 3 | **Corroboration**: at least `m` independent errors (threshold crossings) were seen in the last `L` h. Each source is TRUE, FALSE or UNKNOWN. The vote is TRUE if #true ≥ m, FALSE if #true + #unknown < m, otherwise UNKNOWN. | `bench.alarm_definition.vote` |
| 4 | **Persistence (on-delay)**: a corroborated error raises the counter (cap `k`). A FALSE vote or no error lowers it. An UNKNOWN vote holds it. The alarm opens when a counter reaches `k`. | `AlarmPolicy.step(..., evidence=)` |
| 5 | **Release (off-delay)**: the alarm clears when both counters are back at 0. That takes `k` normal windows after a saturating deviation. | `AlarmPolicy.step` |
| 6 | **Actionability** (evaluated, not triggered): hours before the report, and hours before LPS, with positive meaning early. | `actionability`, `event_trace` |

**Error sources**, one vote per independent physical cause:
- **Dataset:** the MetroPT documentation's own threshold crossings.
  - `low_pressure`: LPS active, or TP3 below 7 bar. Both measure the same documented 7 bar
    condition, so they form one vote.
  - `oil_level`: the `Oil_level` flag departs from its calibration value.
- **Calibrated** (comparison): the 7 v5 per-channel crossings. A channel is UNKNOWN in a state
  that has no reference.

When every vote is TRUE, the definition is the actual alarm (`AlarmPolicy`) exactly.

## 3. The uEngram steps, checked

An independent audit reimplemented each step without the repository's evaluation code:

| step | check | result |
|---|---|---|
| Conformal setpoints | Own quantiles at `α=0.003`, states 0, 1 and GLOBAL, versus `AlarmPolicy.thresholds` | exact (for example, state 0 novelty 2372.1701, `m=572`, `k=572`) |
| Persistence counters | Own loop versus `AlarmPolicy.run`, three configs × 21,556 windows | 0 mismatches |
| Metrics | Own events, false alarms/week, clean share, first-alarm offset versus `apply_policy` | exact |
| v6 gated metrics | Own gate versus published v6 selections | exact, once the audit script's denominator was corrected to eligible clean time (it had used all clean windows) |
| Causality | Encoder and reference fit on commissioning only; novelty computed before the window is absorbed; online pass in time order | no leakage |
| Data | 25,126 windows (fit 2,658, calibration 911, test 21,556); window length 583–722 s, median 585 s; no window spans a logging gap; 495 eligible event and 19,399 eligible clean windows | consistent |
| v7 (this study) | Own implementation of section 2 versus the published v7 rows and trace | section 9 |

A repository test also rebuilds the v7 alarm from the recorded exceedances and votes alone,
with one counter that rises on TRUE+error, falls on FALSE, and holds on UNKNOWN. It does so on
real MetroPT3 data, including votes with unknowns.

## 4. Corrections

1. **Unknown is not normal (three-valued votes).**
   - In v5, a window with no loaded samples ended a hold run.
   - In v6, the calibrated errors read FALSE in sustained load, which has no reference. That
     produced the "recency cap" that met the budget.
   - Under ISA bad-quality handling, such windows must hold the alarm state. With that
     correction, the v6 4/4-at-1.06% result disappears: config 42 stays at 4.62% (section 5).
2. **Persistence applies to the whole alarm condition.** v6 latched errors onto an alarm that
   had already persisted on uEngram alone. Here the counter rises only on windows where the
   uEngram error is corroborated. That is the on-delay of the combined condition: "the
   conditions hold for long enough".
3. **Independent votes.** LPS and TP3 < 7 bar are one condition seen by two sensors, so they
   form one vote, not two.
4. **Early is positive.** `lead_hours` in `evaluate_alarms` is the first alarm minus the report
   start, so positive means late. v7 reports `first_alarm_minus_report_h` (same sign, honest
   name), `events_warned_before_report` and `hours_before_report` (positive = early), and
   events warned at least 2 h before LPS. An event counts as "warned before its report" when
   the alarm's first active window within `[start - 24 h, end]` ends before the report start,
   even if the alarm no longer stands when the report starts. Selection never uses this
   metric.
5. **`AlarmPolicy` documentation.** Clearing takes as many normal windows as the counters
   hold, which is `k` only after a saturating deviation. The docstring now says so.

## 5. Results on MetroPT3 (development)

The sweep covers 44 `AlarmConfig`s (42 calibratable): the actual frozen v2 alarm first, then
the v4 grids. Each is crossed with 30 corroboration rules:
- dataset: `m ∈ {1,2}`
- calibrated: `m ∈ {1,2,3}`
- both: `L ∈ {0,1,3,6,12,24}` h

Selection rules were fixed in code before any corroborated row existed.

| | dataset errors | calibrated crossings |
|---|---|---|
| rows | 504 | 756 |
| rows meeting the full budget | **0** | **0** |
| most events with clean ≤ 2% and false alarms ≤ 1/week | 2 | 3 |
| any 4/4 row | none: event-coverage ceiling 3 (`m=1`) and 0 (`m=2`) | 110 rows; min clean 1.62% at 1.80 fa/wk; min fa 0.63/wk at 4.62% |
| selection (fallback) | config 20, `m=1`, `L=6 h`: **2/4**, 0.05 fa/wk, 0.02% clean | config 20, `m=1`, `L=24 h`: **3/4**, 0.11 fa/wk, 0.08% clean |
| the same config uncorroborated | 3/4, 0.11 fa/wk, 0.08% | (same) |

- **Config 20** is the actual alarm with adaptive novelty level `aci_gamma=0.01`.
- **Event-coverage ceiling:** an event can be caught only if the corroboration vote is TRUE on
  at least one of its windows. With dataset errors that happens for at most 3 events at `m=1`
  and none at `m=2`.
- **The selected calibrated alarm equals its uncorroborated policy.** Its vote is TRUE or
  UNKNOWN on 99.6% of clean windows, so corroboration constrains nothing.
- **The two-valued dataset votes behave as the theory requires** (504 rows, 0 violations).
  Corroboration only removes alarm time and recall. Clean share rises with `L` and falls with
  `m`.
- **The three-valued calibrated votes can extend alarms.** UNKNOWN holds the counters, so 73
  of 756 rows have more clean alarm time than the policy, and clean share is not monotone in
  `L` or `m`.
- **Early warnings are rare and almost all concern Apr 18.** Within both budgets:
  - 17 dataset rows (configs 3, 4, 5: the actual alarm at `α` = 0.01, 0.03, 0.1) warn
    Apr 18 13.96 h before its report, with or without corroboration. The oil-level departure
    starts at −14.6 h.
  - 25 calibrated rows (configs 3, 4, 5, 41, 43) warn Apr 18 13.4–14.0 h early. One of them
    (config 5, `m=1`, `L=0`) also stands 5.8 h before Jul 15, but its alarms clear before
    either failure window, so it detects 0/4.

## 6. How each failure did or did not cause an alarm (MetroPT3, selected dataset alarm)

Times are hours from the report start (negative = before the report), over
`[start - 24 h, end]` (`results/v7_alarm_definition_trace.json`).

| event | report length | first uEngram error (cond. 2) | first dataset error (cond. 3) | first corroborated error | alarm opens (cond. 4) | uncorroborated policy opens |
|---|---|---|---|---|---|---|
| 0, Apr 18 | 24.0 h | +2.2 | −14.6 (oil level) | +2.2 | **+3.2** | +3.2 |
| 1, May 29–30 | 6.5 h | **−21.5** | none | none | missed | +3.0 |
| 2, Jun 5–7 | 52.5 h | **−8.4** | +34.0 (low pressure) | +34.9 | **+45.2** | +14.3 |
| 3, Jul 15 | 4.5 h | none | +3.2 (low pressure) | none | missed | missed |

The alarm is caused, or fails, as follows:

- **May 29–30 and Jun 5–7: uEngram crossed its setpoint before the report** (−21.5 h, −8.4 h),
  but never for 3 windows in a row before it. So condition 4 is the limit on early warning, not
  condition 2.
- **Corroboration by dataset errors costs recall and time.** May 29–30 has no dataset error at
  all, so it is lost. On Jun 5–7, low pressure appears 34 h into the report, which delays the
  alarm by 31 h.
- **Jul 15:** uEngram never crosses within the window. The v4 study traced this to the
  new-state veto and the unbaselined health channel. LPS fires, but corroboration cannot create
  a uEngram error.

## 7. MetroPT2 (consumed; replay and trace only)

Both selected alarms, and their uncorroborated policies, detect 0/3 (Jan) and 0/2 (Apr), with 0
false alarms.

- **Apr:** uEngram never crosses within `[start - 24 h, end]` of either event. Yet the
  `oil_level` error is present from −1.9 h and −1.8 h, which is 3.0 h and 59.4 h before the LPS
  warnings (published at +1.1 h and +57.6 h). The dataset errors meet the "2 h before LPS" goal
  on both events, but condition 2 never does.
- **Jan:** uEngram crosses 15.7 h and 17.3 h before events 1 and 2, but never 3 windows in a
  row. Event 0 has a low-pressure error at +1.1 h and no uEngram error.

## 8. Caveats

- MetroPT3 has 4 events. All results are single deterministic runs with no confidence
  intervals. Both MetroPT2 recordings were already consumed, so section 7 is diagnostic only.
- ISA-18.2 and IEC 62682 are copyrighted. Their definitions are quoted briefly from accessible
  excerpts, and recommended timer or deadband values are not attributed to them (the standard
  sets none universally).
- "Actionability" is evaluated here, not enforced. No configuration studied warns 2 h before
  LPS on any MetroPT2 event.
- This is a mechanism study, not a production recommendation.

## 9. Independent recomputation of the v7 numbers

A separate implementation was written from section 2 alone. It does not use `AlarmPolicy`,
`apply_policy`, `evaluate_alarms` or any v6/v7 harness code; script
`~/scripts/uengram_v7_independent.py`. It recomputed seven sweep rows: both selections, plus
configs 0, 22, 34 and 42 under both families. All four metrics matched exactly (max abs
difference 0). It also recomputed the four MetroPT3 trace rows of section 6, and every time
matched exactly.

## Reproduce

```bash
P="python -m bench.alarm_definition_sweep"
$P sweep --jobs 4 && $P select && $P replay && $P relations && $P trace
```
