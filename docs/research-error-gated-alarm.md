# Error-gated alarm: the AlarmPolicy alarm AND at least m threshold-crossing errors

> The calibrated-gate result below (4/4 at 1.06%) relies on treating unjudged windows as
> normal. [research-alarm-definition.md](research-alarm-definition.md) (v7) corrects that with
> three-valued evidence, and the result does not survive.

This study keeps the existing alarm measure unchanged and adds one requirement: at least `m`
error conditions, each a threshold crossing, must be present before the alarm stands.
The earlier [two-condition study](research-two-condition-alarm.md) replaced the alarm measure
instead. The budget and the evaluation are those of
[research-alarm-tradeoff.md](research-alarm-tradeoff.md):

```
events_detected == 4  AND  alarm_time_share_clean <= 0.02  AND  false_alarm_per_week <= 1
```

Code:
- Harness: `bench/error_gated_alarm.py`
- Sweep, selection, replay and relations: `bench/error_gated_sweep.py`
- Tests: `tests/test_error_gated_alarm.py`
- Results: `results/v6_error_gated_*`

All scores are the frozen seed-0 DSP uEngram scores; nothing is retrained or rescored.

## 1. Definition

- **P(t)** is the `AlarmPolicy` output (`src/uengram/alarm.py`): conformal novelty/health
  thresholds, persistence counters, new-state veto and gating, exactly as before.
- **Combined alarm:** `A(t) = P(t) AND C(t)`, with `C(t) = 1[sum_j e~_j(t) >= m]`.
- **Lookback:** `e~_j(t) = 1` when error `j` occurred in an eligible test window `s <= t` with
  `end(t) - end(s) <= L` hours. `L = 0` means the same window.
- **Openings:** an episode opens where `A` rises from the previous eligible window. A logging
  gap neither opens nor closes it.
- **Metrics:** the same eligibility, event/clean definitions and metrics as
  `bench.alarm_eval.evaluate_alarms`. `C = 1` everywhere reproduces `apply_policy` exactly (R1).

The **actual alarm definition** is the frozen v2 `AlarmConfig`: `alpha=0.003`, `k_novelty=3`,
`k_health=0`, `per_state`, `new_state_alert`, `health_needs_baseline`. The sweep starts there
and also covers every `AlarmConfig` the v4 study explored: 44 configurations, 42 calibratable.

## 2. The errors

**Dataset family.** These are errors the MetroPT documentation defines as threshold crossings
(UCI MetroPT-3 variable list; MetroPT2 logs the same APU signals). Rates are shares of
MetroPT3 eligible test windows.

| error | dataset definition | rule used | clean | event |
|---|---|---|---|---|
| `lps` | "activates when the pressure drops below 7 bars" | any sample unlike LPS's calibration value (0) | 0.19% | 2.2% |
| `oil_level` | "active when the oil is below the expected values" | any sample unlike its calibration value | 14.4% | 0.2% |
| `tp3_below_7bar` | same 7 bar threshold, on analog TP3 | window minimum < 7 bar | 0.18% | 1.2% |

- `Oil_level` is logged as 1 throughout MetroPT3 calibration. The documented polarity would
  call every healthy window "low oil", so the healthy value is taken from calibration. It is
  constant 0 on MetroPT2-Jan, so it never errs there.
- Excluded, because the documentation describes them as normal control or operating signals:
  - `MPG` (pressure below 8.2 bar starts loading)
  - `Pressure_switch` and `Towers` (air-dryer tower discharge)
  - `COMP` and `DV_eletric` (load state)

**Calibrated family**, for comparison. These are the v5 per-channel crossings on 7 physical
channels: robust `z^2` above a conformal per-state threshold at `alpha=0.05`, Bonferroni, duty
modality off.

**Gates swept:** family x `m in {1,2,3}` x `L in {0,1,3,6,12,24}` h. That makes 36 gates per
`AlarmConfig` and 1512 gated rows.

## 3. Outcome

The predeclared rule applies the budget to gated rows only. If no row qualifies, the fallback
keeps both alarm budgets and takes the most events. Both rules were fixed in code before any
gated row was computed. The rule was applied separately per family.

| | alarm config | gate | detected | fa/wk | clean | first alarm vs report |
|---|---|---|---|---|---|---|
| actual alarm, ungated | v2 frozen | — | 3 | 0.74 | 0.60% | +6.8 h |
| **dataset (fallback)** | v2 + `health_needs_baseline=False`, `k_health=10` | m=1, L=1 h | **3** | 0.21 | 1.97% | +12.8 h |
| the same, ungated | | — | 4 | 0.95 | 5.08% | +0.8 h |
| **calibrated (primary)** | v2 + `health_needs_baseline=False`, `k_health=6`, `k_novelty=6` | m=1, L=6 h | **4** | 0.95 | 1.06% | +1.0 h |
| the same, ungated | | — | 4 | 0.63 | 4.62% | +1.0 h |

- **Dataset errors.** No row meets the budget (0 of 756). The best keeps both alarm budgets at
  3/4 detected.
- **Calibrated crossings.** Two rows meet the budget, both with the same alarm config: L=6 h
  (1.06%) and L=12 h (1.98%). Section 5 shows that this mostly comes from operating-state
  recency, not from the errors.

Both selections were replayed unchanged on MetroPT2, together with their ungated policies:

| stream | dataset gated | calibrated gated | both ungated policies |
|---|---|---|---|
| metropt2_jan | 0/3, 0 fa/wk | 0/3, 0 fa/wk | 0/3, 0 fa/wk, 0% clean |
| metropt2_apr | 0/2, 0 fa/wk | 0/2, 0 fa/wk | 0/2, 0 fa/wk, 0% clean |

## 4. Exact relations (all 1512 MetroPT3 gated rows; `results/v6_error_gated_relations.json`)

- **R1. Identity.** `C = 1` reproduces the actual alarm's metrics exactly (pinned by a test).
- **R2. The gate only removes alarm time.**
  - `A` is a subset of `P`, so clean share is at or below the ungated policy's, and detected
    events are at or below both the policy's and the error ceiling. Violations: 0, 0 and 0.
  - The false-alarm rate is *not* monotone. In 150 rows it rises, because a flickering error
    splits one policy episode into several openings.
- **R3. Nesting.**
  - A longer `L` gives a superset condition, so clean share is nondecreasing in `L`.
  - A larger `m` gives a subset, so clean share is nonincreasing in `m`.
  - Violations: 0 and 0.
- **R4. Error ceiling.** An event can be detected only if `C = 1` on one of its windows. For
  the dataset family on MetroPT3, that ceiling is:
  - `m=1`: 3 events, at every `L <= 24` h
  - `m=2`: 2 events
  - `m=3`: 0 events

  The May 29–30 failure (event 1) has no dataset-defined error from 24 h before its start to
  its end (pinned by a test). **4/4 is unreachable with dataset errors for any `AlarmConfig`
  and any `L <= 24` h.**

## 5. Mechanisms

**The dataset gate is selective but late, and depends on one oil-level episode.**

Of the fallback's 320 clean alarm windows:
- 312 have `oil_level` latched, and 303 of those fall on 2020-06-22..25. That is the single
  309-window `Oil_level` departure.
- That episode is exactly the longest healthy sustained-load run: both start 2020-06-22 15:01
  and last 309 windows ([two-condition study](research-two-condition-alarm.md), R3). On
  MetroPT3, the oil-level flag departs under prolonged load.

Detections:

| event | caught by | first alarm vs report start |
|---|---|---|
| 0 | one `oil_level` window | +1.1 h |
| 1 | not detectable (R4) | — |
| 2 | `lps` / `tp3_below_7bar` | +34.0 h, when LPS first fires 34 h into the 52.5 h report |
| 3 | `lps` | +3.2 h |

An `LPS`-gated alarm cannot precede the train's own low-pressure warning. It confirms a
pressure loss; it does not anticipate one.

**The calibrated gate is mostly a sustained-load recency cap.**
- Calibration has no own baseline for sustained load (state 2), and unbaselined states are not
  judged. So calibrated errors never fire in state 2: 0 of 1832 state-2 test windows, against
  4598 of 9219 (state 0) and 6417 of 10505 (state 1).
- With lookback `L`, the gate therefore reads "a state-0/1 window occurred within L hours".
  The selected alarm config trusts health in unbaselined states, so it stands through long
  sustained-load runs. The gate cuts those runs to their first `L` hours.
- Of the policy's 838 clean alarm windows, the gate removes 636, and 633 of those are in
  state 2.
- A diagnostic recency gate with no error content (computed to attribute the mechanism, never
  a selection candidate) nearly reproduces the selected rows:

| L | calibrated m=1 | state-0/1 recency only |
|---|---|---|
| 6 h | 4/4, 0.95 fa/wk, 1.06% | 4/4, 0.95 fa/wk, 1.25% |
| 12 h | 4/4, 0.95 fa/wk, 1.98% | 4/4, 0.79 fa/wk, 2.14% |

The threshold crossings themselves account for only the 1.25% to 1.06% difference: 39
policy-active windows where the two conditions disagree.

**The actual alarm, gated** (`results/v6_error_gated_sweep_metropt3.jsonl`, `alarm_index` 0).
It already meets the budget at 3/4 detected (0.60% clean time).

| gate | dataset errors | calibrated crossings |
|---|---|---|
| m=1, L=0 | 0/4, 0.01% | 0/4, 0.02% |
| m=1, L=6 h | 2/4, 0.13% | 2/4, 0.23% |
| m=1, L=24 h | 2/4, 0.40% | 3/4, 0.41% |
| m=2, L=24 h | 1/4, 0.38% | 3/4, 0.40% |

At `L=0`, no window inside any event has both the policy alarm and an error.

## 6. MetroPT2: the gate inherits a silent policy

The ungated policies detect nothing on MetroPT2, so no AND gate can.

**Inside events, the frozen scores barely cross their thresholds:**
- Jan: 0 novelty and 0 health crossings across all 352 event windows.
- Apr: 3 novelty crossings in event 0 and 3 + 9 in event 1. The longest run is 3 windows.
- All three Apr event-0 crossings are `new_candidate` windows in unbaselined sustained load.
  The `new_state_alert` veto turns them into alerts, as on MetroPT3 in the v4 study.

**On clean windows, the crossing rates are far from the nominal 0.3% level:**
- Jan: 0.008% (novelty) and 0.045% (health).
- Apr: 0.44% and 0.10%.

**The dataset errors would have separated these events:**
- Apr (m=1, L=1 h): present on every window of both events, and on 0% of clean windows.
- Jan (m=1, L=1 h): present on event 0 (19 of 25 windows) and on 3.6% of clean windows.

**The calibrated crossings would not:** at L=6 h they are on 83% (Jan) and 58% (Apr) of clean
windows.

## 7. Caveats

- One development stream with 4 events; single deterministic runs; no confidence intervals.
- Every first-alarm offset in section 3 is positive: these alarms confirm failures that are
  already reported. They do not warn early.
- The gate applies to alarm episodes, not to the policy's `new_state` / `unbaselined_*`
  alerts. The two selected rows still raise about 40 notifications per week (all gated rows:
  6.7–41.9).
- The `Oil_level` polarity is inferred from calibration, not documented. An OR combination (an
  error alone raises the alarm) was not studied, because the requested rule requires the
  alarm measure.
- Both MetroPT2 recordings are consumed. A follow-up motivated by section 6 needs a fresh
  recording, for example an error-only or error-OR-policy rule.
- This is a mechanism study, not a recommended production configuration.

## Reproduce

```bash
P="python -m bench.error_gated_sweep"
$P sweep && $P select && $P replay && $P relations
```
