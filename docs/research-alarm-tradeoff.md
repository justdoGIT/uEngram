# MetroPT3 recall-vs-clean-alarm-time: a parameter sensitivity study

Development-only replay of frozen DSP novelty/health scores through `AlarmPolicy`
(`src/uengram/alarm.py`) under many configurations. No retraining, no new
inference; read-only on the cached scores at `~/.local/state/uengram/v2scores`.
Harness: `bench/alarm_sensitivity.py`. Raw sweeps: `results/v4_alarm_sensitivity_ofat.jsonl`,
`results/v4_alarm_sensitivity_pairwise_A.jsonl`, `results/v4_alarm_sensitivity_pairwise_B.jsonl`,
`results/v4_alarm_sensitivity_summary.json`. Full math/state inventory:
`results/v4_error_definition.json`.

## 1. The error, precisely

No explored configuration jointly satisfies, on MetroPT3 (DSP encoder, seed 0):

```
events_detected == 4  AND  alarm_time_share_clean <= 0.02  AND  false_alarm_per_week <= 1
```

The frozen baseline (`v2_frozen`: alpha=0.003, k_novelty=3, k_health=0, per_state=True,
new_state_alert=True, health_needs_baseline=True) satisfies the budget but misses event 3
(July 15 air leak, 26 eligible windows, 0 alarm windows). Every configuration found so far
that detects all 4 events pushes `alarm_time_share_clean` to 4.6-9.1% — 2.3-4.5x the 2%
budget — while `false_alarm_per_week` stays comparatively unaffected or even improves.

## 2. Mechanism (from `AlarmPolicy.step`, alarm.py:132-182)

Event 3 occurs in an operating regime the memory engine has never seen
(`action == "new_candidate"`). `new_state_alert` forces `n_count = False` whenever a new
candidate opens *and health is not also exceeding its threshold*: the novelty exceedance is
logged only as a `"new_state"` alert and never reaches the alarm counter. Separately,
`health_needs_baseline` makes health "untrusted" (`h_count` forced `False`) for any state
without its own calibration baseline when `per_state=True` — exactly event 3's state.

## 3. Single-variable sweep (OFAT): no lever alone closes the gap

| Parameter | Effect direction | Closest single-variable result |
|---|---|---|
| `alpha` (0.0003-0.1) | ↑alpha → ↓threshold → more alarms, elasticity on `false_alarm_per_week` 0.02-0.06 (weak), on `alarm_time_share_clean` 0.02-0.04 (weak), monotonic | `alpha=0.1`: **4/4 detected**, clean_share=1.41% (within budget) but `false_alarm_per_week=2.06` (over budget) — the single closest near-miss |
| `k_novelty` (1-10) | ↑k → fewer, shorter-lived alarms; strongly monotonic down on `false_alarm_per_week` (elasticity −3 to −10) and `events_detected` (elasticity down to −1: `k_novelty=4` already drops a *different* event, 3→2 detected) | Never reaches 4; raising it only loses more recall |
| `k_health` alone (`health_needs_baseline` left at baseline `True`) | **No effect at all**: 3.0 detected at every k in {1,3,6,10} | Health never "own-baselined" for event 3's state, so `h_count` stays forced `False` regardless of persistence |
| `health_needs_baseline=False` alone (`k_health` left at baseline `0`) | **No effect**: `k_health=0` disables the channel outright | — |
| `per_state=False` | Global thresholds raise `alarm_time_share_clean` 0.60%→2.81% for the *same* 3/4 recall — a pure regression, no recall gain | — |
| `new_state_alert=False` alone (`k_health` left at `0`) | **No effect**: still 3.0 detected | Confirms the veto is not sufficient by itself; the health gate must also be open |
| `aci_gamma` (1e-4 to 1e-2) | Adapts alpha only on healthy, quiet, finite-novelty windows; raising it *tightens* the effective level over time here (fewer false alarms, `alarm_time_share_clean` 0.60%→0.08% at 1e-2), no recall change | Cannot help recall: it only adjusts the already-failing novelty-only path |

`results/v4_alarm_sensitivity_summary.json` records `any_recall_and_budget: false` for
every parameter varied alone.

## 4. The real interaction: an AND-gate, not an additive effect

Pairwise sweep A (`health_needs_baseline` × `k_health`, `new_state_alert` held at baseline
`True`):

| `health_needs_baseline` | `k_health` | `events_detected` | `alarm_time_share_clean` |
|---|---:|---:|---:|
| True | 0, 1, 3, 6, 10 | 3.0 (always) | 0.60-1.39% |
| False | 0 | 3.0 | 2.83% |
| False | 1, 3, 6, 10 | **4.0** | **4.75-6.17%** |

Detection flips to 4/4 only in the bottom-right cell: both conditions must hold
simultaneously. Neither margin alone moves `events_detected`; this is a logical AND, not a
smooth or additive interaction, and it is exactly what the step-function veto conditions in
section 2 predict.

**`new_state_alert` turns out not to be the active ingredient once the health channel is
open.** Pairwise sweep B (`new_state_alert` x `k_health`, `health_needs_baseline` fixed
`False`) shows `events_detected` and `alarm_time_share_clean` are **exactly identical**
between `new_state_alert=True` and `False` at `k_health in {3,6,10}`. At `k_health=1`
(fastest open/close) there is a small genuine residual: 6.179% vs 6.168% clean share
(0.17% relative) -- the novelty path still occasionally extends a flicker episode the
health path alone would not, but this is a minor edge effect, not the mechanism. Once the
health channel is trusted and enabled, event 3 opens mainly through the *health* counter
path, which `new_state_alert` never gates (alarm.py:149, the veto only ever sets
`n_count=False`). The published fix name "no-veto, with health" names a mechanism that is
at most a minor contributor at low `k_health`: what actually works is "health channel
trusted + enabled," not "veto removed."

`k_health` is **non-monotonic** in `alarm_time_share_clean` once the gate is open: `k_health=1`
(fastest open *and* close) gives the *highest* clean share (6.17%), higher than `k_health=3..10`
(4.75-5.08%, roughly flat). A low persistence threshold is noisiest — many brief health
exceedances outside the four reported events each open a short-lived alarm, and total
standing time from many short episodes exceeds that from fewer, longer, better-targeted
ones. `k_health=3-6` is the better region of this axis, not the smallest value.

## 5. Does a 3-knob combination close the budget?

Grid over `k_health in {1,3,6}` x `k_novelty in {3,4,6}` x `alpha in {0.003,0.01}`, all with
`health_needs_baseline=False`: every combination that reaches 4/4 detection lands
`alarm_time_share_clean` in **4.6-9.1%**; the best observed point (`k_health=6, k_novelty=6,
alpha=0.003`) still gives 4.62%, 2.3x the 2% budget, with `false_alarm_per_week=0.635` (within
its own budget). `k_novelty` has almost no effect once the health channel dominates
(3→6 moves clean_share by under half a point). **No point in this grid satisfies all three
conditions.**

## 6. Conclusion

The recall gap on MetroPT3 is not a threshold-tuning problem solvable within this parameter
family: detecting event 3 requires trusting and enabling the health channel for an
unbaselined state, and doing so necessarily lets the policy stand in alarm for a large share
of every event's duration (not just event 3's), because the health score stays elevated for
as long as the underlying condition does. The false-alarm *episode* rate stays nearly flat
across this change; the budget is broken by alarm *duration*, not alarm *frequency*. Closing
this gap needs either a duration-aware budget definition, a decaying/capped health trigger
that still opens on first exceedance but does not hold the alarm open for the health score's
full elevated span, or an entirely different signal for event 3's regime. None of those are
explored here.

## Scope and caveats

Development-only, MetroPT3 (consumed data), DSP encoder, seed 0. Single-seed point
estimates, not confidence intervals; no claim of transfer to MetroPT2 or to any
unseen machine. This is a mechanism study of an already-exploratory follow-up, not a new
confirmation and not a recommended production configuration.
