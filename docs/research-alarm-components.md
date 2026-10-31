# How each alarm component behaves, and why (v8)

This study follows the [alarm definition](research-alarm-definition.md) (v7). It asks four
questions:

- How does each component of the alarm behave, physically and mathematically?
- How does each component behave when the others are changed?
- Which of the open issues from v7 can be fixed, and which cannot?
- Are the fixes correct, verified independently?

The alarm measure throughout is uEngram's own: the frozen seed-0 DSP scores, streamed through
`AlarmPolicy`. Nothing is retrained or rescored.

Code:
- On/off-delay filters: `src/uengram/persistence.py`
- Logic options on `AlarmPolicy`: `src/uengram/alarm.py` (`AlarmLogic`)
- False-alarm math: `bench/alarm_math.py`
- Component profile: `bench/alarm_components.py`
- Factorial study: `bench/alarm_interaction.py`
- Physical error sources and early-warning metric: `bench/alarm_definition.py`
- Three-valued v5 hold: `bench/modality_alarm.py`
- Tests: `tests/test_alarm_math.py`, `tests/test_alarm_v8.py`
- Results: `results/v8_*`

Every default reproduces the earlier alarms exactly. The v4, v6 and v7 artifacts reproduce byte
for byte.

## 1. Physical state is the fault signature (MetroPT3)

The operating state comes from the share of the window the compressor runs loaded: 0 idle, 1
cycling, 2 sustained load.

**Eligible windows per state:**

| | idle | cycling | sustained |
|---|---|---|---|
| commissioning | 1,699 | 933 | **0** |
| calibration | 572 | 333 | **1** |
| test | 9,156 | 10,396 | 1,744 |

**What each component does, per class of window** (% of eligible windows;
`results/v8_component_profile.json`):

| class, state | windows | novelty > setpoint | health > setpoint | new memory state | low pressure | oil departed |
|---|---|---|---|---|---|---|
| clean, idle | 8,600 | 0.3 | 0.4 | 1.8 | 0.0 | 13.2 |
| clean, cycling | 9,726 | 0.8 | 1.2 | 16.8 | 0.1 | 13.4 |
| clean, sustained | 1,073 | **58.2** | **87.2** | 51.8 | 3.0 | 33.2 |
| 24 h before a report, sustained | 53 | 5.7 | 5.7 | 45.3 | 0.0 | 0.0 |
| event, sustained | **492 of 495** | 94.9 | 97.0 | 68.5 | 2.2 | 0.0 |

The physics, in four points:

1. **An air leak makes the compressor run loaded without stopping.** 492 of 495 event windows
   are sustained load.
2. **uEngram never saw sustained load before the test.** Commissioning has 0 such windows and
   calibration has 1, so its memory and frozen health reference contain no loaded-running
   state.
   - Every sustained-load window is far from the reference: 87% of *healthy* ones exceed the
     health setpoint.
   - The score is high because the state is new, not because the machine is failing.
   - Correlation of each component with sustained load on clean windows: health 0.86, novelty
     0.70.
3. **Within sustained load, uEngram does see something, but nothing early.**
   - AUROC, clean vs event windows: health 0.84, novelty 0.78.
   - AUROC, clean vs the 24 h *before* a report: health 0.07, novelty 0.35.
   - Before a failure, sustained-load windows look *more* normal than healthy sustained load.
     The scores carry no early-warning signal in that state.
4. **The dataset errors behave as their physics says** (`physics.event_low_pressure`).
   - Healthy start-up: TP3 reaches the 8.2 bar cut-in within the first window of every
     logging restart before the test.
   - MetroPT3's event low pressure (Jun 5–7, Jul 15) begins in the first window after a
     restart, and TP3 never reaches the cut-in. The compressor cannot build pressure, which
     an air leak causes and a healthy restart never shows.
   - MetroPT2's comes after TP3 had reached the cut-in in the same run (0.3–39 h into it):
     pressure was built and then lost.
   - The oil-level flag departs about twice as often under sustained load (26% vs 13%), a load
     effect on MetroPT3.

## 2. The false-alarm arithmetic (exact)

A persistence filter's false-alarm rate depends on how exceedances cluster in time.
`bench/alarm_math.py` models the counted exceedance on healthy windows as a two-state Markov
chain:
- `a` = P(exceed | previous window normal)
- `b` = P(exceed | previous window exceeded)

It solves the filter plus chain as an absorbing Markov chain for the mean windows to a false
trigger (ARL0). This is exact for `counter`, `consecutive` and `m_of_n`, and uses a fine grid
for `ewma`.

The solver is checked three ways:
- against the closed form `(1-p^k)/((1-p)p^k)` for independent windows;
- against a simulation of the real filter code;
- for unreachable triggers, where it correctly returns infinity.

**MetroPT3, actual alarm, clean windows:** p = 0.0123, a = 0.0089, b = 0.297 (19,180 window
pairs, 18.9 clean weeks).

| filter, k = 3 | ARL0 assuming independence | false triggers/wk | ARL0 with clustering | false triggers/wk |
|---|---|---|---|---|
| counter (actual) | 534,904 | 0.002 | 1,283 | **0.80** |
| consecutive | 541,411 | 0.002 | 1,289 | 0.80 |
| m of 2k | 56,899 | 0.018 | 1,171 | 0.88 |
| EWMA | 174,432 | 0.006 | 1,230 | 0.83 |

**Observed: 0.74 openings per week.**

- The clustering model predicts the observed rate within 8%. Independence underpredicts it
  about 400 times.
- Clustering also explains why the choice of filter barely matters (section 4). False alarms
  come from long runs of exceedance during healthy sustained load, and any filter of length 3
  to 6 lets a long run through.
- On MetroPT2, `b = 0`: the few exceedances never come two in a row.
  - For the counter and consecutive filters, a 3-window trigger is unreachable, so the
    predicted rate is 0.
  - m-of-n and EWMA can accumulate non-consecutive exceedances; they predict at most 2e-5 per
    week.
  - Observed: 0.

## 3. The fixes

| issue (v7) | fix | effect (section 4) |
|---|---|---|
| Unknown evidence with a uEngram crossing held the counter, but so did unknown evidence *without* one | Corroboration is now the Kleene AND of the crossing and the vote. No crossing means normal even when the vote is unknown; only a crossing with an unknown vote holds. | Each corroborated counter can never exceed the uncorroborated one, so corroboration provably only removes alarm time (v7 calibrated rows above the policy: 73 → 0). v7 dataset and uncorroborated rows, selections, replay and trace are byte-identical; 256 calibrated rows changed. |
| Persistence was a single counter | Four on/off-delay filters (counter, ISA consecutive timer, m-of-n generalized timer, Jakobs et al. EWMA), all three-valued | measured below |
| The new-state veto counted a vetoed crossing as *normal* | `veto="hold"`: a new memory state with normal health is unknown | measured below |
| A state without a reference was judged against the GLOBAL setpoint | `unbaselined="unknown"` (not judged), or `"commission"`: the state's first ⌈(1−α)/α⌉ eligible windows become its own reference, causally | measured below |
| Unknown inputs could hold an alarm forever | `stale_windows`: more than this many unknown inputs in a row reset the filter | measured below |
| Low pressure during start-up could be counted | `physical` sources: low pressure is unknown only within a measured healthy fill time; oil level is judged only while idle | Identical to dataset low pressure on eligible windows of all three streams. Healthy fill takes one window, and the first window after a restart is never eligible. |
| "Warned before report" counted an alarm that cleared before the failure | The episode that first catches the failure must open before the report. "2 h before LPS" uses the same episode. | tested |
| v5's hold restarted on unjudged windows | `hold(..., known=, stale=)`: undecided windows pause the run | section 6 |
| One long line in `alarm.py` | wrapped | lint clean |

## 4. Factorial study: each component at every level of the others

The design crosses every level of eight components: 1,728 alarms on MetroPT3, all evaluated.

| component | levels (first = actual alarm) |
|---|---|
| alpha | 0.003, 0.01, 0.03 |
| channels | novelty only, or novelty + trusted health |
| unbaselined | global, unknown, commission |
| veto | normal, hold |
| persistence | counter, consecutive, m_of_n, ewma |
| k | 3, 6 |
| corroboration | none, dataset, physical (m=1, L=6 h) |
| stale | 0, 36 |

**Main effects.** Each row is the median change against the actual alarm's level, holding all
other components fixed (`results/v8_interaction_effects.json`):

| change | pairs | events | clean time | false alarms/wk |
|---|---|---|---|---|
| health channel on | 864 | **+1** | +1.99 pp | +0.16 |
| alpha 0.003 → 0.01 | 576 | 0 | +1.17 pp | +0.16 |
| unbaselined → unknown | 576 | **−2** | −0.99 pp | −0.16 |
| unbaselined → commission | 576 | 0 | −0.58 pp | −0.11 |
| corroboration dataset / physical | 576 | **−1** | −1.40 pp | −0.21 |
| k 3 → 6 | 864 | 0 | −0.06 pp | −0.16 |
| veto → hold | 864 | 0 (+1 in 48 pairs, never −) | 0 (up in 328, never down) | 0 |
| persistence → consecutive / m_of_n / ewma | 432 each | 0 | 0 / +0.09 / −0.01 pp | 0 |
| stale 36 | 864 | 0 (never changes) | 0 (down in 88, never up) | 0 |

**Interactions** (median change of X at each level of Y):

- **Health channel × unbaselined.**
  - The health channel adds clean time at every level: +3.31 pp with global, +1.93 pp with
    unknown, +1.96 pp with commission.
  - It adds an event with global and unknown, but **none with commission**.
- **Commission × channels.**
  - Novelty only: −0.24 pp.
  - With health: −2.26 pp and **−1 event**.
  - The commissioned reference (2–12 Mar 2020, no event windows inside) learns healthy
    sustained load. Jul 15's windows sit inside that healthy range, so it is lost. So when the
    health channel catches Jul 15 under a GLOBAL setpoint, it detects *sustained load*, not
    the leak.
- **Veto × unbaselined.** The hold mode matters only where vetoes occur. 827 of the 930
  vetoed crossings are in sustained load (+0.12 pp with global, 0 elsewhere).
- **Persistence × k.** No interaction worth naming (|change| ≤ 0.08 pp), as section 2
  predicts.
- **Corroboration × channels.** −1 event at both levels. The error sources never add recall;
  they only remove it.

**Per-event ceilings over all 1,728 alarms:**

| event | caught in | within both alarm budgets | caught only with |
|---|---|---|---|
| Apr 18 | 1,224 | 644 | any |
| May 29–30 | 384 | 188 | corroboration none, unbaselined global/commission |
| Jun 5–7 | 1,128 | 676 | unbaselined global/commission |
| Jul 15 | 228 | **0** | health channel, unbaselined global/unknown |

**Every alarm that catches Jul 15 spends at least 2.53% of healthy time in alarm.** That is
because it must trust health in sustained load, which 87% of healthy sustained-load windows
exceed. So **4/4 within the budget is impossible for any combination of these components**: 0
of 1,728 rows. 96 rows reach 4/4; the lowest clean time among them is 4.52%.

**Selection.** The rule was fixed before the sweep:

- No row met the full budget, so the fallback applied: within both budgets, the most events.
  It picked the EWMA filter, otherwise the actual alarm: 3/4, 0.90 false alarms/wk, 0.54%.
  This is the Jakobs et al. filter, and it opens Jun 5–7 at +3.7 h instead of +14.3 h.
- The row with the most early warnings within both budgets warns Apr 18 13.4 h before its
  report: `alpha=0.01`, health on, commissioned sustained load, `k=6`, dataset corroboration.
  It detects 2/4.

## 5. MetroPT2 (consumed; diagnostic replay only)

The selected alarm and the actual alarm both detect 0/3 (Jan) and 0/2 (Apr), with no false
alarms. The component profile shows why no component setting can change that:

| | MetroPT2-Jan | MetroPT2-Apr |
|---|---|---|
| eligible event windows in sustained load | 31 of 352 | 29 of 382 |
| health AUROC, clean vs event, idle / cycling | 0.77 / 0.58 | **0.99 / 0.93** |
| novelty or health crossings on clean windows | under 0.1% | under 0.5% |
| crossings two windows in a row (`b`) | 0 | 0 |

- **MetroPT2 failures happen at idle or while cycling**, unlike the MetroPT3 leaks.
- **On Apr, health separates the failures almost perfectly within the baselined states**, but
  the scores never cross the calibrated setpoint twice in a row.
- **The dataset errors would have caught both Apr events, and they are clean** (section 7 of
  v7). That needs a rule in which an error can raise the alarm on its own. The requested rule
  requires the uEngram crossing, and both recordings are now consumed.

## 6. The v5 rule with a three-valued hold

| stream | v5 (reset) | three-valued hold |
|---|---|---|
| MetroPT3 | 4/4, 0.32 fa/wk, 3.11% | 4/4, 1.11 fa/wk, 6.64% |
| MetroPT2-Jan | 0/3, 0.39, 1.16% | 1/3, 0.08, **54.3%** |
| MetroPT2-Apr | 0/2, 0, 0% | 1/2, 0, 0% |

The fix is logically correct, but it removes an accidental safeguard:

- The old reset kept breaking long runs and so hid **calibration drift**.
- On judged healthy Jan windows, `reservoirs_min_bar` is in error 0% of the time in January
  and 100% in April–May 2022.
- On MetroPT3, `oil_temp_c` is in error on 19–76% of judged clean windows per month from March
  to August 2020.
- A staleness bound does not help (unchanged at 36 windows), because these runs consist of
  *decided* TRUE windows, not unknown ones.

The calibration week does not represent later months for these channels. That is a reference
problem, not a hold-rule problem.

## 7. What remains unfixable with this data

1. **4/4 within budget on MetroPT3.** Section 4: catching Jul 15 needs a sustained-load
   setpoint, and the only such setpoint available (GLOBAL) flags healthy sustained load.
   Fixing it would need sustained-load commissioning data, which MetroPT3 does not have before
   the test split, or scores that separate faults *within* sustained load before the report.
   Measured AUROC there is 0.07–0.35.
2. **Early warning.** uEngram's scores carry no pre-report signal in the fault state. The only
   early warning within budgets is Apr 18 at 13.4 h, through health in cycling/idle windows.
3. **MetroPT2.** The policy never fires. Changing that needs an error-alone rule or a
   recalibrated reference, to be evaluated on a fresh recording.
4. **Calibration drift** (section 6) needs an adaptive or seasonally representative reference.

## 8. Independent verification

Separate implementations were written only from the specification in section 3. They do not
use `AlarmPolicy`, `uengram.persistence`, `apply_policy`, `evaluate_alarms` or any v6–v8
harness code.

| check | items | result |
|---|---|---|
| v8 sweep rows (every component at a non-default level at least once) | 8 rows × 5 metrics | all exact (max abs difference 0) |
| clustering: p, a, b, pairs, clean weeks; ARL0 independent and Markov, counter k=3 | 1 profile | exact; Markov rate differs by 1.9e-15 |
| v7 rows after the Kleene correction | 6 rows × 4 metrics | all exact |
| early-warning metric (new definition) | 3 rows | all exact |

Scripts: `~/scripts/uengram_v8_independent.py`, `~/scripts/uengram_v7_kleene_full.py`.

## Reproduce

```bash
export OPENBLAS_NUM_THREADS=1
python -m bench.alarm_components
P="python -m bench.alarm_interaction"
$P sweep --jobs 4 && $P select && $P effects && $P replay && $P trace
```

With one BLAS thread every `results/v8_*` file reproduces byte for byte. Other thread counts
change only the EWMA ARL0 of the profile, by under 1e-11 relative (threaded linear solve).
