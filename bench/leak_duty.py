"""Secondary analysis for MetroPT2 (pre-registered, not a pass criterion).

The standard compressed-air leak test measures leakage as the share of time a
load/unload compressor runs loaded while nothing consumes air (Compressed Air
Challenge, Fact Sheet 7: leakage % = 100 T / (T + t)). On a train, a window
spent stopped inside a workshop or parking polygon approximates that
no-demand context (GPS exists only in MetroPT2, so this could not be developed
on MetroPT-3). The indicator is the load share of such windows; its threshold
is the split-conformal (1 - alpha) quantile over commissioning and calibration
parked windows, and alarms need K parked exceedances (persistence counter
that only moves in parked windows). Parameters are fixed, not tuned.
"""
from __future__ import annotations

import numpy as np

from bench.alarm_eval import evaluate_alarms
from uengram.alarm import AlarmConfig, AlarmPolicy

ALPHA, K, MIN_PARKED = 0.01, 3, 0.9


def leak_duty(stream):
    parked = stream.extra["parked_share"] >= MIN_PARKED
    load = stream.features[:, stream.feature_names.index("load_share")]
    ref = (stream.fit | stream.cal) & parked & stream.monitorable
    cfg = AlarmConfig(alpha=ALPHA, k_novelty=K, k_health=0, per_state=False, new_state_alert=False)
    zeros = np.zeros(len(load), np.int64)
    pol = AlarmPolicy(cfg).calibrate(load[ref], np.full(ref.sum(), np.nan), zeros[ref], np.ones(ref.sum(), bool))
    test = np.flatnonzero(stream.test)
    mon = parked[test] & stream.monitorable[test]
    active, opened = pol.run(stream.end[test], load[test], np.full(len(test), np.nan),
                             ["score"] * len(test), zeros[test], mon)
    m, d = evaluate_alarms(stream.start[test], stream.end[test], stream.event_id()[test],
                           stream.clean_negative()[test], mon, stream.events,
                           active, opened, np.where(mon, load[test], np.nan),
                           np.full(len(test), np.nan), (), stream.lps)
    d.update({"threshold_load_share": pol.thresholds(0)[0], "reference_parked_windows": int(ref.sum()),
              "test_parked_windows": int(mon.sum())})
    d["episodes"] = pol.episodes
    return m, d
