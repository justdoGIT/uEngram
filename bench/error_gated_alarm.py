"""AlarmPolicy alarm that also requires at least m error conditions (threshold crossings).

The alarm measure is unchanged: `AlarmPolicy` (src/uengram/alarm.py) streams the frozen
novelty/health scores with its conformal thresholds, persistence counters and gating, and
yields P(t) in {0, 1} per eligible test window. The combined alarm adds one condition:

  A(t) = P(t) AND C(t),  C(t) = 1[ sum_j e~_j(t) >= m ]
  e~_j(t) = 1[ exists eligible test window s <= t with e_j(s) = 1 and end(t) - end(s) <= L h ]

L = 0 means the error must be in the same window. An episode opens where A rises from the
previous eligible window. Metrics, eligibility and event/clean definitions are those of
`bench.alarm_eval.evaluate_alarms`, so C(t) = 1 everywhere reproduces `apply_policy` exactly.

Error families, all label-free:

  dataset     errors the MetroPT documentation defines as threshold crossings
              (UCI MetroPT-3 variable list; MetroPT2 uses the same APU signals):
    lps             LPS "activates when the pressure drops below 7 bars": any sample of the
                    window differs from LPS's calibration-majority value (0 on every stream)
    oil_level       Oil_level "is active when the oil is below the expected values": any
                    sample differs from its calibration-majority value. The logged polarity
                    is not the documented one (1 throughout MetroPT3 calibration), so the
                    healthy value is taken from calibration; a constant channel never errs
    tp3_below_7bar  the same documented 7 bar threshold on the analog TP3 pressure:
                    min over the window's samples < 7 bar
  calibrated  v5 per-channel crossings (bench/modality_alarm.py, the v5 sweep baseline
              alpha=0.05, duty modality off): robust z^2 above a conformal per-state threshold

Excluded on purpose: MPG (pressure < 8.2 bar starts loading: normal control), Pressure_switch
and Towers (air-dryer tower discharge: normal operation), COMP/DV_eletric (load state).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bench.alarm_eval import apply_policy, evaluate_alarms
from bench.metropt_eval import episodes
from bench.modality_alarm import ModalityConfig, calibrate, modality_errors
from bench.streams import DIGITAL
from bench.v2_scoring import cached_scores
from uengram import metropt
from uengram.alarm import AlarmConfig

DATASET_ERRORS = ("lps", "oil_level", "tp3_below_7bar")
LPS_BAR = 7.0  # MetroPT documentation: LPS activates below 7 bar
SCORE_DIRS = {"metropt3": Path.home() / ".local/state/uengram/v2scores",
              "metropt2_jan": Path("data/metropt2/v2_cddfbf1"),
              "metropt2_apr": Path("data/metropt2/v2_cddfbf1")}
H = np.timedelta64(1, "h")


@dataclass(frozen=True)
class Gate:
    family: str = "dataset"  # "dataset" or "calibrated"
    min_errors: int = 1  # m: errors that must be present
    lookback_h: float = 0.0  # L: an error counts for this long after it occurred

    def label(self) -> str:
        return f"{self.family}:m{self.min_errors}:L{self.lookback_h:g}h"


def frozen_scores_available(name: str) -> bool:
    """Whether a dataset's frozen v2 scores are cached locally. CI runners don't have
    this developer-workstation cache; tests should skip rather than fail when it's absent."""
    return (SCORE_DIRS[name] / f"{name}_uengram_dsp_0_final.npz").exists()


def frozen_scores(stream) -> dict:
    """The cached seed-0 DSP uEngram scores; never rescored here."""
    path = SCORE_DIRS[stream.name] / f"{stream.name}_uengram_dsp_0_final.npz"
    if not path.exists():
        raise FileNotFoundError(f"frozen scores missing: {path}")
    return cached_scores(stream, "uengram", "dsp", 0, str(path.parent))


def _raw_flag(stream, name: str) -> np.ndarray:
    """[N, K] samples of a digital column that `Stream.cond` does not keep."""
    if stream.name == "metropt3":
        t, X, cols = metropt.load(metropt.fetch())
    else:
        from uengram import metropt2
        t, X, cols = metropt2.load(stream.name)
    W, start, _, _ = metropt.windows(t, X, cols, stream.W.shape[2], channels=(name,))
    if len(start) != len(stream.start) or not (start == stream.start).all():
        raise ValueError(f"{name} windows do not align with stream {stream.name}")
    return W[:, 0]


def _departs_from_calibration(samples: np.ndarray, stream) -> np.ndarray:
    """Window has a sample unlike the flag's calibration-majority value."""
    ref = stream.cal & stream.monitorable
    healthy = float(samples[ref].mean() > 0.5)
    return (np.abs(samples - healthy) > 0.5).any(axis=1)


def dataset_errors(stream) -> tuple[np.ndarray, list[str]]:
    """[N, 3] per-window dataset-defined errors over all windows."""
    lps = _departs_from_calibration(stream.cond[:, DIGITAL.index("LPS")], stream)
    oil = _departs_from_calibration(_raw_flag(stream, "Oil_level"), stream)
    tp3 = stream.W[:, list(metropt.ANALOG).index("TP3")].min(axis=1) < LPS_BAR
    return np.column_stack([lps, oil, tp3]), list(DATASET_ERRORS)


def calibrated_errors(stream) -> tuple[np.ndarray, list[str]]:
    """[N, 7] v5 per-channel crossings (test windows filled, others False)."""
    cfg = ModalityConfig(alpha=0.05, duty=False)  # v5 sweep baseline; 0.01 leaves all unjudgeable
    test = np.flatnonzero(stream.test)
    errors, names = modality_errors(stream, test, cfg, calibrate(stream, cfg))
    out = np.zeros((len(stream.start), errors.shape[1]), bool)
    out[test] = errors
    return out, names


def latch(errors: np.ndarray, end: np.ndarray, valid: np.ndarray, lookback_h: float) -> np.ndarray:
    """e~_j(t): an eligible error at or before t ended within lookback_h hours of end(t)."""
    n, J = errors.shape
    out = np.zeros((n, J), bool)
    idx = np.arange(n)
    limit = np.timedelta64(int(round(lookback_h * 3600)), "s")
    for j in range(J):
        hit = errors[:, j] & valid
        last = np.maximum.accumulate(np.where(hit, idx, -1))
        seen = last >= 0
        out[seen, j] = (end[seen] - end[last[seen]]) <= limit
    return out & valid[:, None]


def gate_alarm(active: np.ndarray, valid: np.ndarray, condition: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A = P AND C on eligible windows; an episode opens where A rises from the previous
    eligible window (a logging gap neither opens nor closes it)."""
    gated = np.asarray(active, bool) & np.asarray(condition, bool) & valid
    prev = np.zeros(len(gated), bool)
    last = False
    for i in np.flatnonzero(valid):
        prev[i] = last
        last = gated[i]
    return gated, gated & ~prev


class Replay:
    """One stream: frozen scores, error matrices, and cached AlarmPolicy runs."""

    def __init__(self, stream):
        self.stream = stream
        self.scores = frozen_scores(stream)
        self.test = np.flatnonzero(stream.test)
        self.errors = {"dataset": dataset_errors(stream), "calibrated": calibrated_errors(stream)}
        self._runs: dict[AlarmConfig, tuple] = {}

    def policy(self, cfg: AlarmConfig):
        if cfg not in self._runs:
            self._runs[cfg] = apply_policy(self.stream, self.scores, cfg)
        return self._runs[cfg]

    def condition(self, gate: Gate, valid: np.ndarray) -> np.ndarray:
        E, _ = self.errors[gate.family]
        latched = latch(E[self.test], self.stream.end[self.test], valid, gate.lookback_h)
        return latched.sum(axis=1) >= gate.min_errors

    def evaluate(self, cfg: AlarmConfig, gate: Gate | None) -> tuple[dict, dict]:
        """Metrics of the policy alone (gate None) or of A = P AND C."""
        m0, d0, pol = self.policy(cfg)
        s, t = self.stream, self.test
        valid = d0["valid"]
        condition = np.ones(len(t), bool) if gate is None else self.condition(gate, valid)
        active, opened = gate_alarm(d0["active"], valid, condition)
        alert_index = [a["index"] for a in pol.alerts]
        m, d = evaluate_alarms(s.start[t], s.end[t], s.event_id()[t], s.clean_negative()[t], valid,
                               s.events, active, opened, self.scores["novelty"][t],
                               self.scores["health"][t], alert_index, s.lps)
        alerts = np.zeros(len(t), bool)
        alerts[alert_index] = True
        weeks = d0["runtime_observed_hours"] / (7 * 24)
        m["notifications_per_week_all"] = episodes(s.end[t][(alerts | opened) & valid]) / weeks
        event = s.event_id()[t]
        d["policy_events_detected"] = m0["events_detected"]
        d["error_ceiling_events"] = int(len({int(e) for e in event[condition & valid & (event >= 0)]}))
        d["condition_share_clean"] = float(condition[valid & s.clean_negative()[t]].mean())
        return m, d
