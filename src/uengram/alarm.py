"""Alarm policy v2: per-window scores -> alarm episodes and alerts.

Protocol in docs/EXPLAINED.md section 17. Each window has a novelty score
(distance to the adaptive memory's normal states) and a health score (distance
to the frozen commissioning reference). Both are compared with split-conformal
thresholds calibrated on a label-free calibration period, per operating state
when the state has enough calibration windows, otherwise on all states.

- Persistence: an exceedance raises its counter by 1 (capped at its k), a
  normal window lowers it by 1 (not below 0). An alarm opens when a counter
  reaches its k and closes when both counters are back at 0, so clearing takes
  k normal windows however long the deviation lasted (a fixed off-delay).
- State gating: windows the caller marks unmonitorable (logging restarts,
  maintenance, stopped machine) neither raise nor lower the counters.
- Alerts, not alarms: entering a state without its own baseline raises an
  "unbaselined_state" alert; a window that opens a new memory candidate while
  health is normal raises a "new_state" alert and does not count as novelty;
  optionally, health exceedances in a state without its own baseline raise
  "unbaselined_health" alerts instead of counting, because the frozen
  reference never saw that state.
- Adaptive level (Gibbs & Candes 2021, ACI): optionally, the novelty level
  alpha_t moves by gamma * (alpha - exceeded) on selected healthy-looking
  windows (no open alarm, health below its threshold). This adjusts score
  exceedance frequency on those selected windows; it does not guarantee rates
  of actual failures, event false alarms, or behavior on excluded windows.
  It is not clipped; alpha_t stays in [-gamma, 1 + gamma] (level <= 0 means
  no exceedance). Updates are gated so that a developing fault is not adapted
  away.
- Freeze frame: every alarm episode records the opening window's state,
  scores, thresholds, memory action, snapshot features and the features that
  deviate most from the calibration median of that state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .stats import conformal_threshold

GLOBAL = -1  # baseline key for the all-states fallback
MAX_SNAPSHOT = 20  # freeze frames list all feature values only up to this many


@dataclass(frozen=True)
class AlarmConfig:
    alpha: float = 0.01  # per-window exceedance level of every threshold
    k_novelty: int = 3  # novelty persistence in windows; 0 disables novelty alarms
    k_health: int = 6  # health persistence in windows; 0 disables health alarms
    per_state: bool = True  # own thresholds for states with enough calibration windows
    min_state_windows: int = 100  # conformal alpha = 0.01 needs at least 99 scores
    new_state_alert: bool = True  # new candidates with normal health alert, not count
    aci_gamma: float = 0.0  # adaptive novelty level step; 0 = fixed level
    # States without their own baseline cannot be judged by the frozen health
    # reference: their health exceedances raise alerts instead of counting.
    health_needs_baseline: bool = False


@dataclass
class StepResult:
    active: bool  # alarm open after this window
    opened: bool  # this window opened an alarm episode
    alert: str | None  # alert kind raised by this window


def _conformal_at(sorted_scores: np.ndarray, level: float) -> float:
    """conformal_threshold on pre-sorted scores at any level in [0, 1]."""
    m = len(sorted_scores)
    if level >= 1.0 or m == 0:
        return -math.inf if m else math.inf
    k = int(math.ceil((1.0 - level) * (m + 1)))
    return float(sorted_scores[k - 1]) if k <= m else math.inf


@dataclass
class _Baseline:
    novelty: np.ndarray  # sorted finite calibration novelty scores
    health_thr: float
    feat_med: np.ndarray | None = None
    feat_scale: np.ndarray | None = None


@dataclass
class AlarmPolicy:
    cfg: AlarmConfig = field(default_factory=AlarmConfig)
    feature_names: tuple[str, ...] = ()

    def __post_init__(self):
        self.baselines: dict[int, _Baseline] = {}
        self.baseline_counts: dict[int, dict] = {}
        self.episodes: list[dict] = []
        self.alerts: list[dict] = []
        self.level = self.cfg.alpha
        self.c_nov = self.c_health = 0
        self.active = False
        self._prev_state: int | None = None
    # -- calibration -----------------------------------------------------
    def calibrate(self, novelty, health, state, monitorable, features=None) -> "AlarmPolicy":
        """Thresholds from label-free calibration windows (monitorable only)."""
        novelty, health = np.asarray(novelty, float), np.asarray(health, float)
        state, ok = np.asarray(state), np.asarray(monitorable, bool)
        groups = [(GLOBAL, ok)]
        if self.cfg.per_state:
            groups += [(int(s), ok & (state == s)) for s in np.unique(state[ok])]
        for key, m in groups:
            if key != GLOBAL and m.sum() < self.cfg.min_state_windows:
                continue
            nov = np.sort(novelty[m][np.isfinite(novelty[m])])
            h = health[m]
            finite_health = h[np.isfinite(h)]
            b = _Baseline(nov, conformal_threshold(finite_health, self.cfg.alpha))
            if features is not None and m.any():
                f = np.asarray(features, float)[m]
                q25, b.feat_med, q75 = np.nanpercentile(f, [25, 50, 75], axis=0)
                b.feat_scale = np.maximum((q75 - q25) / 1.349, 1e-9)
            self.baselines[key] = b
            self.baseline_counts[key] = {
                "monitorable_windows": int(m.sum()),
                "novelty_scores": int(np.isfinite(novelty[m]).sum()),
                "health_scores": int(np.isfinite(h).sum()),
                "novelty_threshold_supported": len(nov) > 0 and math.isfinite(_conformal_at(nov, self.cfg.alpha)),
                "health_threshold_supported": len(finite_health) > 0 and math.isfinite(conformal_threshold(finite_health, self.cfg.alpha)),
            }
        return self

    def thresholds(self, state: int) -> tuple[float, float]:
        b = self.baselines.get(int(state), self.baselines[GLOBAL])
        return _conformal_at(b.novelty, self.level), b.health_thr

    # -- streaming ----------------------------------------------------------
    def step(self, index: int, time, novelty: float, health: float, action: str,
             state: int, monitorable: bool, features=None) -> StepResult:
        if not monitorable:
            return StepResult(False, False, None)
        own = state in self.baselines
        b = self.baselines[state] if own else self.baselines[GLOBAL]
        alert = None
        if self.cfg.per_state and not own and self._prev_state != state:
            alert = "unbaselined_state"
        self._prev_state = state
        n_thr, h_thr = self.thresholds(state)
        h_ex = bool(health > h_thr)  # NaN -> False
        n_ex = bool(novelty > n_thr)
        # Without its own baseline, a state's health is not judged against the
        # frozen reference (health_needs_baseline): it neither counts nor vetoes.
        h_trusted = not (self.cfg.health_needs_baseline and self.cfg.per_state and not own)
        n_count, h_count = n_ex, h_ex and h_trusted
        if self.cfg.new_state_alert and action == "new_candidate" and not h_count:
            n_count = False
            alert = alert or "new_state"
        if h_ex and not h_trusted:
            alert = alert or "unbaselined_health"
        if alert:
            self.alerts.append({"index": index, "time": str(time), "kind": alert, "state": state})
        if self.cfg.aci_gamma and not self.active and not h_ex and math.isfinite(novelty):
            self.level += self.cfg.aci_gamma * (self.cfg.alpha - n_ex)
        # Counters are capped at their k, so an alarm clears after k normal
        # windows however long the deviation lasted (a fixed off-delay).
        self.c_nov = (min(self.c_nov + 1, self.cfg.k_novelty) if n_count else
                      max(self.c_nov - 1, 0)) if self.cfg.k_novelty > 0 else 0
        self.c_health = (min(self.c_health + 1, self.cfg.k_health) if h_count else
                         max(self.c_health - 1, 0)) if self.cfg.k_health > 0 else 0
        opened = False
        if self.active:
            ep = self.episodes[-1]
            ep["peak_novelty"] = max(ep["peak_novelty"], novelty) if math.isfinite(novelty) else ep["peak_novelty"]
            ep["peak_health"] = max(ep["peak_health"], health) if math.isfinite(health) else ep["peak_health"]
            ep["windows"] += 1
            if self.c_nov == 0 and self.c_health == 0:
                self.active = False
                ep["close_index"], ep["close_time"] = index, str(time)
        else:
            by_nov = self.cfg.k_novelty > 0 and self.c_nov >= self.cfg.k_novelty
            by_health = self.cfg.k_health > 0 and self.c_health >= self.cfg.k_health
            if by_nov or by_health:
                self.active = opened = True
                self.episodes.append(self._freeze_frame(
                    index, time, novelty, health, action, state, own, n_thr, h_thr,
                    "both" if by_nov and by_health else "novelty" if by_nov else "health",
                    b, features))
        return StepResult(self.active, opened, alert)

    def run(self, end, novelty, health, actions, state, monitorable, features=None):
        """Stream whole arrays; returns eligible active/opened bool arrays."""
        n = len(novelty)
        active, opened = np.zeros(n, bool), np.zeros(n, bool)
        for i in range(n):
            eligible = bool(monitorable[i])
            r = self.step(i, end[i], float(novelty[i]), float(health[i]), actions[i],
                          int(state[i]), eligible,
                          None if features is None else features[i])
            active[i], opened[i] = r.active, r.opened
        return active, opened

    def _freeze_frame(self, index, time, novelty, health, action, state, own, n_thr, h_thr,
                      trigger, b: _Baseline, features) -> dict:
        frame = {"open_index": index, "open_time": str(time), "trigger": trigger,
                 "state": state, "baselined_state": own, "novelty": novelty, "health": health,
                 "novelty_threshold": n_thr, "health_threshold": h_thr, "action": action,
                 "novelty_level": self.level, "snapshot": {}, "top_deviations": [],
                 "peak_novelty": novelty if math.isfinite(novelty) else -math.inf,
                 "peak_health": health if math.isfinite(health) else -math.inf,
                 "windows": 1, "close_index": None, "close_time": None}
        if features is not None and self.feature_names:
            f = np.asarray(features, float)
            if len(f) <= MAX_SNAPSHOT:
                frame["snapshot"] = {k: float(v) for k, v in zip(self.feature_names, f)}
            if b.feat_med is not None:
                z = (f - b.feat_med) / b.feat_scale
                order = np.argsort(-np.abs(np.nan_to_num(z)))[:3]
                frame["top_deviations"] = [(self.feature_names[j], float(z[j])) for j in order]
        return frame
