"""Two-condition alarm: at least m modalities in error, AND that holding for long.

`AlarmPolicy` (src/uengram/alarm.py) opens on one pooled novelty/health distance. This rule
instead judges each modality separately and requires BOTH:

  A. at least `min_modalities` of the J modalities are in error on the window, where
     - a physical sensor channel j is in error when its robust z^2 exceeds its own
       label-free threshold for the current operating state, and
     - the load-duty modality is in error on every sustained-load window (state 2);
  B. condition A has held for `k_hold` consecutive eligible windows.

The alarm releases after `k_release` consecutive eligible windows without condition A. A
logging gap (unmonitorable window) resets the run, because continuity across unobserved
time cannot be claimed.

Thresholds come from stream.cal (or fit+cal) only, never stream.test. A physical channel in
an operating state with no own reference is not judged (unbaselined="ignore") rather than
compared with another mode's range; "global" keeps that cross-mode comparison for contrast.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from bench.alarm_eval import evaluate_alarms
from bench.metropt_eval import episodes
from bench.streams import DIGITAL
from uengram import metropt
from uengram.stats import conformal_threshold

GLOBAL = -1
SUSTAINED_LOAD = 2  # bench.streams LOAD_EDGES: load_share >= 0.4
# Physical sensor channels: excludes load_share/comp_switches (operating-mode descriptors,
# represented here by the duty modality) and lps_share (an existing low-pressure warning).
PHYSICAL_CHANNELS = ("motor_current_a", "oil_temp_c", "tp2_bar", "tp3_bar", "h1_bar",
                    "dv_pressure_bar", "reservoirs_min_bar")
# Pneumatic subset, fixed by physics before any sweep: an air leak is a pneumatic fault, and
# oil temperature / motor current also move with healthy load duration (duty-confounded).
PNEUMATIC_CHANNELS = ("tp2_bar", "tp3_bar", "h1_bar", "dv_pressure_bar", "reservoirs_min_bar")
DUTY = "sustained_load"
ANALOG_OF = {"tp2_bar": "TP2", "tp3_bar": "TP3", "h1_bar": "H1", "dv_pressure_bar": "DV_pressure",
             "reservoirs_min_bar": "Reservoirs", "oil_temp_c": "Oil_temperature",
             "motor_current_a": "Motor_current"}
MIN_LOADED_SAMPLES = 3  # a window needs this many loaded samples for a loaded-sample median


@dataclass(frozen=True)
class ModalityConfig:
    alpha: float = 0.01  # family-wise exceedance level over the physical channels
    correction: str = "bonferroni"  # "bonferroni": alpha/J per channel; "none": alpha each
    calibration: str = "conformal"  # "conformal" empirical quantile; "parametric" Gaussian
    per_state: bool = True
    min_state_windows: int = 100
    reference: str = "cal"  # "cal" or "fit+cal" (both label-free, both before test)
    unbaselined: str = "ignore"  # "ignore" or "global" for states without own reference
    duty: bool = True  # include the sustained-load duty modality
    min_modalities: int = 1  # condition A: m of J modalities in error
    k_hold: int = 6  # condition B: consecutive eligible windows
    k_release: int = 1  # consecutive non-error eligible windows that clear the alarm
    channels: tuple[str, ...] = PHYSICAL_CHANNELS
    conditioning: str = "state"  # "state": window summary vs its operating-state reference.
    # "loaded": median of each channel over the window's loaded samples (DV_eletric active)
    # vs one pooled reference of loaded samples, so cycling-load calibration data can judge
    # sustained-load windows; per_state is not used in this mode.


def chi2_1_isf(alpha: float) -> float:
    """x with P(Z^2 > x) = alpha for Z~N(0,1) (one degree of freedom), by bisection."""
    lo, hi = 0.0, 200.0
    while hi - lo > 1e-10:
        mid = 0.5 * (lo + hi)
        if math.erfc(math.sqrt(mid / 2.0)) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _reference_mask(stream, cfg: ModalityConfig) -> np.ndarray:
    return stream.cal | stream.fit if cfg.reference == "fit+cal" else stream.cal


def _per_state(cfg: ModalityConfig) -> bool:
    return cfg.per_state and cfg.conditioning == "state"


def channel_values(stream, cfg: ModalityConfig) -> np.ndarray:
    """[N, J] per-window channel values under the configured conditioning."""
    if cfg.conditioning == "state":
        return stream.features[:, [stream.feature_names.index(c) for c in cfg.channels]]
    analog = list(metropt.ANALOG)
    loaded = stream.cond[:, DIGITAL.index("DV_eletric")] > 0.5
    enough = loaded.sum(axis=1) >= MIN_LOADED_SAMPLES
    out = np.full((len(stream.W), len(cfg.channels)), np.nan)
    for j, name in enumerate(cfg.channels):
        x = np.where(loaded[enough], stream.W[enough, analog.index(ANALOG_OF[name])], np.nan)
        out[enough, j] = np.nanmedian(x, axis=1)
    return out  # NaN where too few loaded samples: never in error, never a reference row


def calibrate(stream, cfg: ModalityConfig, values: np.ndarray | None = None) -> dict[int, dict]:
    """state -> {median, scale, threshold[J]} from label-free reference windows."""
    values = channel_values(stream, cfg) if values is None else values
    if values.shape[1] == 0:  # duty-only: nothing to calibrate
        return {GLOBAL: {"windows": 0, "judgeable": [], "per_channel_level": None}}
    ref = _reference_mask(stream, cfg) & stream.monitorable & np.isfinite(values).any(axis=1)
    per_channel_alpha = cfg.alpha / values.shape[1] if cfg.correction == "bonferroni" else cfg.alpha

    def _fit(mask):
        q25, med, q75 = np.nanpercentile(values[mask], [25, 50, 75], axis=0)
        scale = np.maximum((q75 - q25) / 1.349, 1e-9)
        z2 = ((values[mask] - med) / scale) ** 2
        if cfg.calibration == "conformal":
            thr = np.array([conformal_threshold(z2[:, j], per_channel_alpha) for j in range(values.shape[1])])
        else:
            thr = np.full(values.shape[1], chi2_1_isf(per_channel_alpha))
        # conformal_threshold returns +inf when the reference is too small for the level
        # (needs m >= 1/level - 1 windows): that channel is unjudgeable, not healthy.
        return {"median": med, "scale": scale, "threshold": thr, "windows": int(mask.sum()),
                "judgeable": [c for c, t in zip(cfg.channels, thr) if np.isfinite(t)],
                "per_channel_level": per_channel_alpha}

    baselines = {GLOBAL: _fit(ref)}
    if _per_state(cfg):
        for state in np.unique(stream.state[ref]):
            mask = ref & (stream.state == state)
            if mask.sum() >= cfg.min_state_windows:
                baselines[int(state)] = _fit(mask)
    return baselines


def modality_errors(stream, test: np.ndarray, cfg: ModalityConfig, baselines: dict,
                    values: np.ndarray | None = None) -> tuple[np.ndarray, list[str]]:
    """Boolean matrix [len(test), J]: which modalities are in error on each test window."""
    values = (channel_values(stream, cfg) if values is None else values)[test]
    state = stream.state[test]
    errors = np.zeros(values.shape, bool)
    for s in np.unique(state) if values.shape[1] else ():
        rows = state == s
        own = int(s) in baselines  # only GLOBAL exists without per-state baselines
        if _per_state(cfg) and not own and cfg.unbaselined == "ignore":
            continue  # no reference for this operating mode: physical channels unjudged
        b = baselines[int(s)] if own else baselines[GLOBAL]
        z2 = ((values[rows] - b["median"]) / b["scale"]) ** 2
        errors[rows] = z2 > b["threshold"]  # NaN compares False: missing never errs
    names = list(cfg.channels)
    if cfg.duty:
        errors = np.column_stack([errors, state == SUSTAINED_LOAD])
        names.append(DUTY)
    return errors, names


def hold(condition: np.ndarray, monitorable: np.ndarray, k_hold: int,
         k_release: int) -> tuple[np.ndarray, np.ndarray]:
    """Open after k_hold consecutive eligible windows of `condition`; release after
    k_release consecutive eligible windows without it. Gaps reset the run."""
    if k_hold < 1 or k_release < 1:
        raise ValueError("k_hold and k_release must be >= 1")
    n = len(condition)
    active, opened = np.zeros(n, bool), np.zeros(n, bool)
    run = quiet = 0
    on = False
    for i in range(n):
        if not monitorable[i]:
            run = 0
        elif condition[i]:
            run, quiet = run + 1, 0
        else:
            run, quiet = 0, quiet + 1
        if monitorable[i]:
            if not on and run >= k_hold:
                on = opened[i] = True
            elif on and quiet >= k_release:
                on = False
        active[i] = on
    return active, opened


def evaluate(stream, cfg: ModalityConfig) -> tuple[dict, dict]:
    """Same eligibility, clean/event definitions and metrics as the AlarmPolicy harness."""
    test = np.flatnonzero(stream.test)
    values = channel_values(stream, cfg)
    baselines = calibrate(stream, cfg, values)
    errors, names = modality_errors(stream, test, cfg, baselines, values)
    condition = errors.sum(axis=1) >= cfg.min_modalities
    valid = stream.monitorable[test]
    active, opened = hold(condition, valid, cfg.k_hold, cfg.k_release)
    start, end = stream.start[test], stream.end[test]
    score = errors.sum(axis=1).astype(float)
    m, d = evaluate_alarms(start, end, stream.event_id()[test], stream.clean_negative()[test],
                           valid, stream.events, active, opened, score,
                           np.full(len(test), np.nan), (), stream.lps)
    seconds = float((end[valid] - start[valid]).astype(np.int64).sum())
    weeks = seconds / (7 * 86400) if seconds else float("nan")
    m["notifications_per_week_all"] = episodes(end[opened & valid]) / weeks if weeks else float("nan")
    event = stream.event_id()[test] >= 0
    labeled = valid & (event | stream.clean_negative()[test])
    if stream.metric_mask is not None:
        labeled &= stream.metric_mask[test]
    tp, fp = int((active & event & labeled).sum()), int((active & ~event & labeled).sum())
    fn, tn = int((~active & event & labeled).sum()), int((~active & ~event & labeled).sum())
    m["window_sensitivity"] = tp / (tp + fn) if tp + fn else float("nan")
    m["window_specificity"] = tn / (tn + fp) if tn + fp else float("nan")
    m["window_precision"] = tp / (tp + fp) if tp + fp else float("nan")
    window_minutes = float(np.median((end - start).astype(np.int64))) / 60
    m["hold_minutes"] = cfg.k_hold * window_minutes
    clean = stream.clean_negative()[test] & valid
    d["opening_modalities"] = _attribution(errors, names, opened & valid, event, clean)
    d["reference_windows"] = {str(k): v["windows"] for k, v in baselines.items()}
    d["judgeable_channels"] = {str(k): v["judgeable"] for k, v in baselines.items()}
    d["per_channel_level"] = baselines[GLOBAL]["per_channel_level"]
    d["clean_error_rate"] = {n: float(errors[clean, j].mean()) if clean.any() else None
                            for j, n in enumerate(names)}
    d["event_error_rate"] = {n: float(errors[event & valid, j].mean()) if (event & valid).any() else None
                            for j, n in enumerate(names)}
    d["window_confusion"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return m, d


def _attribution(errors, names, opened, event, clean) -> dict:
    """Which modalities were in error when each alarm opened, split by event/clean windows."""
    out = {"event": {}, "clean": {}, "other": {}}
    for i in np.flatnonzero(opened):
        key = "event" if event[i] else "clean" if clean[i] else "other"
        label = "+".join(n for n, e in zip(names, errors[i]) if e) or "none"
        out[key][label] = out[key].get(label, 0) + 1
    return out


def sustained_runs(stream) -> list[tuple[int, int]]:
    """(first, last) test indices of maximal runs of eligible sustained-load windows.

    Matches `hold` exactly for the duty modality alone: an unmonitorable window or any
    eligible window outside sustained load ends the run."""
    test = np.flatnonzero(stream.test)
    loaded = (stream.state[test] == SUSTAINED_LOAD) & stream.monitorable[test]
    runs, start = [], None
    for i, flag in enumerate(loaded):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(loaded) - 1))
    return runs


def duty_only_prediction(stream, k_hold: int) -> dict:
    """Closed-form duty-only alarm (k_release=1): a run of length L >= k is active on its last
    L - k + 1 windows. A logging gap resets the run count but does not release an open alarm,
    so a run that follows an active run across unmonitorable windows only is active from its
    first window. Detection, clean alarm time and first-alarm offset follow directly."""
    test = np.flatnonzero(stream.test)
    start, end = stream.start[test], stream.end[test]
    valid = stream.monitorable[test]
    active = np.zeros(len(test), bool)
    carry, previous_end = False, None
    for a, b in sustained_runs(stream):
        only_gap_between = previous_end is not None and not valid[previous_end + 1:a].any()
        first = a if carry and only_gap_between else a + k_hold - 1
        if first <= b:
            active[first:b + 1] = True
        carry, previous_end = first <= b, b
    clean = stream.clean_negative()[test] & valid
    dur = (end - start).astype(np.int64)
    event_id = stream.event_id()[test]
    detected, offsets = 0, []
    for e, (fs, fe) in enumerate(stream.events):
        detected += bool((active & valid & (event_id == e)).any())
        fs, fe = np.datetime64(fs, "s"), np.datetime64(fe, "s")
        early = active & valid & (end >= fs - np.timedelta64(24, "h")) & (end <= fe)
        if early.any():
            offsets.append(float((end[early].min() - fs).astype(np.int64)) / 3600)
    return {"events_detected": float(detected),
            "alarm_time_share_clean": float(dur[clean & active].sum() / dur[clean].sum()),
            "lead_hours": float(np.mean(offsets)) if offsets else None}
