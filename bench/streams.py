"""Benchmark streams for alarm protocol v2 (docs/EXPLAINED.md section 17).

A Stream carries everything the v2 harness needs besides scores: model input
windows, time bounds, logging segments, a discrete operating state, which
windows the alarm counters may use, label-free snapshot features for freeze
frames, the commissioning / calibration / test split and labeled events.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from uengram import metropt

# Compressor operating state from the share of the window the compressor runs
# loaded (DV_eletric active): idle/offloaded, cycling, sustained load.
LOAD_EDGES = (0.05, 0.4)
DIGITAL = ("COMP", "DV_eletric", "MPG", "LPS")
COMPRESSOR_FEATURES = ("load_share", "comp_switches", "lps_share", "motor_current_a",
                       "oil_temp_c", "tp2_bar", "tp3_bar", "h1_bar", "dv_pressure_bar",
                       "reservoirs_min_bar")


@dataclass
class Stream:
    name: str
    W: np.ndarray  # model input: [N, C, K] sample windows or [N, F] tabular rows
    start: np.ndarray  # datetime64[s]
    end: np.ndarray
    segment: np.ndarray  # changes at every logging gap
    state: np.ndarray  # int operating state
    monitorable: np.ndarray  # bool: alarm counters may move
    fit: np.ndarray  # commissioning: encoder, memory warm-up, healthy reference
    cal: np.ndarray  # label-free threshold calibration
    test: np.ndarray
    events: tuple  # ((start, end), ...) labeled event intervals, ISO strings
    raw_bytes: int  # bytes of one raw window (pinned exemplars)
    features: np.ndarray  # [N, f] snapshot features
    feature_names: tuple
    cond: np.ndarray | None = None  # [N, Cs, K] digital state channels
    lps: tuple | None = None  # per-event low-pressure warning time (ISO) or None
    before_h: int = 24  # clean negatives exclude [event start - before_h,
    after_h: int = 48  # event end + after_h]
    extra: dict = field(default_factory=dict)
    point_labels: bool = False  # SCADA annotations identify row times, not input intervals
    metric_mask: np.ndarray | None = None  # evaluation annotations; never prediction feedback

    def _bounds(self):
        fs = np.array([a for a, _ in self.events], dtype="datetime64[s]")
        fe = np.array([b for _, b in self.events], dtype="datetime64[s]")
        return fs, fe

    def event_id(self) -> np.ndarray:
        """Event membership: interval overlap, or exact end time for SCADA rows."""
        out = np.full(len(self.start), -1, dtype=np.int64)
        for i, (a, b) in enumerate(zip(*self._bounds())):
            label_start = self.end if self.point_labels else self.start
            out[(label_start <= b) & (self.end >= a)] = i
        return out

    def clean_negative(self) -> np.ndarray:
        ok = np.ones(len(self.start), dtype=bool)
        for a, b in zip(*self._bounds()):
            label_start = self.end if self.point_labels else self.start
            ok &= ((self.end < a - np.timedelta64(self.before_h, "h"))
                   | (label_start > b + np.timedelta64(self.after_h, "h")))
        return ok


def compressor_features(W, D, analog=metropt.ANALOG) -> np.ndarray:
    """Interpretable per-window values: digital shares, analog means/minimum."""
    a = {c: W[:, i] for i, c in enumerate(analog)}
    d = {c: D[:, i] for i, c in enumerate(DIGITAL)}
    return np.column_stack([
        d["DV_eletric"].mean(1), np.abs(np.diff(d["COMP"], axis=1)).sum(1), d["LPS"].mean(1),
        a["Motor_current"].mean(1), a["Oil_temperature"].mean(1), a["TP2"].mean(1),
        a["TP3"].mean(1), a["H1"].mean(1), a["DV_pressure"].mean(1), a["Reservoirs"].min(1),
    ]).astype(np.float64)


def compressor_stream(name, t, X, cols, K, fit_end, train_end, events, lps=None,
                      analog=metropt.ANALOG) -> Stream:
    """Gap-aware windows of a Metro do Porto APU log (MetroPT-3, MetroPT2)."""
    W, S, E, G = metropt.windows(t, X, cols, K, channels=analog)
    D = metropt.windows(t, X, cols, K, channels=DIGITAL)[0]
    feats = compressor_features(W, D, analog)
    state = np.digitize(feats[:, 0], LOAD_EDGES).astype(np.int64)
    # The first window after a logging gap starts mid-cycle from an unknown
    # state; its novelty must not move the alarm counters.
    monitorable = np.r_[False, G[1:] == G[:-1]]
    fit = E < np.datetime64(fit_end)
    cal = (S >= np.datetime64(fit_end)) & (E < np.datetime64(train_end))
    test = S >= np.datetime64(train_end)
    return Stream(name, W, S, E, G, state, monitorable, fit, cal, test, tuple(events),
                  len(analog) * K * 4, feats, COMPRESSOR_FEATURES, cond=D,
                  lps=None if lps is None else tuple(lps))


def metropt3_stream(K: int = 60) -> Stream:
    """MetroPT-3 (development set since v1 was scored): same splits as v1."""
    t, X, cols = metropt.load(metropt.fetch())
    return compressor_stream("metropt3", t, X, cols, K, metropt.FIT_END, metropt.TRAIN_END,
                             metropt.FAILURES)


def metropt2_stream(name: str, K: int = 60) -> Stream:
    """MetroPT2 confirmation streams (uengram.metropt2): fresh, never scored."""
    from uengram import metropt2

    spec = metropt2.STREAMS[name]
    t, X, cols = metropt2.load(name)
    s = compressor_stream(name, t, X, cols, K, spec["fit_end"], spec["train_end"],
                          spec["events"], spec["lps"])
    # Share of each window spent stopped (speed 0, GPS fix) inside a workshop or
    # parking polygon: the no-demand context of the compressor leak test.
    lat, lon, speed, quality = (metropt.windows(t, X, cols, K, channels=(c,))[0][:, 0]
                                for c in ("gpsLat", "gpsLong", "gpsSpeed", "gpsQuality"))
    parked = (speed == 0) & (quality > 0) & metropt2.in_park(lat, lon)
    s.extra["parked_share"] = parked.mean(1)
    return s


def load_stream(name: str) -> Stream:
    if name == "metropt3":
        return metropt3_stream()
    if name.startswith("metropt2_"):
        return metropt2_stream(name)
    if name.startswith("care_"):
        from bench.stream_care import care_stream
        return care_stream(name)
    raise ValueError(f"unknown stream {name}")
