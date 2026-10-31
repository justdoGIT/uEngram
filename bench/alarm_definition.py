"""What raises a uEngram alarm: the corrected, standards-based definition (v7).

An alarm on test window t requires ALL of (sources in docs/research-alarm-definition.md):

 1. Validity: t is eligible (monitorable and scored). Ineligible windows neither raise
    nor clear the alarm (ISA-18.2 bad-quality handling; AlarmPolicy state gating).
 2. uEngram error: novelty or health crosses its split-conformal setpoint for the current
    operating state, as AlarmPolicy counts it (new-state veto, health trust, GLOBAL
    fallback as configured). This is the alarm measure; nothing about it changes.
 3. Corroboration: at least m independent errors (threshold crossings) were seen in the
    last L hours. Each source is true, false or unknown; the vote is three-valued
    (Kleene): TRUE if #true >= m, FALSE if #true + #unknown < m, otherwise UNKNOWN.
 4. Persistence (on-delay): only a corroborated uEngram error raises AlarmPolicy's
    counter. A FALSE vote counts the window as normal; an UNKNOWN vote holds the counters
    (src/uengram/alarm.py `evidence`). The alarm opens when a counter reaches its k.
 5. Release (off-delay): the alarm clears when both counters are back at 0.
 6. Actionability (evaluated, not triggered): hours before the failure report and before
    the train's LPS low-pressure warning, positive = early. MetroPT practice targets at
    least 2 h before LPS (Jakobs, Veloso & Gama 2025).

Sources, one vote per independent physical cause:
  dataset     low_pressure  LPS active OR TP3 below 7 bar (the documented 7 bar
                            condition seen by the flag and by the analog sensor: one vote)
              oil_level     Oil_level departs from its calibration value
  calibrated  the 7 v5 per-channel crossings (bench/modality_alarm.py, alpha 0.05);
              unknown where the window's state has no reference or the value is missing

With every vote TRUE the definition reproduces AlarmPolicy exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bench.alarm_eval import apply_policy, eligible
from bench.error_gated_alarm import dataset_errors, frozen_scores, latch
from bench.modality_alarm import ModalityConfig, calibrate, channel_judgement
from uengram import metropt
from uengram.alarm import AlarmConfig, AlarmLogic


@dataclass(frozen=True)
class Corroboration:
    family: str = "dataset"  # "dataset", "physical" or "calibrated"
    min_errors: int = 1  # m
    lookback_h: float = 0.0  # L

    def label(self) -> str:
        return f"{self.family}:m{self.min_errors}:L{self.lookback_h:g}h"


def dataset_sources(stream, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(true, known, names) over test windows; digital/analog flags are always known."""
    E, names = dataset_errors(stream)
    E = E[test]
    low_pressure = E[:, names.index("lps")] | E[:, names.index("tp3_below_7bar")]
    true = np.column_stack([low_pressure, E[:, names.index("oil_level")]])
    return true, np.ones_like(true), ["low_pressure", "oil_level"]


CUT_IN_BAR = 8.2  # MetroPT documentation: MPG loads the compressor below 8.2 bar
IDLE = 0  # operating state: compressor idle or offloaded


def fill_windows(stream) -> int:
    """Healthy start-up length, label-free: the most windows any commissioning or
    calibration logging segment needed for TP3 to reach the 8.2 bar cut-in."""
    tp3_max = stream.W[:, list(metropt.ANALOG).index("TP3")].max(axis=1)
    starts = np.r_[0, np.flatnonzero(stream.segment[1:] != stream.segment[:-1]) + 1]
    ends, worst = np.r_[starts[1:], len(stream.segment)], 0
    for a, b in zip(starts, ends):
        if stream.fit[a] or stream.cal[a]:
            hit = np.flatnonzero(tp3_max[a:b] >= CUT_IN_BAR)
            worst = max(worst, int(hit[0]) + 1 if len(hit) else int(b - a))
    return worst


def physical_sources(stream, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """The dataset errors, judged only where the physics makes them meaningful.

    low_pressure  unknown only during a healthy start-up: within the first
                  `fill_windows` windows of a logging segment, before TP3 has reached the
                  8.2 bar cut-in, pressure below 7 bar is the reservoirs filling. Later,
                  low pressure without ever reaching cut-in is a failure to build pressure
                  and counts as an error
    oil_level     judged only while idle: under load the flag follows oil temperature and
                  churn, so a departure while running is unknown, not a low-oil error"""
    true, _, names = dataset_sources(stream, test)
    tp3_max = stream.W[:, list(metropt.ANALOG).index("TP3")].max(axis=1)
    fill = fill_windows(stream)
    starting = np.zeros(len(stream.start), bool)  # within a healthy fill, cut-in not reached
    pos, seen = 0, False
    for i in range(len(starting)):
        if i and stream.segment[i] != stream.segment[i - 1]:
            pos, seen = 0, False
        starting[i] = pos < fill and not seen
        pos, seen = pos + 1, seen or tp3_max[i] >= CUT_IN_BAR
    starting, idle = starting[test], stream.state[test] == IDLE
    low, oil = true[:, 0], true[:, 1]
    known = np.column_stack([~(low & starting), idle])
    return np.column_stack([low & ~starting, oil & idle]), known, names


def calibrated_sources(stream, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cfg = ModalityConfig(alpha=0.05, duty=False)  # the v5 sweep baseline
    errors, known = channel_judgement(stream, test, cfg, calibrate(stream, cfg))
    return errors, known, list(cfg.channels)


def vote(true: np.ndarray, known: np.ndarray, end: np.ndarray, valid: np.ndarray,
         c: Corroboration) -> np.ndarray:
    """Per test window: True / False / None (unknown) after an L-hour lookback.

    A source is true at t when it erred on an eligible window within L hours; otherwise it
    is false when judged at t and unknown when not."""
    seen = latch(true, end, valid, c.lookback_h)
    unknown = ~seen & ~(known & valid[:, None])
    n_true, n_unknown = seen.sum(axis=1), unknown.sum(axis=1)
    out = np.full(len(end), None, dtype=object)
    out[n_true >= c.min_errors] = True
    out[(n_true < c.min_errors) & (n_true + n_unknown < c.min_errors)] = False
    return out


def actionability(stream, test: np.ndarray, d: dict) -> dict:
    """Early warnings, positive = early. An event is warned when the alarm episode that
    first catches it (holds its first active eligible event window) opened within the
    24 h before the report start; an alarm that cleared before the failure is no warning.
    With published LPS times, the same episode must open at least 2 h before LPS."""
    end, event = stream.end[test], stream.event_id()[test]
    active, opened = d["active"] & d["valid"], np.flatnonzero(d["opened"] & d["valid"])
    early, before_lps = [], 0
    for e, (a, _) in enumerate(stream.events):
        hit = np.flatnonzero(active & (event == e))
        prior = opened[opened <= hit[0]] if len(hit) else opened[:0]
        if not len(prior):
            continue  # missed, or standing since before the test split
        fs, o = np.datetime64(a, "s"), end[prior[-1]]
        if fs - np.timedelta64(24, "h") <= o < fs:
            early.append(float((fs - o).astype(np.int64)) / 3600)
        if stream.lps is not None and stream.lps[e] is not None:
            before_lps += float((np.datetime64(stream.lps[e], "s") - o).astype(np.int64)) / 3600 >= 2.0
    out = {"events_warned_before_report": float(len(early)),
           "hours_before_report": float(np.mean(early)) if early else float("nan")}
    if stream.lps is not None:
        out["events_warned_2h_before_lps"] = float(before_lps)
    return out


class Definition:
    """One stream: frozen scores and the error sources of both families."""

    def __init__(self, stream):
        self.stream = stream
        self.scores = frozen_scores(stream)
        self.test = np.flatnonzero(stream.test)
        self.sources = {"dataset": dataset_sources(stream, self.test),
                        "physical": physical_sources(stream, self.test),
                        "calibrated": calibrated_sources(stream, self.test)}

    def evidence(self, cfg: AlarmConfig, c: Corroboration) -> np.ndarray:
        """The vote on every test window; errors count only on windows the policy may use."""
        true, known, _ = self.sources[c.family]
        valid = eligible(self.stream, self.scores, cfg, self.test)
        return vote(true, known, self.stream.end[self.test], valid, c)

    def evaluate(self, cfg: AlarmConfig, c: Corroboration | None, logic: AlarmLogic | None = None):
        """(metrics, details, policy) of the alarm; c None = uncorroborated AlarmPolicy;
        logic None = the v2 condition logic."""
        evidence = None if c is None else self.evidence(cfg, c)
        m, d, pol = apply_policy(self.stream, self.scores, cfg, evidence, logic)
        m.update(actionability(self.stream, self.test, d))
        if evidence is not None:
            valid = d["valid"]
            clean = self.stream.clean_negative()[self.test] & valid
            d["vote_share_clean"] = {k: float(np.mean([e is v for e in evidence[clean]]))
                                     for k, v in (("true", True), ("false", False), ("unknown", None))}
            event = self.stream.event_id()[self.test]
            d["corroboration_ceiling_events"] = len({int(e) for e, v in zip(event[valid], evidence[valid])
                                                     if e >= 0 and v is True})
        return m, d, pol


def event_trace(definition: "Definition", cfg: AlarmConfig, c: Corroboration | None,
                logic: AlarmLogic | None = None) -> list[dict]:
    """When each condition was first met around every event, in hours from report start
    (negative = before the report), over [report start - 24 h, report end]. With no
    corroboration rule every window counts as corroborated."""
    s, t = definition.stream, definition.test
    evidence = None if c is None else definition.evidence(cfg, c)
    _, d, pol = apply_policy(s, definition.scores, cfg, evidence, logic)
    valid, end = d["valid"], s.end[t]
    crossing = pol.exceeded & valid  # condition 2 exactly as the policy counted it
    corroborated = valid.copy() if evidence is None else np.array([v is True for v in evidence]) & valid
    out = []
    for e, (a, b) in enumerate(s.events):
        fs, fe = np.datetime64(a, "s"), np.datetime64(b, "s")
        span = valid & (end >= fs - np.timedelta64(24, "h")) & (end <= fe)

        def first(mask):
            hit = span & mask
            return float((end[hit].min() - fs).astype(np.int64)) / 3600 if hit.any() else None
        lps = None if s.lps is None or s.lps[e] is None else \
            float((np.datetime64(s.lps[e], "s") - fs).astype(np.int64)) / 3600
        out.append({"event": e, "report_hours": float((fe - fs).astype(np.int64)) / 3600,
                    "lps_warning_h": lps, "first_uengram_error_h": first(crossing),
                    "first_corroboration_h": first(corroborated),
                    "first_corroborated_error_h": first(crossing & corroborated),
                    "alarm_opened_h": first(d["opened"]), "first_alarm_h": first(d["active"]),
                    "standing_at_window_start": bool((span & d["active"]).any()) and first(d["opened"]) is None})
    return out
