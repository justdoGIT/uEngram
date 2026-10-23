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
from uengram.alarm import AlarmConfig


@dataclass(frozen=True)
class Corroboration:
    family: str = "dataset"  # "dataset" or "calibrated"
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


def actionability(stream, d: dict) -> dict:
    """Early-is-positive summaries of evaluate_alarms' per-event first alarm."""
    early = [-e["lead_hours"] for e in d["events"] if e["lead_hours"] is not None and e["lead_hours"] < 0]
    out = {"events_warned_before_report": float(len(early)),
           "hours_before_report": float(np.mean(early)) if early else float("nan")}
    if stream.lps is not None:
        out["events_warned_2h_before_lps"] = float(sum(
            (e.get("hours_before_lps") or 0.0) >= 2.0 for e in d["events"]))
    return out


class Definition:
    """One stream: frozen scores and the error sources of both families."""

    def __init__(self, stream):
        self.stream = stream
        self.scores = frozen_scores(stream)
        self.test = np.flatnonzero(stream.test)
        self.sources = {"dataset": dataset_sources(stream, self.test),
                        "calibrated": calibrated_sources(stream, self.test)}

    def evidence(self, cfg: AlarmConfig, c: Corroboration) -> np.ndarray:
        """The vote on every test window; errors count only on windows the policy may use."""
        true, known, _ = self.sources[c.family]
        valid = eligible(self.stream, self.scores, cfg, self.test)
        return vote(true, known, self.stream.end[self.test], valid, c)

    def evaluate(self, cfg: AlarmConfig, c: Corroboration | None):
        """(metrics, details, policy) of the alarm; c None = uncorroborated AlarmPolicy."""
        evidence = None if c is None else self.evidence(cfg, c)
        m, d, pol = apply_policy(self.stream, self.scores, cfg, evidence)
        m.update(actionability(self.stream, d))
        if evidence is not None:
            valid = d["valid"]
            clean = self.stream.clean_negative()[self.test] & valid
            d["vote_share_clean"] = {k: float(np.mean([e is v for e in evidence[clean]]))
                                     for k, v in (("true", True), ("false", False), ("unknown", None))}
            event = self.stream.event_id()[self.test]
            d["corroboration_ceiling_events"] = len({int(e) for e, v in zip(event[valid], evidence[valid])
                                                     if e >= 0 and v is True})
        return m, d, pol


def event_trace(definition: "Definition", cfg: AlarmConfig, c: Corroboration) -> list[dict]:
    """When each condition was first met around every event, in hours from report start
    (negative = before the report), over [report start - 24 h, report end]."""
    s, t = definition.stream, definition.test
    evidence = definition.evidence(cfg, c)
    _, d, pol = apply_policy(s, definition.scores, cfg, evidence)
    valid, end = d["valid"], s.end[t]
    crossing = pol.exceeded & valid  # condition 2 exactly as the policy counted it
    corroborated = np.array([v is True for v in evidence]) & valid
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
