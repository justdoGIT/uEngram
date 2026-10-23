"""One node's memory: encoder + uEngram engine + error-counter test (numpy).

Commissioning (declared healthy run of ticks):
* the first (1 - calib_frac) of the run fits the encoder and the event model
  and seeds the memory. Because the run is declared normal, any state seen in
  `m_reconfirm` windows anywhere in it becomes semantic (normal); afterwards
  promotion reverts to the engine's sustained-dwell rule.
* the tail is fed as deployed and its normal-state scores set the alarm
  threshold by split-conformal calibration (uEngram v2, docs/EXPLAINED.md 7b).
  Windows are non-overlapping, so the engine's sequential statistics
  (M-of-N promotion, MEWMA step test) keep their meaning; for fitting and
  calibration the run is also cut at 4 phase offsets (K/4 apart), giving four
  non-overlapping sequences instead of one overlapping one.
* alpha is split between the Gaussian latent and the error counters
  (Bonferroni), and inside the counters between the counters.
* the sorted calibration scores are kept, so every window also gets a
  conformal p-value (uengram.host.validate checks their calibration live).
* uEngram v3: the step detector is calibrated to the time correlation of
  the commissioning innovations (stepcal.py), and the normal states learned
  from the declared-healthy fit part are frozen as a healthy reference; every
  window gets its health score, the D2 to the nearest frozen state, which
  keeps growing with slow drift that the adaptive memory absorbs.
"""
from __future__ import annotations

import copy
import json
import os
import pickle
import socket
from collections import deque
from dataclasses import dataclass

import numpy as np

from ..engine import Engram, EngramConfig
from ..lifecycle import CAND, EPI, SEM, SUPER
from ..stats import conformal_threshold, surprisal
from . import schema, stepcal
from .encoder import HostEncoder
from .events import EventModel
from .features import Columns, kernel_releases, select, tick_columns, windows, windows_from_columns
from .psi import fast_bounds


@dataclass(frozen=True)
class Options:
    """Model switches. Defaults are the deployed model; the others exist so
    uengram.host.validate can measure what each part contributes."""
    on_delay: int = 2  # consecutive novel windows before a latent alarm (1 = none)
    use_thermal: bool = True  # thermal RC residual feature
    use_spe: bool = True  # out-of-subspace (Q) coordinate
    use_events: bool = True  # Poisson-Gamma test on error counters
    use_saturation: bool = True  # pressure beyond commissioning
    use_step_calibration: bool = True  # step detector scaled to correlated innovations (uEngram v3)


PHASES = 4
PRESSURES = ("cpu_pressure", "mem_pressure", "io_pressure")
SAT_FLOOR = 0.01  # 1 % of the time stalled is never flagged
KIND_NAME = {CAND: "candidate", EPI: "episodic", SEM: "semantic", SUPER: "superseded"}
LEVELS = {"cpu_busy": "angular", "cpu_pressure": "angular", "mem_used": "angular",
          "mem_pressure": "angular", "io_pressure": "angular", "ipc": "log", "freq_ratio": "log",
          "llc_mpki": "log", "pkg_power": "log", "thermal_residual": "raw"}


def _natural(name: str, v: float) -> float:
    kind = LEVELS[name]
    if not np.isfinite(v):
        return float("nan")
    return float(np.sin(v) ** 2 if kind == "angular" else np.exp(v) if kind == "log" else v)


def split_columns(cols: Columns, calib_frac: float, K: int,
                  exclude: tuple[tuple[float, float], ...] = ()) -> tuple[Columns, Columns, float]:
    """Drop excluded intervals, then cut at the phase-0 window boundary that
    leaves the last `calib_frac` of the remaining ticks for calibration (a
    share of the data, not of the time span: a node that was down for hours
    would otherwise leave an empty calibration part)."""
    t = cols["t"]
    keep = np.ones(len(t), bool)
    for lo, hi in exclude:
        keep &= ~((t >= lo) & (t <= hi))
    cols, t = select(cols, keep), t[keep]
    split = float(np.floor(t[int((1.0 - calib_frac) * (len(t) - 1))] / K) * K)
    return select(cols, t < split), select(cols, t >= split), split


class NodeMemory:
    def __init__(self, enc: HostEncoder, eng: Engram, ev: EventModel, K: int, t_origin: float,
                 alpha: float, meta: dict, options: Options = Options()):
        self.enc, self.eng, self.ev, self.K = enc, eng, ev, K
        self.t_origin, self.alpha, self.meta, self.opt = t_origin, alpha, meta, options
        self.host = meta.get("host", socket.gethostname())
        self.thermal_state = None
        self.profile: dict[int, np.ndarray] = {}  # item id -> [windows, energy J, instructions, seconds]
        self.windows_seen = 0
        self.gaps = 0
        self._buf: list[dict] = []
        self.commission_report: dict = {}
        self.alarm_threshold = float("inf")
        self.calib = np.empty(0)  # sorted calibration (persistent) scores
        self._recent: deque = deque(maxlen=options.on_delay)  # (window step, score, health)
        self.alarmed_states: dict[int, float] = {}  # item id -> window start of its first alarm
        self.sat_bound = {n: SAT_FLOOR for n in PRESSURES}
        self.fast_bound = {n: SAT_FLOOR for n in PRESSURES}  # 2 s shares, for psi.PsiTriggers

    # -- commissioning ---------------------------------------------------
    @classmethod
    def commission(cls, ticks: list[dict] | Columns, meta: dict, window: int = 32, alpha: float = 0.01,
                   calib_frac: float = 0.3, d: int = 16, B_sem: int = 192, B_epi: int = 64,
                   options: Options = Options(),
                   exclude: tuple[tuple[float, float], ...] = ()) -> "NodeMemory":
        """`ticks`: tick dicts or columns (features.read_columns). `exclude`:
        wall-time intervals left out, e.g. injected faults, which must not be
        learned as healthy."""
        K = window
        cols = ticks if isinstance(ticks, dict) else tick_columns(ticks)
        releases = kernel_releases(cols)
        if len(releases) > 1:
            raise ValueError(f"commissioning data contains multiple kernel releases: {releases}; "
                             "select one release before commissioning")
        model_meta = dict(meta)
        if releases:
            model_meta["kernel"] = releases[0]
        A, B, split = split_columns(cols, calib_frac, K, exclude)
        t = np.concatenate([A["t"], B["t"]])
        offsets = [k * K / PHASES for k in range(PHASES)]
        seqA = [windows_from_columns(A, K, off) for off in offsets]
        seqB = [windows_from_columns(B, K, off) for off in offsets]
        enc = HostEncoder(d, K, options.use_thermal, options.use_spe).fit(seqA)
        ev = EventModel().fit(seqA[0].events)  # one phase: overlapping phases would count events 4 times
        cfg = EngramConfig(B_sem=B_sem, B_epi=B_epi)
        raw_bytes = 4 * len(schema.TICK_FIELDS) * K
        n_reconfirm = cfg.n_reconfirm
        cfg.n_reconfirm = len(seqA[0]) + 1  # declared-normal run: recurrence anywhere promotes
        t_origin = float(seqA[0].t0[0])
        inflation, stepcal_report = None, {"estimator": "off"}
        if options.use_step_calibration:
            Z, _, _, _ = enc.encode(seqA[0])
            steps = np.floor((seqA[0].t0 - t_origin) / K + 1e-9).astype(np.int64)
            inflation, stepcal_report = stepcal.calibrate(Z, steps, cfg, enc.sigma0, raw_bytes)
        eng = Engram(cfg, d, enc.sigma0, raw_bytes, name=f"uengram-{model_meta.get('host', '')}",
                     step_inflation=inflation)
        node = cls(enc, eng, ev, K, t_origin, alpha, model_meta, options)
        node._ingest(seqA[0])
        cfg.n_reconfirm = n_reconfirm
        eng.set_reference(*eng.reference())
        node.thermal_state = None
        node._recent.clear()
        copies = [copy.deepcopy(node) for _ in offsets[1:]]
        recs = node._ingest(seqB[0])
        for c, seq in zip(copies, seqB[1:]):
            recs += c._ingest(seq)
        persist = np.array([r["persistent_score"] for r in recs], dtype=np.float64)
        m, need = int(np.isfinite(persist).sum()), int(np.ceil(2.0 / alpha)) - 1
        if m < need:  # the conformal quantile would be +inf: alarms silently off
            raise ValueError(f"{m} calibration scores; alpha={alpha} needs at least {need}. "
                             f"Record longer or raise alpha (latent alpha is alpha/2).")
        node.alarm_threshold = conformal_threshold(persist, alpha / 2)
        node.calib = np.sort(persist[np.isfinite(persist)])
        # Saturation flags name the resource under contention. D2 ranks the most
        # *surprising* measurements, and on a quiet node any job moves power and
        # clock first (measured: a 39 % memory stall on hp got 1 % of D2), so the
        # USE saturation signals are also checked against the largest value of
        # the declared-healthy run (a rank bound at level 1/(m+1) per resource).
        idx = [enc.names.index(n) for n in PRESSURES]
        seen = np.sin(np.vstack([seqA[0].F[:, idx], seqB[0].F[:, idx]])) ** 2
        node.sat_bound = {n: float(max(np.nanmax(seen[:, j]), SAT_FLOOR)) for j, n in enumerate(PRESSURES)}
        node.fast_bound = fast_bounds({k: np.concatenate([A[k], B[k]]) for k in A}, SAT_FLOOR)
        single = np.array([r["score"] for r in recs], dtype=np.float64)
        node.commission_report = {
            "host": node.host, "ticks": int(len(t)), "window_s": K, "t_start": float(t.min()),
            "t_split": float(split), "t_end": float(t.max()), "excluded": [list(x) for x in exclude],
            "fit_windows": len(seqA[0]), "calibration_scores": m,
            "alarm_threshold_d2": node.alarm_threshold, "alarm_alpha_latent": alpha / 2,
            "single_window_q": float(np.nanquantile(single, 1 - alpha / 2)),
            "encoder": enc.fit_report, "events": ev.report(), "slots": node.slot_counts(),
            "calibration_event_alarm_share": float(np.mean([r["alarm_event"] for r in recs])),
            "saturation_bounds": node.sat_bound, "fast_bounds_2s": node.fast_bound,
            "step_calibration": stepcal_report, "reference_states": int(len(eng.ref_mu)),
            "options": options.__dict__,
        }
        return node

    def p_latent(self, score: float) -> float:
        """Split-conformal p-value of a persistent score against the calibration scores."""
        if not np.isfinite(score) or len(self.calib) == 0:
            return float("nan")
        m = len(self.calib)
        return float((1 + m - np.searchsorted(self.calib, score, side="left")) / (m + 1))

    # -- streaming -------------------------------------------------------
    def feed_ticks(self, ticks: list[dict]):
        """Buffer ticks; yield one record per completed window (wall-aligned K seconds)."""
        for r in ticks:
            if self._buf and np.floor(r["t"] / self.K) != np.floor(self._buf[0]["t"] / self.K):
                self._buf = []
                self.gaps += 1
            self._buf.append(r)
            if len(self._buf) == self.K:
                batch = windows(self._buf, self.K)
                self._buf = []
                yield from self._ingest(batch)

    def _ingest(self, seq) -> list[dict]:
        Z, X, n_imp, self.thermal_state = self.enc.encode(seq, self.thermal_state)
        out = []
        for i in range(len(seq)):
            out.append(self._one(seq, i, Z[i], X[i], int(n_imp[i])))
        return out

    def _nearest_semantic(self, z):
        idx = np.flatnonzero(self.eng.kind == SEM)
        if len(idx) == 0:
            return None
        var = self.eng._pred_var(idx)
        d2 = np.sum((z - self.eng.mu[idx]) ** 2 / var, axis=1)
        j = int(np.argmin(d2))
        return idx[j], var[j], float(d2[j])

    def _one(self, seq, i: int, z, x, n_imp: int) -> dict:
        eng = self.eng
        # phase-shifted sequences (offsets K/4..3K/4) map to the step of the
        # phase-0 window they start in, so consecutive windows stay consecutive
        t_step = int(np.floor((seq.t0[i] - self.t_origin) / self.K + 1e-9))
        ref = self._nearest_semantic(z)
        contrib = self.enc.attribution(x, z, eng.mu[ref[0]], ref[1]) if ref else None
        res = eng.ingest(z, t_step, raw_idx=t_step)
        self.windows_seen += 1
        # On-delay: a window that straddles a job start/stop is a mixture of two
        # known states and novel on its own; a fault persists. The persistent
        # score is the smallest of the last `on_delay` consecutive windows. The
        # same holds for the distance to the healthy reference.
        if self._recent and self._recent[-1][0] != t_step - 1:
            self._recent.clear()
        self._recent.append((t_step, res.score, res.health))
        persist = persist_health = float("nan")
        if len(self._recent) == self.opt.on_delay:  # NaN stays NaN
            persist = float(np.min([s for _, s, _ in self._recent]))
            persist_health = float(np.min([h for _, _, h in self._recent]))
        p = self.ev.pvalues(seq.events[i])
        p_min = float(np.nanmin(p)) if np.isfinite(p).any() else 1.0
        alarm_event = self.opt.use_events and p_min < self.alpha / 2 / len(p)
        self._update_profile(res.item_id, seq, i)
        rec = {
            "host": self.host, "t0": float(seq.t0[i]), "item_id": res.item_id, "action": res.action,
            "kind": self._kind_of(res.item_id), "score": res.score, "persistent_score": persist,
            "surprisal": surprisal(res.score, self.eng.d) if np.isfinite(res.score) else None,
            "threshold": self.alarm_threshold, "alarm_latent": bool(persist > self.alarm_threshold),
            "p_latent": self.p_latent(persist),
            "alarm_event": bool(alarm_event), "event_p_min": p_min,
            "events": {n: int(c) for n, c in zip(self.ev.names, seq.events[i]) if np.isfinite(c) and c > 0},
            "levels": {n: _natural(n, seq.F[i, self.enc.names.index(n)]) for n in LEVELS
                       if n != "thermal_residual"},
            "outcomes": {k: float(v[i]) for k, v in seq.outcomes.items()},
            "quality": {"n_imputed": n_imp, **{k: float(v[i]) for k, v in seq.quality.items()}},
        }
        rec["levels"]["thermal_residual"] = float(x[self.enc.i_thermal] * self.enc.scale[self.enc.i_thermal]
                                                 + self.enc.med[self.enc.i_thermal])
        rec["alarm"] = rec["alarm_latent"] or rec["alarm_event"]
        rec["saturation"] = {n: rec["levels"][n] for n in PRESSURES
                             if self.opt.use_saturation and rec["levels"][n] > self.sat_bound[n]}
        # uEngram promotes a persistent new state to normal (5 of 8 windows), so a
        # long fault stops alarming once learned. The memory keeps when a state
        # was first seen alarming, and reports it on every later visit.
        if rec["alarm_latent"] and res.item_id not in self.alarmed_states:
            self.alarmed_states[res.item_id] = rec["t0"]
        rec["state_alarmed_at"] = self.alarmed_states.get(res.item_id)
        if contrib is not None:
            mods = {m: float(contrib[self.enc.modality == m].sum()) for m in schema.MODALITIES}
            rec["modality_d2"] = mods
            top = np.argsort(-contrib)[:3]
            rec["top_features"] = [[self.enc.names[j], float(contrib[j])] for j in top]
        # Health (uEngram v3, EXPLAINED 7c): D2 to the nearest state of the frozen
        # healthy reference, with the same attribution. Not an alarm: a new but
        # legitimate workload also moves it; it ranks slow drift that the memory
        # has absorbed (a leak, rising thermal resistance) above routine windows.
        rec["health"], rec["persistent_health"] = res.health, persist_health
        if len(eng.ref_mu):
            j = int(np.argmin(np.sum((z - eng.ref_mu) ** 2 / eng.ref_var, axis=1)))
            ch = self.enc.attribution(x, z, eng.ref_mu[j], eng.ref_var[j])
            rec["health_modality_d2"] = {m: float(ch[self.enc.modality == m].sum()) for m in schema.MODALITIES}
        for ev in eng.drain_events():
            self._apply_event(ev)
        return rec

    # -- per-state efficiency profile ----------------------------------------
    def _update_profile(self, iid: int, seq, i: int) -> None:
        dt = seq.quality["dt"][i]
        energy = seq.outcomes["pkg_watts"][i] * dt
        inst = seq.outcomes["ips"][i] * dt
        if not (np.isfinite(energy) and np.isfinite(inst)):
            return
        row = self.profile.setdefault(iid, np.zeros(4))
        row += (1.0, energy, inst, dt)

    def _apply_event(self, ev: tuple) -> None:
        if ev[0] == "merge":
            src = self.profile.pop(ev[1], None)
            if src is not None:
                self.profile[ev[2]] = self.profile.get(ev[2], np.zeros(4)) + src
            t_src = self.alarmed_states.pop(ev[1], None)
            if t_src is not None:  # conservative: a merge with an alarmed state stays alarmed
                self.alarmed_states[ev[2]] = min(t_src, self.alarmed_states.get(ev[2], t_src))
        elif ev[0] == "evict":
            self.profile.pop(ev[1], None)
            self.alarmed_states.pop(ev[1], None)

    def state_profile(self, iid: int) -> dict | None:
        row = self.profile.get(iid)
        if row is None or row[2] <= 0:
            return None
        return {"windows": int(row[0]), "watts": float(row[1] / row[3]),
                "nj_per_inst": float(row[1] / row[2] * 1e9), "gips": float(row[2] / row[3] / 1e9)}

    # -- views, persistence ----------------------------------------------------
    def _kind_of(self, iid: int) -> str:
        idx = np.flatnonzero(self.eng.item_id == iid)
        return KIND_NAME.get(int(self.eng.kind[idx[0]]), "empty") if len(idx) else "gone"

    def slot_counts(self) -> dict:
        return {name: int(np.sum(self.eng.kind == k)) for k, name in KIND_NAME.items()}

    def state(self, rec: dict) -> dict:
        """The node's current state: its latest window record plus the memory's view."""
        return {**rec, "slots": self.slot_counts(), "state_profile": self.state_profile(rec["item_id"]),
                "windows_seen": self.windows_seen, "gaps": self.gaps}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    @staticmethod
    def load(path: str) -> "NodeMemory":
        with open(path, "rb") as f:
            return pickle.load(f)
