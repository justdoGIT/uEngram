"""Hybrid temporal-vector-severity ranking of stored sensor events.

    S(q, m) = w_s * sim(q, v_m) + w_t * exp(-lambda * dt) + w_r * g(r_m)

for a query latent q and a stored event m with latent snapshot v_m, age dt and severity r_m. The
three score components are returned next to the total, so a caller can show why an event ranked
where it did. Asset, subsystem, operating mode and feature schema are strict filters, never
score terms: an event from another identity is not ranked lower, it is not returned.

  sim(q, v)   exp(-||q - v||^2 / (2 d)): 1 for identical latents, falling with the squared
              distance per dimension (latents are whitened, so unit variance per dimension)
  recency     2 ** (-dt / half_life_ms), half-life form of exp(-lambda * dt)
  g(r)        tanh(max(r, 0) / severity_scale); an event without a severity scores 0

An event lacking a field a component needs (no stored latent, no timestamp, no severity)
scores 0 on that component and lists it in ``missing``; it is never dropped for that reason.
"""
from __future__ import annotations

import base64
import json
import math
from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence

import numpy as np

_DAY_MS = 86_400_000


@dataclass(frozen=True)
class RetrievalWeights:
    similarity: float = 1.0
    recency: float = 0.5
    severity: float = 0.5
    half_life_ms: float = 7 * _DAY_MS
    severity_scale: float = 1.0

    def __post_init__(self) -> None:
        for name in ("similarity", "recency", "severity"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} weight must be finite and non-negative")
        if self.similarity + self.recency + self.severity <= 0:
            raise ValueError("at least one weight must be positive")
        for name in ("half_life_ms", "severity_scale"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class RankedEvent:
    event: object
    score: float
    similarity: float
    recency: float
    severity: float
    missing: tuple[str, ...] = field(default_factory=tuple)


def encode_vector(vector: np.ndarray) -> dict:
    """Compact int8 snapshot of a latent: {"q": base64, "s": scale}; per-element error <= s / 254."""
    v = np.asarray(vector, dtype=np.float64)
    scale = float(f"{max(float(np.abs(v).max()), 1e-12):.4g}")
    codes = np.clip(np.rint(v / scale * 127.0), -127, 127).astype(np.int8)
    return {"q": base64.b64encode(codes.tobytes()).decode("ascii"), "s": scale}


def decode_vector(snapshot: object) -> np.ndarray | None:
    if not isinstance(snapshot, Mapping):
        return None
    try:
        codes = np.frombuffer(base64.b64decode(snapshot["q"], validate=True), dtype=np.int8)
        scale = float(snapshot["s"])
    except (KeyError, TypeError, ValueError):
        return None
    if codes.size == 0 or not math.isfinite(scale) or scale <= 0:
        return None
    return codes.astype(np.float64) * scale / 127.0


def _metadata(item: object) -> Mapping | None:
    raw = item.get("structured_value") if isinstance(item, Mapping) else getattr(item, "structured_value", None)
    try:
        meta = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    except ValueError:
        return None
    return meta if isinstance(meta, Mapping) else None


def score_event(meta: Mapping, query: np.ndarray, now_ms: int, w: RetrievalWeights):
    """(similarity, recency, severity, missing) for one event's metadata."""
    missing = []
    stored = decode_vector(meta.get("vec"))
    if stored is None or stored.shape != query.shape:
        similarity = 0.0
        missing.append("vector")
    else:
        similarity = math.exp(-float(np.sum((query - stored) ** 2)) / (2.0 * query.size))
    t_ms = meta.get("t_ms")
    if isinstance(t_ms, bool) or not isinstance(t_ms, (int, float)) or not math.isfinite(t_ms):
        recency = 0.0
        missing.append("time")
    else:
        recency = 2.0 ** (-max(float(now_ms) - float(t_ms), 0.0) / w.half_life_ms)
    severity_value = meta.get("severity")
    if isinstance(severity_value, bool) or not isinstance(severity_value, (int, float)) \
            or not math.isfinite(severity_value):
        severity = 0.0
        missing.append("severity")
    else:
        severity = math.tanh(max(float(severity_value), 0.0) / w.severity_scale)
    return similarity, recency, severity, tuple(missing)


def rank_events(
    events: Sequence[object],
    query_vector: np.ndarray,
    now_ms: int,
    weights: RetrievalWeights | None = None,
    *,
    asset_id: str,
    subsystem_id: str,
    operating_mode: str | None = None,
    feature_schema: str | None = None,
    limit: int = 16,
) -> list[RankedEvent]:
    """Rank stored events by the hybrid score, best first; ties break on newer time, then input order."""
    weights = weights or RetrievalWeights()
    query = np.asarray(query_vector, dtype=np.float64)
    if query.ndim != 1 or query.size == 0 or not np.isfinite(query).all():
        raise ValueError("query_vector must be a non-empty finite 1-D vector")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    required = {"asset": asset_id, "subsystem": subsystem_id}
    if operating_mode is not None:
        required["mode"] = operating_mode
    if feature_schema is not None:
        required["feature_schema"] = feature_schema
    ranked = []
    for order, item in enumerate(events):
        meta = _metadata(item)
        if meta is None or any(meta.get(key) != value for key, value in required.items()):
            continue
        sim, rec, sev, missing = score_event(meta, query, now_ms, weights)
        total = weights.similarity * sim + weights.recency * rec + weights.severity * sev
        t_ms = meta.get("t_ms")
        ranked.append((-total, -(t_ms if isinstance(t_ms, (int, float)) else 0), order,
                       RankedEvent(item, total, sim, rec, sev, missing)))
    ranked.sort(key=lambda row: row[:3])
    return [row[3] for row in ranked[:limit]]
