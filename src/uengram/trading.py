"""Bounded market-regime and risk monitoring built on uEngram.

This module deliberately does not predict direction or place orders.  It turns
mark-price, funding, and aggressor-volume observations into a fixed-width feature
vector, learns a bounded commissioning memory, and returns a conservative
``allow``/``reduce``/``halt`` decision for a separate execution policy.
"""
from __future__ import annotations

import math
import struct
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .engine import Engram, EngramConfig
from .stats import conformal_threshold

_FIXED = 1e8
_MARK = struct.Struct("<QqqqqQ8x")
_TRADE = struct.Struct("<QqqQQQB7x")


@dataclass(frozen=True)
class MarketTick:
    """One aligned observation, normally one row per mark-price second."""

    timestamp_ms: int
    mark_price: float
    index_price: float
    funding_rate: float = 0.0
    trade_volume: float = 0.0
    signed_volume: float = 0.0
    spread_bps: float = 0.0
    book_imbalance: float = 0.0


@dataclass(frozen=True)
class MarkRecord:
    timestamp_ms: int
    mark_price: float
    index_price: float
    estimated_settle: float
    funding_rate: float
    next_funding_ms: int


@dataclass(frozen=True)
class TradeRecord:
    timestamp_ms: int
    price: float
    quantity: float
    is_buyer_maker: bool


@dataclass(frozen=True)
class TradingFeatureConfig:
    return_window: int = 30
    volume_window: int = 30
    max_gap_ms: int = 5_000


class TradingFeatureBank:
    """Causal, bounded market features with no future or label access."""

    FEATURE_NAMES = (
        "log_return", "realized_volatility", "signed_volume_imbalance",
        "volume_surprise", "basis_bps", "funding_bps", "spread_bps",
        "book_imbalance",
    )

    def __init__(self, cfg: TradingFeatureConfig | None = None):
        self.cfg = cfg or TradingFeatureConfig()
        if self.cfg.return_window < 2 or self.cfg.volume_window < 2:
            raise ValueError("feature windows must be at least 2")
        self._previous: MarketTick | None = None
        self._returns: deque[float] = deque(maxlen=self.cfg.return_window)
        self._volumes: deque[float] = deque(maxlen=self.cfg.volume_window)

    def update(self, tick: MarketTick) -> np.ndarray | None:
        self._validate(tick)
        previous = self._previous
        if previous is not None and tick.timestamp_ms <= previous.timestamp_ms:
            raise ValueError("timestamps must be strictly increasing")
        self._previous = tick
        if previous is None:
            return None
        if tick.timestamp_ms - previous.timestamp_ms > self.cfg.max_gap_ms:
            self._returns.clear()
            self._volumes.clear()
            return None
        ret = math.log(tick.mark_price / previous.mark_price)
        self._returns.append(ret)
        volume = max(float(tick.trade_volume), 0.0)
        self._volumes.append(volume)
        logs = np.log1p(np.asarray(self._volumes, dtype=np.float64))
        volume_surprise = float(logs[-1] - np.median(logs[:-1])) if len(logs) > 1 else 0.0
        realized = float(np.sqrt(np.dot(self._returns, self._returns)))
        features = np.array([
            ret,
            realized,
            float(tick.signed_volume) / volume if volume > 0 else 0.0,
            volume_surprise,
            1e4 * (tick.mark_price / tick.index_price - 1.0),
            1e4 * tick.funding_rate,
            float(tick.spread_bps),
            float(tick.book_imbalance),
        ], dtype=np.float64)
        return np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

    @staticmethod
    def _validate(tick: MarketTick) -> None:
        if tick.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative")
        if tick.mark_price <= 0 or tick.index_price <= 0:
            raise ValueError("mark and index prices must be positive")
        if tick.trade_volume < 0 or not np.isfinite(tick.trade_volume):
            raise ValueError("trade_volume must be finite and non-negative")
        if abs(tick.signed_volume) > tick.trade_volume + 1e-12:
            raise ValueError("signed_volume cannot exceed trade_volume")
        if not all(np.isfinite(v) for v in (
            tick.funding_rate, tick.signed_volume, tick.spread_bps, tick.book_imbalance)):
            raise ValueError("market features must be finite")


@dataclass(frozen=True)
class TradingMonitorConfig:
    commissioning_steps: int = 300
    alpha: float = 0.01
    persistence: int = 3
    reduce_multiplier: float = 0.5
    B_sem: int = 32
    B_epi: int = 16
    raw_window_bytes: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        if self.commissioning_steps < math.ceil(1.0 / self.alpha):
            raise ValueError("commissioning_steps is too small for conformal calibration")
        if self.persistence < 1:
            raise ValueError("persistence must be at least 1")
        if not 0 < self.reduce_multiplier < 1:
            raise ValueError("reduce_multiplier must be between 0 and 1")
        if self.B_sem < 1 or self.B_epi < 1 or self.raw_window_bytes < 0:
            raise ValueError("memory budgets must be positive and raw bytes non-negative")


@dataclass(frozen=True)
class TradingDecision:
    phase: str
    action: str
    size_multiplier: float
    novelty: float
    health: float
    novelty_threshold: float
    health_threshold: float
    persistence: int
    reason: str

    @property
    def should_trade(self) -> bool:
        return self.action != "halt"


class TradingRegimeMonitor:
    """Fixed-budget detector for a market-risk control plane.

    Commissioning is label-free and must contain only the intended healthy
    operating regime.  After commissioning, the adaptive Engram memory tracks
    recurring states while the frozen reference keeps a persistent baseline
    health signal.  Either signal can reduce size; a persistent breach halts.
    """

    def __init__(self, cfg: TradingMonitorConfig | None = None,
                 feature_cfg: TradingFeatureConfig | None = None):
        self.cfg = cfg or TradingMonitorConfig()
        self.features = TradingFeatureBank(feature_cfg)
        self._commission: list[np.ndarray] = []
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._model: Engram | None = None
        self._novelty_threshold = float("inf")
        self._health_threshold = float("inf")
        self._persistence = 0
        self._step = 0

    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def memory_bytes(self) -> int:
        if self._model is None:
            return 0
        return int(self._model.index_bytes() + self._model.pinned_raw_bytes())

    def observe(self, tick: MarketTick) -> TradingDecision:
        raw = self.features.update(tick)
        if raw is None:
            return self._decision("warmup", "halt", 0.0, math.nan, math.nan,
                                  "insufficient_contiguous_history")
        if self._model is None:
            self._commission.append(raw)
            if len(self._commission) < self.cfg.commissioning_steps:
                return self._decision("commissioning", "halt", 0.0, math.nan, math.nan,
                                      "commissioning")
            self._finish_commissioning()
            return self._decision("ready", "allow", 1.0, math.nan, math.nan, "commissioned")
        z = (raw - self._mean) / self._scale
        result = self._model.ingest(z, self._step, self._step)
        self._step += 1
        novelty, health = float(result.score), float(result.health)
        novelty_bad = np.isfinite(novelty) and novelty > self._novelty_threshold
        health_bad = np.isfinite(health) and health > self._health_threshold
        bad = novelty_bad or health_bad
        self._persistence = self._persistence + 1 if bad else 0
        if self._persistence >= self.cfg.persistence:
            action, multiplier = "halt", 0.0
        elif bad:
            action, multiplier = "reduce", self.cfg.reduce_multiplier
        else:
            action, multiplier = "allow", 1.0
        reasons = []
        if novelty_bad:
            reasons.append("novelty")
        if health_bad:
            reasons.append("baseline_drift")
        reason = "+".join(reasons) if reasons else "within_baseline"
        return self._decision("monitoring", action, multiplier, novelty, health, reason)

    def _finish_commissioning(self) -> None:
        X = np.asarray(self._commission, dtype=np.float64)
        self._mean = np.median(X, axis=0)
        mad = np.median(np.abs(X - self._mean), axis=0)
        scale = 1.4826 * mad
        fallback = np.std(X, axis=0, ddof=1)
        self._scale = np.maximum(np.where(scale > 1e-8, scale, fallback), 1e-8)
        Z = (X - self._mean) / self._scale
        model_cfg = EngramConfig(
            B_sem=self.cfg.B_sem, B_epi=self.cfg.B_epi,
            alpha_gate=1e-4, m_promote=5, n_promote=8,
            alarm_threshold=None, use_step_detect=True,
        )
        self._model = Engram(
            model_cfg, Z.shape[1], np.ones(Z.shape[1]), self.cfg.raw_window_bytes
        )
        scores = np.array([self._model.ingest(z, i, i).score for i, z in enumerate(Z)])
        self._novelty_threshold = conformal_threshold(scores[np.isfinite(scores)], self.cfg.alpha)
        ref_mu, ref_var = self._model.reference()
        if len(ref_mu) == 0:
            ref_mu = np.mean(Z, axis=0, keepdims=True)
            ref_var = np.ones_like(ref_mu)
        self._model.set_reference(ref_mu, np.maximum(ref_var, 1e-8))
        health = self._model.health(Z)
        self._health_threshold = conformal_threshold(health[np.isfinite(health)], self.cfg.alpha)
        self._commission.clear()

    def _decision(self, phase: str, action: str, multiplier: float,
                  novelty: float, health: float, reason: str) -> TradingDecision:
        return TradingDecision(
            phase, action, multiplier, novelty, health,
            self._novelty_threshold, self._health_threshold,
            self._persistence, reason,
        )


def iter_mark_records(path: str | Path) -> Iterator[MarkRecord]:
    yield from _records(path, _MARK, lambda r: MarkRecord(
        r[0], r[1] / _FIXED, r[2] / _FIXED, r[3] / _FIXED,
        r[4] / _FIXED, r[5]))


def iter_trade_records(path: str | Path) -> Iterator[TradeRecord]:
    yield from _records(path, _TRADE, lambda r: TradeRecord(
        r[5], r[1] / _FIXED, r[2] / _FIXED, bool(r[6])))


def iter_market_ticks(mark_path: str | Path, trade_path: str | Path) -> Iterator[MarketTick]:
    """Align sorted daily capture files without loading either file in memory."""
    trades = iter(iter_trade_records(trade_path))
    pending = next(trades, None)
    for mark in iter_mark_records(mark_path):
        volume = signed = 0.0
        while pending is not None and pending.timestamp_ms <= mark.timestamp_ms:
            volume += pending.quantity
            signed += -pending.quantity if pending.is_buyer_maker else pending.quantity
            pending = next(trades, None)
        yield MarketTick(mark.timestamp_ms, mark.mark_price, mark.index_price,
                         mark.funding_rate, volume, signed)


def _records(path: str | Path, layout: struct.Struct, convert) -> Iterable:
    with Path(path).open("rb") as stream:
        while True:
            data = stream.read(layout.size)
            if not data:
                return
            if len(data) != layout.size:
                raise ValueError(f"truncated capture record in {path}")
            yield convert(layout.unpack(data))
