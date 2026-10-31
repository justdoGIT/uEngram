import struct

import numpy as np
import pytest

from uengram import (
    MarketTick,
    TradingFeatureBank,
    TradingFeatureConfig,
    TradingMonitorConfig,
    TradingRegimeMonitor,
    iter_market_ticks,
)


_FIXED = 100_000_000
_MARK = struct.Struct("<QqqqqQ8x")
_TRADE = struct.Struct("<QqqQQQB7x")


def tick(i, price=100.0, **kwargs):
    index = kwargs.pop("index_price", price * 0.9999)
    return MarketTick(i * 1000, price, index, **kwargs)


def test_feature_bank_is_causal_and_reports_aggressor_imbalance():
    bank = TradingFeatureBank(TradingFeatureConfig(return_window=3, volume_window=3))
    assert bank.update(tick(0)) is None
    first = bank.update(tick(1, 100.01, trade_volume=2.0, signed_volume=1.0))
    assert first is not None
    assert first.shape == (8,)
    assert first[2] == pytest.approx(0.5)
    earlier = first.copy()
    later = bank.update(tick(2, 100.02, trade_volume=2.0, signed_volume=-1.0))
    assert later[2] == pytest.approx(-0.5)
    assert not np.array_equal(earlier, later)


def test_gap_does_not_turn_missing_data_into_a_price_jump():
    bank = TradingFeatureBank(TradingFeatureConfig(max_gap_ms=2_000))
    assert bank.update(tick(0)) is None
    assert bank.update(tick(1, 100.1)) is not None
    assert bank.update(tick(10, 200.0)) is None
    assert bank.update(tick(11, 200.1)) is not None


def test_monitor_calibrates_then_reduces_and_halts_persistent_regime_shift():
    monitor = TradingRegimeMonitor(
        TradingMonitorConfig(commissioning_steps=80, alpha=0.02, persistence=2),
        TradingFeatureConfig(return_window=5, volume_window=5),
    )
    decisions = []
    for i in range(100):
        price = 100.0 + 0.25 * np.sin(i / 3.0)
        decisions.append(monitor.observe(tick(
            i, price, trade_volume=1.0 + (i % 4) * 0.1,
            signed_volume=0.2 if i % 2 else -0.2,
        )))
    assert monitor.ready
    assert any(d.phase == "ready" for d in decisions)
    normal = monitor.observe(tick(101, 100.0 + 0.25 * np.sin(101 / 3.0),
                                  trade_volume=1.0, signed_volume=0.0))
    assert normal.action == "allow"

    shock = []
    for i in range(102, 106):
        shock.append(monitor.observe(tick(
            i, 125.0 + i, index_price=100.0, funding_rate=0.01,
            trade_volume=100.0, signed_volume=100.0,
        )))
    assert any(d.action == "reduce" for d in shock)
    assert shock[-1].action == "halt"
    assert shock[-1].size_multiplier == 0.0
    assert monitor.memory_bytes > 0


def test_capture_alignment_streams_and_preserves_signed_aggressor_volume(tmp_path):
    mark_path = tmp_path / "BTCUSDT-mark.bin"
    trade_path = tmp_path / "BTCUSDT-trades.bin"
    mark_path.write_bytes(b"".join(_MARK.pack(
        1_000 * i, int(100 * _FIXED), int(100 * _FIXED), int(100 * _FIXED),
        10_000, 2_000,
    ) for i in (1, 2)))
    trade_path.write_bytes(
        _TRADE.pack(1, 100 * _FIXED, 3 * _FIXED, 1, 1, 1_000, 0)
        + _TRADE.pack(2, 100 * _FIXED, 1 * _FIXED, 2, 2, 1_000, 1)
    )
    rows = list(iter_market_ticks(mark_path, trade_path))
    assert len(rows) == 2
    assert rows[0].trade_volume == pytest.approx(4.0)
    assert rows[0].signed_volume == pytest.approx(2.0)
    assert rows[1].trade_volume == pytest.approx(0.0)


def test_capture_reader_rejects_truncated_record(tmp_path):
    mark_path = tmp_path / "bad.bin"
    trade_path = tmp_path / "empty.bin"
    mark_path.write_bytes(b"short")
    trade_path.write_bytes(b"")
    with pytest.raises(ValueError, match="truncated capture record"):
        list(iter_market_ticks(mark_path, trade_path))
