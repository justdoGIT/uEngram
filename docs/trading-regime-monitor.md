# Trading regime and risk monitor

`uengram.trading.TradingRegimeMonitor` is a bounded **risk-control component**
for market systems. It is not a strategy, price predictor, or order executor.
A separate execution policy may use its `TradingDecision` to permit normal size,
reduce size, or halt new entries.

## Features

The causal feature bank consumes one aligned row per mark-price interval:

| Feature | Source | Meaning |
|---|---|---|
| `log_return` | mark price | current price movement |
| `realized_volatility` | recent returns | short-window movement energy |
| `signed_volume_imbalance` | aggTrade | aggressor buy/sell pressure; buyer-maker trades are negative |
| `volume_surprise` | recent volume | log-volume departure from its rolling median |
| `basis_bps` | mark and index | mark/index divergence |
| `funding_bps` | funding rate | current funding pressure |
| `spread_bps` | optional book data | quoted transaction cost |
| `book_imbalance` | optional book data | quoted bid/ask pressure |

Missing book features are represented as zero by the capture adapter. They are
not fabricated from the mark stream.

## Lifecycle

1. **Warmup:** the first row and any row after a timestamp gap are not scored.
   The decision is `halt` because no valid contiguous observation exists.
2. **Commissioning:** the monitor collects a label-free, healthy reference
   cohort. Decisions remain `halt`; do not trade during this phase.
3. **Ready:** robust median/MAD scaling is fitted and a bounded `Engram` is
   populated. Novelty and frozen-baseline health thresholds are split-conformal
   thresholds at `TradingMonitorConfig.alpha`.
4. **Monitoring:** a novelty or persistent-baseline breach returns `reduce`.
   Consecutive breaches at `persistence` return `halt`. A clean observation
   returns `allow`.

The adaptive memory is bounded by `B_sem + B_epi` slots. The frozen reference
never adapts, so a recurring new regime cannot erase the baseline-drift signal.
The thresholds are only valid for an exchangeable commissioning population;
markets are reflexive and non-stationary, so this is not a false-alarm or PnL
guarantee.

## Captured Binance files

The adapter reads the capture layout documented by the companion
`binance-futures-bot` repository without loading an entire day into memory:

```python
from uengram import TradingMonitorConfig, TradingRegimeMonitor, iter_market_ticks

monitor = TradingRegimeMonitor(TradingMonitorConfig(
    commissioning_steps=900,
    alpha=0.01,
    persistence=3,
))
for tick in iter_market_ticks("BTCUSDT-mark-2026-10-31.bin",
                              "BTCUSDT-trades-2026-10-31.bin"):
    decision = monitor.observe(tick)
    if decision.phase == "monitoring":
        # The order path must enforce this decision independently.
        print(tick.timestamp_ms, decision.action, decision.size_multiplier,
              decision.reason)
```

`iter_market_ticks` joins trades to the next mark interval, maps fixed-point
values back to decimals, and uses `is_buyer_maker` to sign aggressor volume. It
rejects truncated records and does not silently bridge gaps.

## Required validation before any deployment

This implementation has unit and behavior tests for feature causality, gap
handling, calibration, persistent-shift escalation, and binary capture parsing.
Those tests do **not** establish trading performance. Before connecting it to
an execution path, use a separate out-of-sample protocol with:

- time-disjoint commissioning, calibration, and evaluation periods;
- symbol and regime holdouts;
- fees, spread, slippage, funding, latency, partial fills, and rejects;
- maximum drawdown, exposure, and restart/kill-switch tests;
- a comparison against a no-monitor control and a simple volatility baseline;
- explicit reporting of missed regime shifts and false halts.

The monitor must be placed before order sizing and behind the execution system's
existing hard risk and kill-switch gates. It does not authorize real orders.
