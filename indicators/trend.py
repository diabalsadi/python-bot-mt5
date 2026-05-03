"""
Trend Indicators
----------------
Two distinct trend tools ported from MQL5:

1. calculate_trend()   → linear-regression slope in points/bar
                         (MQL5: CalculateTrend)

2. get_technical_trend() → composite signal using EMA position, RSI level,
                           and short-term price direction
                           (MQL5: GetTechnicalTrend)
                           Returns  1 (bullish),  -1 (bearish), or  0 (neutral)
"""

import MetaTrader5 as mt5
import numpy as np


def calculate_trend(symbol: str, timeframe: int, shift: int, feature_window: int) -> float:
    """
    Fit a straight line (OLS) through the closes of the last `feature_window`
    bars and return its slope in points per bar.

    Args:
        symbol:         Trading symbol
        timeframe:      MT5 timeframe constant
        shift:          Starting bar index (1 = last closed bar)
        feature_window: Number of bars used for the regression

    Returns:
        Slope in points/bar.  Positive = upward drift, negative = downward.
        Returns 0.0 on data error.
    """
    sym_info = mt5.symbol_info(symbol)
    if sym_info is None:
        return 0.0

    point = sym_info.point
    count = shift + feature_window + 2
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)

    if rates is None or len(rates) < count:
        return 0.0

    closes_rev = rates["close"][::-1]  # index 0 = current bar

    n = feature_window
    x = np.arange(n, dtype=float)
    y = np.array([closes_rev[shift + i] for i in range(n)])

    sum_x   = x.sum()
    sum_y   = y.sum()
    sum_xy  = (x * y).sum()
    sum_x2  = (x * x).sum()

    denom = n * sum_x2 - sum_x ** 2
    if denom == 0:
        return 0.0

    slope = (n * sum_xy - sum_x * sum_y) / denom
    return slope / point if point > 0 else slope


def get_technical_trend(
    symbol: str,
    timeframe: int,
    trend_bars: int,
    ema_period: int,
    rsi_period: int,
) -> int:
    """
    Composite technical trend signal combining:
      - Price position relative to EMA
      - RSI level vs 50
      - Short-term price direction over `trend_bars` bars

    Buy  signal: price > EMA  AND  RSI < 30  AND  close[1] > close[1 + trend_bars]
    Sell signal: price < EMA  AND  RSI > 70  AND  close[1] < close[1 + trend_bars]

    Args:
        symbol:      Trading symbol
        timeframe:   MT5 timeframe constant
        trend_bars:  Short-term look-back (MQL5 TrendBars)
        ema_period:  EMA period for baseline trend
        rsi_period:  RSI period

    Returns:
        1  → bullish signal
       -1  → bearish signal
        0  → neutral / insufficient data
    """
    # Import here to avoid circular dependencies
    from indicators.rsi import calculate_rsi
    from indicators.ema import calculate_ema

    count = max(ema_period, rsi_period, trend_bars) + 20
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) < count:
        return 0

    closes = rates["close"][::-1]  # index 0 = current bar

    current_close = closes[1]                  # last fully closed bar
    old_close     = closes[1 + trend_bars]     # trend_bars before that

    current_rsi = calculate_rsi(symbol, timeframe, 1, rsi_period)
    current_ema = calculate_ema(symbol, timeframe, 1, ema_period)

    if current_ema == 0.0:
        return 0

    # ── Bullish ────────────────────────────────────────────────────────
    if current_close > current_ema and current_rsi < 30.0 and current_close > old_close:
        return 1

    # ── Bearish ────────────────────────────────────────────────────────
    if current_close < current_ema and current_rsi > 70.0 and current_close < old_close:
        return -1

    return 0
