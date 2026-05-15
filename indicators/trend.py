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


def calculate_trend(
    symbol: str, timeframe: int, shift: int, feature_window: int
) -> float:
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

    sum_x = x.sum()
    sum_y = y.sum()
    sum_xy = (x * y).sum()
    sum_x2 = (x * x).sum()

    denom = n * sum_x2 - sum_x**2
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
    Advanced Confluence Signal combining multiple technical layers:
    1. EMA Stack (20, 50, 100, 200)
    2. Ichimoku Kumo Cloud
    3. Stochastic (13, 8, 8)
    4. MACD (12, 26, 9)
    5. RSI Pullback Filter
    """
    from indicators.rsi import calculate_rsi
    from indicators.ema import calculate_ema
    from indicators.ichimoku import calculate_ichimoku
    from indicators.stochastic import calculate_stochastic
    from indicators.macd import calculate_macd

    # 1. PRICE DATA
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1)
    if rates is None or len(rates) == 0: return 0
    price = rates[0]["close"]

    # 2. EMA STACK (Trend Foundation)
    ema20 = calculate_ema(symbol, timeframe, 0, 20)
    ema50 = calculate_ema(symbol, timeframe, 0, 50)
    ema100 = calculate_ema(symbol, timeframe, 0, 100)
    ema200 = calculate_ema(symbol, timeframe, 0, 200)
    
    if ema200 == 0: return 0
    
    bullish_stack = (price > ema20 > ema50 > ema100 > ema200)
    bearish_stack = (price < ema20 < ema50 < ema100 < ema200)

    # 3. ICHIMOKU CLOUD
    ichi = calculate_ichimoku(symbol, timeframe, 0)
    if not ichi: return 0
    in_cloud = min(ichi["span_a"], ichi["span_b"]) <= price <= max(ichi["span_a"], ichi["span_b"])
    above_cloud = price > max(ichi["span_a"], ichi["span_b"])
    below_cloud = price < min(ichi["span_a"], ichi["span_b"])

    # 4. MOMENTUM (RSI & STOCHASTIC & MACD)
    rsi = calculate_rsi(symbol, timeframe, 0, rsi_period)
    stoch_k, stoch_d = calculate_stochastic(symbol, timeframe, 0, 13, 8, 8)
    macd_main, macd_sig = calculate_macd(symbol, timeframe, 0, return_tuple=True)

    # ── CONFLUENCE LOGIC ───────────────────────────────────────────
    
    # BULLISH SIGNAL (BUY)
    # Stricter: Price above all EMAs, above Cloud, and Momentum turning up
    if bullish_stack and above_cloud:
        # Check for pullback completion (RSI rising from low, Stoch cross)
        if (30.0 < rsi < 55.0) and (stoch_k > stoch_d) and (macd_main > macd_sig):
            # Exhaustion filter
            if rsi < 80.0: return 1
            
    # BEARISH SIGNAL (SELL)
    if bearish_stack and below_cloud:
        if (45.0 < rsi < 70.0) and (stoch_k < stoch_d) and (macd_main < macd_sig):
            if rsi > 20.0: return -1

    return 0
