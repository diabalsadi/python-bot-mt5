"""
UT Bot Alerts Indicator
-----------------------
Python port of the Pine Script "UT Bot Alerts" by QuantNomad.

Uses ATR-based trailing stops to generate buy/sell signals:
  - Computes an ATR Trailing Stop that adapts to price movement
  - Buy signal:  price crosses above the trailing stop
  - Sell signal: price crosses below the trailing stop

Key parameters:
  - key_value:  Sensitivity multiplier (default 1). Higher = wider stops
  - atr_period: ATR lookback period (default 10)

Features:
  - calculate_ut_bot()       → raw trailing stop + position state
  - calculate_ut_bot_signal() → normalized signal for ML [-1, 1]
"""

from __future__ import annotations

import MetaTrader5 as mt5
import numpy as np


def _calculate_atr_series(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int
) -> np.ndarray:
    """
    Calculate ATR series using Wilder's smoothing (EMA-style).

    Args:
        highs:  High prices array (oldest first for calculation)
        lows:   Low prices array
        closes: Close prices array
        period: ATR period

    Returns:
        ATR array (same length as input, first `period` values are SMA-based)
    """
    n = len(highs)
    tr = np.zeros(n)

    # True Range for each bar
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)
    tr[0] = highs[0] - lows[0]

    # Wilder's smoothing (RMA)
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[:period])  # seed with SMA
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    # Fill initial values with SMA
    for i in range(period - 1):
        atr[i] = np.mean(tr[:i + 1]) if i > 0 else tr[0]

    return atr


def calculate_ut_bot(
    symbol: str,
    timeframe: int,
    shift: int = 0,
    key_value: float = 1.0,
    atr_period: int = 10,
    lookback: int = 100,
) -> dict:
    """
    Full UT Bot calculation returning trailing stop, position, and signals.

    Args:
        symbol:     Trading symbol
        timeframe:  MT5 timeframe constant
        shift:      Bar offset (0 = current forming bar)
        key_value:  ATR multiplier for sensitivity (Pine Script: "Key Value")
        atr_period: ATR period
        lookback:   Number of bars to compute over

    Returns:
        {
            "trailing_stop": float,   # Current ATR trailing stop level
            "position": int,          # 1 = bullish, -1 = bearish, 0 = neutral
            "buy_signal": bool,       # Fresh buy crossover
            "sell_signal": bool,      # Fresh sell crossover
            "bar_buy": bool,          # Price above trailing stop
            "bar_sell": bool,         # Price below trailing stop
        }
    """
    count = shift + lookback + atr_period + 10
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) < lookback + atr_period + 5:
        return {
            "trailing_stop": 0.0, "position": 0,
            "buy_signal": False, "sell_signal": False,
            "bar_buy": False, "bar_sell": False,
        }

    # Work in chronological order (oldest first) for the calculation
    closes = rates["close"].astype(np.float64)
    highs = rates["high"].astype(np.float64)
    lows = rates["low"].astype(np.float64)

    n = len(closes)

    # ATR series
    atr = _calculate_atr_series(highs, lows, closes, atr_period)
    nloss = key_value * atr  # nLoss = key_value * xATR

    # ATR Trailing Stop calculation (Pine Script logic)
    trailing_stop = np.zeros(n)

    for i in range(1, n):
        src = closes[i]
        src_prev = closes[i - 1]
        prev_ts = trailing_stop[i - 1]

        if src > prev_ts and src_prev > prev_ts:
            # Price above trailing stop — raise stop (but never lower it)
            trailing_stop[i] = max(prev_ts, src - nloss[i])
        elif src < prev_ts and src_prev < prev_ts:
            # Price below trailing stop — lower stop (but never raise it)
            trailing_stop[i] = min(prev_ts, src + nloss[i])
        elif src > prev_ts:
            # Crossed above — start new bullish trail
            trailing_stop[i] = src - nloss[i]
        else:
            # Crossed below — start new bearish trail
            trailing_stop[i] = src + nloss[i]

    # Position tracking
    pos = np.zeros(n, dtype=int)
    for i in range(1, n):
        if closes[i - 1] < trailing_stop[i - 1] and closes[i] > trailing_stop[i]:
            pos[i] = 1   # Flipped bullish
        elif closes[i - 1] > trailing_stop[i - 1] and closes[i] < trailing_stop[i]:
            pos[i] = -1  # Flipped bearish
        else:
            pos[i] = pos[i - 1]

    # EMA(close, 1) ≈ close itself (as in original Pine Script)
    ema = closes.copy()

    # Target index (reverse from end: shift=0 → last bar)
    target_idx = n - 1 - shift
    if target_idx < 2:
        return {
            "trailing_stop": 0.0, "position": 0,
            "buy_signal": False, "sell_signal": False,
            "bar_buy": False, "bar_sell": False,
        }

    # Crossover detection
    above = (ema[target_idx] > trailing_stop[target_idx] and
             ema[target_idx - 1] <= trailing_stop[target_idx - 1])
    below = (trailing_stop[target_idx] > ema[target_idx] and
             trailing_stop[target_idx - 1] <= ema[target_idx - 1])

    buy_signal = closes[target_idx] > trailing_stop[target_idx] and above
    sell_signal = closes[target_idx] < trailing_stop[target_idx] and below

    bar_buy = closes[target_idx] > trailing_stop[target_idx]
    bar_sell = closes[target_idx] < trailing_stop[target_idx]

    return {
        "trailing_stop": float(trailing_stop[target_idx]),
        "position": int(pos[target_idx]),
        "buy_signal": bool(buy_signal),
        "sell_signal": bool(sell_signal),
        "bar_buy": bool(bar_buy),
        "bar_sell": bool(bar_sell),
    }


def calculate_ut_bot_signal(
    symbol: str,
    timeframe: int,
    shift: int = 0,
    key_value: float = 1.0,
    atr_period: int = 10,
) -> float:
    """
    Normalized UT Bot signal for ML feature pipeline.

    Returns:
        Score in [-1, 1]:
        +1.0 = fresh buy crossover signal (strongest bullish)
        +0.5 = price above trailing stop (bullish bias)
        -0.5 = price below trailing stop (bearish bias)
        -1.0 = fresh sell crossover signal (strongest bearish)
         0.0 = neutral / error
    """
    ut = calculate_ut_bot(symbol, timeframe, shift, key_value, atr_period)

    if ut["buy_signal"]:
        return 1.0
    elif ut["sell_signal"]:
        return -1.0
    elif ut["bar_buy"]:
        return 0.5
    elif ut["bar_sell"]:
        return -0.5
    else:
        return 0.0
