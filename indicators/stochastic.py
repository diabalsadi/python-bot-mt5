"""
Stochastic Oscillator Indicator
-------------------------------
Calculates the Stochastic Oscillator (%K and %D).

Two call modes:
  1. calculate_stochastic(symbol, tf, shift)             -> float  (for ML features)
  2. calculate_stochastic(symbol, tf, shift, k, d, slow) -> (float, float)  (for trend.py)
"""

import MetaTrader5 as mt5
import numpy as np


def calculate_stochastic(symbol: str, timeframe: int, shift: int,
                         k_period: int = 14, d_period: int = 3,
                         slowing: int = 0) -> "float | tuple[float, float]":
    """
    Stochastic Oscillator.

    When called with 6 positional args (k_period, d_period, slowing),
    returns a tuple (stoch_k, stoch_d) on the 0-100 scale (for confluence logic).

    When called with <= 5 args (no slowing), returns a single float
    normalized to [-1, 1] for the ML feature pipeline.
    """
    count = shift + k_period + d_period + slowing + 20
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)

    if rates is None or len(rates) < count:
        return (50.0, 50.0) if slowing > 0 else 0.0

    highs = rates["high"]
    lows = rates["low"]
    closes = rates["close"]

    # Calculate raw %K for several bars so we can smooth for %D
    num_k = d_period + slowing + 2
    raw_k = np.zeros(num_k)

    for j in range(num_k):
        idx = len(closes) - 1 - shift - j
        if idx < k_period:
            raw_k[j] = 50.0
            continue

        window_highs = highs[idx - k_period + 1 : idx + 1]
        window_lows = lows[idx - k_period + 1 : idx + 1]

        highest_high = np.max(window_highs)
        lowest_low = np.min(window_lows)

        if highest_high == lowest_low:
            raw_k[j] = 50.0
        else:
            raw_k[j] = 100.0 * ((closes[idx] - lowest_low) / (highest_high - lowest_low))

    # Apply slowing (SMA of raw %K)
    if slowing > 1:
        smoothed_k = np.convolve(raw_k, np.ones(slowing) / slowing, mode='valid')
    else:
        smoothed_k = raw_k

    # %D = SMA of smoothed %K
    if len(smoothed_k) >= d_period:
        stoch_d = float(np.mean(smoothed_k[:d_period]))
    else:
        stoch_d = float(smoothed_k[0]) if len(smoothed_k) > 0 else 50.0

    stoch_k = float(smoothed_k[0]) if len(smoothed_k) > 0 else 50.0

    # If slowing was explicitly provided (6-arg call from trend.py),
    # return the raw (k, d) tuple on 0-100 scale
    if slowing > 0:
        return (stoch_k, stoch_d)

    # Otherwise return ML-normalized single float [-1, 1]
    return float((stoch_k - 50.0) / 50.0)
