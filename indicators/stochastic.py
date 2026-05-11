"""
Stochastic Oscillator
---------------------
Calculates %K and %D lines with slowing (smoothing).
Standard formula: %K = 100 * (Close - LL) / (HH - LL)
"""

import MetaTrader5 as mt5
import numpy as np

def calculate_stochastic(symbol: str, timeframe: int, shift: int = 0,
                         k_period: int = 13, d_period: int = 8, slowing: int = 8) -> tuple:
    """
    Returns (percent_k, percent_d) at the given shift.
    """
    # Need history for K + Slowing + D
    count = shift + k_period + slowing + d_period + 10
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) < count:
        return 0.0, 0.0

    closes = rates["close"]
    highs = rates["high"]
    lows = rates["low"]

    # Calculate raw %K for each bar needed
    k_raw = []
    # We need enough K values to calculate D (which is an SMA of K)
    # And K itself is smoothed by 'slowing'
    needed_k = d_period + slowing + 2
    
    for i in range(len(rates) - k_period):
        subset_h = highs[i : i + k_period]
        subset_l = lows[i : i + k_period]
        hh = np.max(subset_h)
        ll = np.min(subset_l)
        
        diff = hh - ll
        if diff == 0:
            k_raw.append(50.0)
        else:
            k_raw.append(100.0 * (closes[i + k_period - 1] - ll) / diff)

    # Apply slowing (smoothing %K)
    k_smoothed = []
    for i in range(len(k_raw) - slowing):
        k_smoothed.append(np.mean(k_raw[i : i + slowing]))
    
    # Calculate %D (SMA of smoothed %K)
    d_values = []
    for i in range(len(k_smoothed) - d_period):
        d_values.append(np.mean(k_smoothed[i : i + d_period]))

    # MT5 index 0 is current. Our arrays are chronologically ordered (oldest first).
    # So the last value is the current one (shift=0).
    target_idx = -1 - shift
    if abs(target_idx) > len(d_values):
        return 0.0, 0.0
        
    return k_smoothed[target_idx], d_values[target_idx]
