"""
Support & Resistance Indicators
--------------------------------
Detects key support and resistance levels on multiple timeframes
and returns features measuring distance and strength.

Features:
  - identify_levels()   → finds S/R levels using pivot points
  - calculate_sr_distance() → distance to nearest S/R (normalized feature)
  - calculate_sr_strength() → how many times level was tested
"""

from __future__ import annotations

import MetaTrader5 as mt5
import numpy as np


def identify_sr_levels(
    symbol: str, timeframe: int, lookback_bars: int = 50
) -> tuple[list[float], list[float]]:
    """
    Identify support and resistance levels using pivot point and swing detection.

    Args:
        symbol:         Trading symbol
        timeframe:      MT5 timeframe constant
        lookback_bars:  Number of bars to analyze

    Returns:
        (support_levels, resistance_levels)  — lists of floats
    """
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, lookback_bars + 10)
    if rates is None or len(rates) < lookback_bars:
        return [], []

    highs = rates["high"]
    lows = rates["low"]

    support_levels = []
    resistance_levels = []

    # Pivot point method: identify local highs and lows
    period = 3
    for i in range(period, len(highs) - period):
        # Local high (resistance)
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            if not resistance_levels or abs(highs[i] - resistance_levels[-1]) > 10:
                resistance_levels.append(float(highs[i]))

        # Local low (support)
        if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            if not support_levels or abs(lows[i] - support_levels[-1]) > 10:
                support_levels.append(float(lows[i]))

    return support_levels[::-1], resistance_levels[::-1]  # most recent first


def calculate_sr_distance(
    symbol: str, timeframe: int, shift: int = 1, lookback_bars: int = 50
) -> float:
    """
    Calculate normalized distance to nearest support/resistance level.
    Returns a value between -1 (at support) and +1 (at resistance),
    0 = midway between S/R.

    Args:
        symbol:         Trading symbol
        timeframe:      MT5 timeframe constant
        shift:          Bar offset (1 = last closed bar)
        lookback_bars:  Lookback window for S/R identification

    Returns:
        Normalized SR distance (-1 to +1)
    """
    sym_info = mt5.symbol_info(symbol)
    if sym_info is None:
        return 0.0

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, shift + 5)
    if rates is None or len(rates) < shift + 1:
        return 0.0

    current_price = float(rates["close"][::-1][shift])

    support_levels, resistance_levels = identify_sr_levels(symbol, timeframe, lookback_bars)

    if not support_levels and not resistance_levels:
        return 0.0

    # Find nearest support and resistance
    nearest_support = None
    nearest_resistance = None

    if support_levels:
        # Find highest support below current price
        below = [s for s in support_levels if s < current_price]
        if below:
            nearest_support = max(below)

    if resistance_levels:
        # Find lowest resistance above current price
        above = [r for r in resistance_levels if r > current_price]
        if above:
            nearest_resistance = min(above)

    # Normalize distance between support and resistance
    if nearest_support is not None and nearest_resistance is not None:
        range_sr = nearest_resistance - nearest_support
        if range_sr > 0:
            # 0 = at support, 1 = at resistance
            normalized = (current_price - nearest_support) / range_sr
            return 2 * (normalized - 0.5)  # Scale to -1 to +1

    elif nearest_support is not None:
        # Only support below, distance is negative (proximity to support)
        distance = current_price - nearest_support
        return -min(1.0, distance / (current_price * 0.01))  # negative = close to support

    elif nearest_resistance is not None:
        # Only resistance above, distance is positive (proximity to resistance)
        distance = nearest_resistance - current_price
        return min(1.0, distance / (current_price * 0.01))  # positive = close to resistance

    return 0.0


def calculate_sr_strength(
    symbol: str, timeframe: int, shift: int = 1, lookback_bars: int = 50
) -> float:
    """
    Measure the strength of nearby support/resistance by counting
    how many times price has tested a level.

    Returns a value 0-1 indicating strength (0 = untested, 1 = very strong).

    Args:
        symbol:         Trading symbol
        timeframe:      MT5 timeframe constant
        shift:          Bar offset (1 = last closed bar)
        lookback_bars:  Lookback window for analysis

    Returns:
        SR strength (0 to 1)
    """
    sym_info = mt5.symbol_info(symbol)
    if sym_info is None:
        return 0.0

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, shift + lookback_bars + 5)
    if rates is None or len(rates) < shift + lookback_bars:
        return 0.0

    highs = rates["high"]
    lows = rates["low"]
    current_price = float(rates["close"][::-1][shift])

    support_levels, resistance_levels = identify_sr_levels(symbol, timeframe, lookback_bars)

    max_tests = 0
    touch_threshold = current_price * 0.0005  # 5 pips equivalent in relative terms

    # Count touches for support levels
    for support in support_levels:
        touches = np.sum(
            (np.abs(lows[::-1][:shift + lookback_bars] - support) < touch_threshold)
        )
        max_tests = max(max_tests, int(touches))

    # Count touches for resistance levels
    for resistance in resistance_levels:
        touches = np.sum(
            (np.abs(highs[::-1][:shift + lookback_bars] - resistance) < touch_threshold)
        )
        max_tests = max(max_tests, int(touches))

    # Normalize strength: 1 test = 0.2, 5+ tests = 1.0
    strength = min(1.0, max_tests / 5.0)
    return float(strength)
