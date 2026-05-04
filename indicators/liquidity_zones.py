"""
Liquidity Zones Indicator
-------------------------
Identifies areas of high liquidity concentration based on:
  - Volume confluence (where multiple timeframes align)
  - Price clustering (where price has spent most time)
  - Spike detection (sudden volume increase or volatility)

Features:
  - calculate_liquidity_score() → 0-1 score indicating zone strength
  - get_liquidity_zones()       → list of key liquidity areas
"""

from __future__ import annotations

import MetaTrader5 as mt5
import numpy as np


def calculate_liquidity_score(
    symbol: str, timeframe: int, shift: int = 1, lookback_bars: int = 50
) -> float:
    """
    Calculate liquidity concentration score at current price.
    High score = price is near a liquidity zone.

    Uses multi-timeframe analysis: confluence of recent highs/lows
    across the current timeframe.

    Args:
        symbol:         Trading symbol
        timeframe:      MT5 timeframe constant
        shift:          Bar offset (1 = last closed bar)
        lookback_bars:  Lookback period for analysis

    Returns:
        Liquidity score (0 to 1), where:
        - 0 = low liquidity concentration
        - 1 = very high concentration
    """
    sym_info = mt5.symbol_info(symbol)
    if sym_info is None:
        return 0.0

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, shift + lookback_bars + 5)
    if rates is None or len(rates) < shift + lookback_bars:
        return 0.0

    closes = rates["close"][::-1]
    highs = rates["high"][::-1]
    lows = rates["low"][::-1]

    current_price = float(closes[shift])

    # Collect all swing points (local highs and lows)
    swing_points = []
    period = 2

    for i in range(period, len(highs) - period):
        if highs[i] > highs[i - 1] and highs[i] >= highs[i + 1]:
            swing_points.append(float(highs[i]))
        if lows[i] < lows[i - 1] and lows[i] <= lows[i + 1]:
            swing_points.append(float(lows[i]))

    if not swing_points:
        return 0.0

    # Count how many swing points are near current price
    clustering_distance = current_price * 0.001  # 0.1% proximity
    nearby_swings = sum(
        1 for sp in swing_points if abs(sp - current_price) < clustering_distance
    )

    liquidity_density = nearby_swings / len(swing_points)

    # Volume check: detect if volatility is increasing (often at liquidity zones)
    if len(rates) >= shift + 10:
        recent_vol = np.std(closes[shift : shift + 5])
        older_vol = np.std(closes[shift + 5 : shift + 15])
        vol_ratio = recent_vol / (older_vol + 1e-10)
        vol_factor = min(1.0, vol_ratio / 2.0)  # Normalize to 0-1
    else:
        vol_factor = 0.0

    # Combine density and volatility
    liquidity_score = min(1.0, liquidity_density * 0.7 + vol_factor * 0.3)

    return float(liquidity_score)


def get_liquidity_zones(
    symbol: str, timeframe: int, lookback_bars: int = 50, sensitivity: float = 0.5
) -> list[dict]:
    """
    Identify key liquidity zones in the recent price action.

    Returns list of zones with:
      - "level": price level
      - "strength": 0-1 strength rating
      - "type": "support", "resistance", or "neutral"

    Args:
        symbol:         Trading symbol
        timeframe:      MT5 timeframe constant
        lookback_bars:  Lookback window
        sensitivity:    0.0-1.0, higher = more zones detected

    Returns:
        List of liquidity zone dictionaries
    """
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, lookback_bars + 10)
    if rates is None or len(rates) < lookback_bars:
        return []

    highs = rates["high"]
    lows = rates["low"]
    closes = rates["close"]

    zones = []

    # Identify swing highs and lows with strength rating
    period = 3
    for i in range(period, len(highs) - period):
        # Swing high (resistance)
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            strength = min(
                1.0,
                (highs[i] - np.mean(highs[i - 3 : i + 4])) / (np.std(highs) + 1e-10),
            )
            if strength > sensitivity:
                zones.append(
                    {
                        "level": float(highs[i]),
                        "strength": strength,
                        "type": "resistance",
                        "bar_ago": len(highs) - i - 1,
                    }
                )

        # Swing low (support)
        if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            strength = min(
                1.0,
                (np.mean(lows[i - 3 : i + 4]) - lows[i]) / (np.std(lows) + 1e-10),
            )
            if strength > sensitivity:
                zones.append(
                    {
                        "level": float(lows[i]),
                        "strength": strength,
                        "type": "support",
                        "bar_ago": len(lows) - i - 1,
                    }
                )

    # Sort by recency (most recent first)
    zones.sort(key=lambda z: z["bar_ago"])

    return zones[:10]  # Return top 10 most recent zones


def calculate_liquidity_confluence(
    symbol: str, base_timeframe: int, higher_timeframe: int, shift: int = 1
) -> float:
    """
    Measure confluence of S/R levels across two timeframes.
    Higher value = stronger confluence = higher liquidity.

    Args:
        symbol:           Trading symbol
        base_timeframe:   Primary timeframe (e.g., M1)
        higher_timeframe: Higher timeframe (e.g., M15)
        shift:            Bar offset

    Returns:
        Confluence score (0 to 1)
    """
    # Get price levels from base timeframe
    base_rates = mt5.copy_rates_from_pos(symbol, base_timeframe, 0, shift + 50)
    if base_rates is None:
        return 0.0

    # Get price levels from higher timeframe
    higher_rates = mt5.copy_rates_from_pos(symbol, higher_timeframe, 0, shift + 10)
    if higher_rates is None:
        return 0.0

    base_highs = base_rates["high"]
    base_lows = base_rates["low"]
    higher_highs = higher_rates["high"]
    higher_lows = higher_rates["low"]

    if len(base_highs) == 0 or len(higher_highs) == 0:
        return 0.0

    current_price = float(base_rates["close"][::-1][shift])
    price_range = (base_highs.max() - base_lows.min()) * 0.005  # 0.5% tolerance

    # Count confluence points
    confluence_count = 0

    for hh in higher_highs:
        if abs(float(hh) - current_price) < price_range:
            confluence_count += 1

    for hl in higher_lows:
        if abs(float(hl) - current_price) < price_range:
            confluence_count += 1

    # Normalize to 0-1
    max_possible = len(higher_highs) + len(higher_lows)
    confluence_score = min(1.0, confluence_count / (max_possible + 1e-10))

    return float(confluence_score)
