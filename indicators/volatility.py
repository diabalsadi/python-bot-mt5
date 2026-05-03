"""
Volatility Indicator
--------------------
Measures average bar range (High - Low) over `feature_window` bars, expressed
in points — a lightweight ATR-like metric used by the ML feature pipeline.
Ported from MQL5: CalculateVolatility(shift)
"""

import MetaTrader5 as mt5
import numpy as np


def calculate_volatility(symbol: str, timeframe: int, shift: int, feature_window: int) -> float:
    """
    Return the average (High - Low) range over `feature_window` bars starting
    at `shift`, expressed in symbol points.

    Args:
        symbol:         Trading symbol
        timeframe:      MT5 timeframe constant
        shift:          Starting bar index (1 = last closed bar)
        feature_window: Number of bars to average

    Returns:
        Average range in points, or 0.0 on data error.
    """
    sym_info = mt5.symbol_info(symbol)
    if sym_info is None:
        return 0.0

    point = sym_info.point
    count = shift + feature_window + 2
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)

    if rates is None or len(rates) < count:
        return 0.0

    # Reverse: index 0 = current bar
    rates_rev = rates[::-1]

    highs = np.array([rates_rev[i]["high"] for i in range(shift, shift + feature_window)])
    lows  = np.array([rates_rev[i]["low"]  for i in range(shift, shift + feature_window)])

    avg_range = np.mean(highs - lows)
    return avg_range / point if point > 0 else 0.0
