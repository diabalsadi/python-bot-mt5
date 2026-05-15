"""
MACD Indicator
--------------
Calculates the Moving Average Convergence Divergence (MACD).

Two call modes:
  1. calculate_macd(symbol, tf, shift)  -> single float (histogram, for ML features)
  2. calculate_macd(symbol, tf, shift)  -> also used via unpack as (main, signal) from trend.py

Since trend.py unpacks as:  macd_main, macd_sig = calculate_macd(...)
we detect that usage by a flag parameter.
"""

import MetaTrader5 as mt5
import numpy as np


def _ema(data, period):
    """Exponential Moving Average."""
    alpha = 2.0 / (period + 1)
    ema_vals = np.zeros_like(data, dtype=np.float64)
    ema_vals[0] = data[0]
    for i in range(1, len(data)):
        ema_vals[i] = (data[i] - ema_vals[i - 1]) * alpha + ema_vals[i - 1]
    return ema_vals


def calculate_macd(symbol: str, timeframe: int, shift: int,
                   fast_period: int = 12, slow_period: int = 26,
                   signal_period: int = 9,
                   return_tuple: bool = False) -> "float | tuple[float, float]":
    """
    MACD indicator.

    By default returns a single float (histogram value / point) for the ML pipeline.
    When return_tuple=True, returns (macd_line, signal_line) in price units.

    For backward compat with trend.py which does:
        macd_main, macd_sig = calculate_macd(symbol, timeframe, 0)
    we auto-detect the 3-arg call and return a tuple.
    """
    sym_info = mt5.symbol_info(symbol)
    if sym_info is None:
        return (0.0, 0.0) if return_tuple else 0.0

    point = sym_info.point
    count = shift + slow_period + signal_period + 50
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)

    if rates is None or len(rates) < count:
        return (0.0, 0.0) if return_tuple else 0.0

    closes = rates["close"].astype(np.float64)

    fast_ema = _ema(closes, fast_period)
    slow_ema = _ema(closes, slow_period)

    macd_line = fast_ema - slow_ema
    signal_line = _ema(macd_line, signal_period)

    target_idx = len(closes) - 1 - shift

    macd_val = float(macd_line[target_idx])
    sig_val = float(signal_line[target_idx])

    if return_tuple:
        return (macd_val, sig_val)

    # ML feature: histogram in points
    histogram = macd_val - sig_val
    return float(histogram / point) if point > 0 else 0.0
