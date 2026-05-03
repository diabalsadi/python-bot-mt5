"""
RSI Indicator
-------------
Relative Strength Index calculated directly from MT5 rate data (no external
indicator library required).  Uses Wilder's smoothing (simple-average seed)
matching MetaTrader 5's built-in iRSI behaviour.
Ported from MQL5: CalculateRSI(shift) via iRSI handle
"""

import MetaTrader5 as mt5
import numpy as np


def calculate_rsi(symbol: str, timeframe: int, shift: int, period: int) -> float:
    """
    Return the RSI value `shift` bars ago.

    Uses Wilder's initial average (SMA seed) then exponential smoothing for
    subsequent bars, which mirrors MT5's built-in RSI indicator.

    Args:
        symbol:    Trading symbol
        timeframe: MT5 timeframe constant
        shift:     Bar index (1 = last closed bar)
        period:    RSI period (MQL5 RSIPeriod, typically 14)

    Returns:
        RSI value in the range [0, 100].  Returns 50.0 (neutral) on error.
    """
    # We need `period + shift + 1` bars so there are enough price changes
    # to seed Wilder's average and then walk forward to `shift`.
    warmup = period * 3          # Extra warmup bars for Wilder smoothing
    count  = shift + warmup + period + 2
    rates  = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)

    if rates is None or len(rates) < period + shift + 2:
        return 50.0

    # Work oldest → newest (natural array order from MT5)
    closes = rates["close"]

    # We want the RSI value at the bar that is `shift` positions from the end.
    # "end" here means index -1-shift of the reversed view, i.e. index
    # (len - 1 - shift) in the normal view.
    target_idx = len(closes) - 1 - shift
    if target_idx < period:
        return 50.0

    # Compute RSI up to target_idx
    subset = closes[: target_idx + 1]
    deltas = np.diff(subset)

    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    if len(gains) < period:
        return 50.0

    # Wilder's seed (simple average of first `period` changes)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    # Wilder's smoothing for bars after the initial period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))
