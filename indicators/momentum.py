"""
Momentum Indicator
------------------
Measures price momentum as a percentage change over `feature_window` bars.
Ported from MQL5: CalculateMomentum(shift)

In MQL5 bar indexing, shift=0 is the current (forming) bar, shift=1 is
the last fully closed bar, etc.  The Python equivalent reverses the rates
array so index 0 == current bar, index 1 == 1 bar ago, and so on.
"""

import MetaTrader5 as mt5


def calculate_momentum(symbol: str, timeframe: int, shift: int, feature_window: int) -> float:
    """
    Return the percentage price change between the bar at `shift` and the bar
    at `shift + feature_window`.

    Args:
        symbol:         Trading symbol, e.g. "BTCUSD"
        timeframe:      MT5 timeframe constant, e.g. mt5.TIMEFRAME_M1
        shift:          Starting bar (1 = last closed bar)
        feature_window: Look-back window in bars

    Returns:
        Momentum as a percentage (positive = bullish, negative = bearish).
        Returns 0.0 on data error.
    """
    count = shift + feature_window + 2
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)

    if rates is None or len(rates) < count:
        return 0.0

    # Reverse so index 0 = current bar, index 1 = 1 bar ago, …
    closes = rates["close"][::-1]

    close_recent = closes[shift]
    close_past   = closes[shift + feature_window]

    if close_past == 0:
        return 0.0

    return (close_recent - close_past) / close_past * 100.0
