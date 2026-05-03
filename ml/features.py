"""
ML Feature Extraction
---------------------
Bundles the four indicator values that feed the linear regression model
into a single convenience call.

Features (identical to MQL5 naming):
  x1 = momentum   (CalculateMomentum)
  x2 = volatility  (CalculateVolatility)
  x3 = trend slope (CalculateTrend)
  x4 = RSI         (CalculateRSI)
"""

from indicators.momentum   import calculate_momentum
from indicators.volatility import calculate_volatility
from indicators.trend      import calculate_trend
from indicators.rsi        import calculate_rsi


def get_features(
    symbol: str,
    timeframe: int,
    shift: int,
    feature_window: int,
    rsi_period: int,
) -> tuple[float, float, float, float]:
    """
    Return the four ML input features evaluated `shift` bars ago.

    Args:
        symbol:         Trading symbol
        timeframe:      MT5 timeframe constant
        shift:          Bar offset (1 = last closed bar)
        feature_window: Look-back window used by momentum / volatility / trend
        rsi_period:     RSI calculation period

    Returns:
        (momentum, volatility, trend_slope, rsi)  — all floats
    """
    x1 = calculate_momentum(symbol, timeframe, shift, feature_window)
    x2 = calculate_volatility(symbol, timeframe, shift, feature_window)
    x3 = calculate_trend(symbol, timeframe, shift, feature_window)
    x4 = calculate_rsi(symbol, timeframe, shift, rsi_period)

    return x1, x2, x3, x4
