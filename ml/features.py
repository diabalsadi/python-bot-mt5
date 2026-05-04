"""
ML Feature Extraction
---------------------
Bundles six indicator values that feed the linear regression model
into a single convenience call.

Features (enhanced with S/R and liquidity):
  x1 = momentum              (CalculateMomentum)
  x2 = volatility            (CalculateVolatility)
  x3 = trend slope           (CalculateTrend)
  x4 = RSI                   (CalculateRSI)
  x5 = SR distance (15m)     (from 15-minute support/resistance)
  x6 = Liquidity score       (from liquidity zones)
"""

import MetaTrader5 as mt5

from indicators.momentum   import calculate_momentum
from indicators.volatility import calculate_volatility
from indicators.trend      import calculate_trend
from indicators.rsi        import calculate_rsi
from indicators.support_resistance import calculate_sr_distance
from indicators.liquidity_zones import calculate_liquidity_score


def get_features(
    symbol: str,
    timeframe: int,
    shift: int,
    feature_window: int,
    rsi_period: int,
) -> tuple[float, float, float, float, float, float]:
    """
    Return the six ML input features evaluated `shift` bars ago.

    Args:
        symbol:         Trading symbol
        timeframe:      MT5 timeframe constant (primary, e.g. M1)
        shift:          Bar offset (1 = last closed bar)
        feature_window: Look-back window used by momentum / volatility / trend
        rsi_period:     RSI calculation period

    Returns:
        (momentum, volatility, trend_slope, rsi, sr_distance, liquidity)  — all floats
    """
    x1 = calculate_momentum(symbol, timeframe, shift, feature_window)
    x2 = calculate_volatility(symbol, timeframe, shift, feature_window)
    x3 = calculate_trend(symbol, timeframe, shift, feature_window)
    x4 = calculate_rsi(symbol, timeframe, shift, rsi_period)
    
    # New features from 15m timeframe
    x5 = calculate_sr_distance(symbol, mt5.TIMEFRAME_M15, shift, lookback_bars=50)
    x6 = calculate_liquidity_score(symbol, mt5.TIMEFRAME_M15, shift, lookback_bars=50)

    return x1, x2, x3, x4, x5, x6
