"""
ML Feature Extraction  (v2 — 9 features)
-----------------------------------------
Features fed to XGBoost and used to build the RL state.

Feature map:
  x1  momentum          — price rate-of-change over feature_window
  x2  volatility        — std of last 14 closes
  x3  trend_slope       — linear slope over feature_window
  x4  rsi               — RSI(14) [0-100]
  x5  sr_distance       — normalised distance to nearest S/R [-1, +1]
  x6  liquidity_score   — swing-point density near price [0, 1]
  x7  volume_delta      — tick-volume buy/sell pressure [-1, +1]  ← NEW
  x8  spread_norm       — spread / ATR(14)  [0, 5]               ← NEW
  x9  bar_range_ratio   — current range / 20-bar avg range        ← NEW

x7–x9 give XGBoost order-flow context the price chart alone can't see.
High spread_norm (>1) or bar_range_ratio (>2) are hard "don't trade" signals.
"""

import MetaTrader5 as mt5

from indicators.momentum           import calculate_momentum
from indicators.volatility         import calculate_volatility
from indicators.trend              import calculate_trend
from indicators.rsi                import calculate_rsi
from indicators.support_resistance import calculate_sr_distance
from indicators.liquidity_zones    import calculate_liquidity_score
from indicators.order_flow         import (
    calculate_volume_delta,
    calculate_spread_norm,
    calculate_bar_range_ratio,
)

FEATURE_NAMES = [
    "momentum",
    "volatility",
    "trend_slope",
    "rsi",
    "sr_distance",
    "liquidity",
    "volume_delta",
    "spread_norm",
    "bar_range_ratio",
]


def get_features(
    symbol: str,
    timeframe: int,
    shift: int,
    feature_window: int,
    rsi_period: int,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    """
    Return the 9 ML input features evaluated `shift` bars ago.

    Args:
        symbol:         Trading symbol
        timeframe:      MT5 timeframe constant (primary, e.g. M1)
        shift:          Bar offset (0 = forming bar, 1 = last closed)
        feature_window: Look-back for momentum / volatility / trend
        rsi_period:     RSI calculation period

    Returns:
        (momentum, volatility, trend_slope, rsi, sr_distance,
         liquidity, volume_delta, spread_norm, bar_range_ratio)
    """
    x1 = calculate_momentum(symbol, timeframe, shift, feature_window)
    x2 = calculate_volatility(symbol, timeframe, shift, feature_window)
    x3 = calculate_trend(symbol, timeframe, shift, feature_window)
    x4 = calculate_rsi(symbol, timeframe, shift, rsi_period)
    x5 = calculate_sr_distance(symbol, mt5.TIMEFRAME_M15, shift, lookback_bars=50)
    x6 = calculate_liquidity_score(symbol, mt5.TIMEFRAME_M15, shift, lookback_bars=50)
    x7 = calculate_volume_delta(symbol, timeframe, shift, feature_window)
    x8 = calculate_spread_norm(symbol, timeframe, shift)
    x9 = calculate_bar_range_ratio(symbol, timeframe, shift)

    return x1, x2, x3, x4, x5, x6, x7, x8, x9
