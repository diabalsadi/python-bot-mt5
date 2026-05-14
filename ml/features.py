"""
Feature extraction and normalization for ML model.

CRITICAL: All features are normalized before returning to prevent
the "exploding features" problem (rsi=1767, spread_norm=-2105, etc.).

Normalization process:
  1. Calculate raw feature value
  2. Subtract historical mean
  3. Divide by historical std
  4. Clip to [-5, 5] to prevent outliers

This must match training and inference exactly.
"""

import numpy as np
import MetaTrader5 as mt5
from indicators.liquidity_zones import calculate_liquidity_score
from indicators.momentum import calculate_momentum
from indicators.order_flow import calculate_bar_range_ratio, calculate_spread_norm, calculate_volume_delta
from indicators.rsi import calculate_rsi
from indicators.support_resistance import calculate_sr_distance
from indicators.trend import calculate_trend
from indicators.volatility import calculate_volatility
from utils.normalization import robust_normalize
from config.feature_stats import FEATURE_STATS


def get_features(
    symbol,
    timeframe,
    shift,
    feature_window,
    rsi_period,
):
    """
    Extract and normalize all 9 features for ML model.
    
    Returns tuple of 9 normalized features:
    (momentum, volatility, trend, rsi, sr_distance, liquidity, volume_delta, spread_norm, bar_range_ratio)
    
    All values are in range [-5, 5] after robust normalization.
    """
    
    # Raw feature calculations
    momentum_raw = calculate_momentum(symbol, timeframe, shift, feature_window)
    volatility_raw = calculate_volatility(symbol, timeframe, shift, feature_window)
    trend_raw = calculate_trend(symbol, timeframe, shift, feature_window)
    rsi_raw = (calculate_rsi(symbol, timeframe, shift, rsi_period) - 50) / 10
    sr_distance_raw = calculate_sr_distance(symbol, mt5.TIMEFRAME_M15, shift, 50)
    liquidity_raw = calculate_liquidity_score(symbol, mt5.TIMEFRAME_M15, shift, 50)
    volume_delta_raw = calculate_volume_delta(symbol, timeframe, shift, feature_window)
    spread_norm_raw = calculate_spread_norm(symbol, timeframe, shift)
    bar_range_ratio_raw = calculate_bar_range_ratio(symbol, timeframe, shift)

    # Normalize using historical mean/std from config
    stats = FEATURE_STATS
    
    momentum = robust_normalize(
        momentum_raw,
        stats["momentum"]["mean"],
        stats["momentum"]["std"]
    )
    
    volatility = robust_normalize(
        volatility_raw,
        stats["volatility"]["mean"],
        stats["volatility"]["std"]
    )
    
    trend = robust_normalize(
        trend_raw,
        stats["trend"]["mean"],
        stats["trend"]["std"]
    )
    
    rsi = robust_normalize(
        rsi_raw,
        stats["rsi"]["mean"],
        stats["rsi"]["std"]
    )
    
    sr_distance = robust_normalize(
        sr_distance_raw,
        stats["sr_distance"]["mean"],
        stats["sr_distance"]["std"]
    )
    
    liquidity = robust_normalize(
        liquidity_raw,
        stats["liquidity"]["mean"],
        stats["liquidity"]["std"]
    )
    
    volume_delta = robust_normalize(
        volume_delta_raw,
        stats["volume_delta"]["mean"],
        stats["volume_delta"]["std"]
    )
    
    spread_norm = robust_normalize(
        spread_norm_raw,
        stats["spread_norm"]["mean"],
        stats["spread_norm"]["std"]
    )
    
    bar_range_ratio = robust_normalize(
        bar_range_ratio_raw,
        stats["bar_range_ratio"]["mean"],
        stats["bar_range_ratio"]["std"]
    )

    return (
        momentum,
        volatility,
        trend,
        rsi,
        sr_distance,
        liquidity,
        volume_delta,
        spread_norm,
        bar_range_ratio,
    )
