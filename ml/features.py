"""
Feature extraction and normalization for ML model.

Includes 15 features:
  1-9:   Core (momentum, volatility, trend, rsi, sr_distance, liquidity,
         volume_delta, spread_norm, bar_range_ratio)
  10-13: Advanced (macd, stochastic, adx, bollinger_prox)
  14:    SMC composite score (Order Blocks + FVG + BOS/CHoCH)
  15:    UT Bot signal (ATR trailing stop crossover)


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
from indicators.macd import calculate_macd
from indicators.stochastic import calculate_stochastic
from indicators.adx import calculate_adx
from indicators.bollinger_bands import calculate_bollinger_proximity
from indicators.smc import calculate_smc_score
from indicators.ut_bot import calculate_ut_bot_signal
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
    Extract and normalize all 15 features for ML model.
    
    Returns tuple of 15 normalized features.
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
    
    macd_raw = calculate_macd(symbol, timeframe, shift)
    stoch_raw = calculate_stochastic(symbol, timeframe, shift)
    adx_raw = calculate_adx(symbol, timeframe, shift)
    bb_prox_raw = calculate_bollinger_proximity(symbol, timeframe, shift)
    smc_raw = calculate_smc_score(symbol, timeframe, shift)
    ut_bot_raw = calculate_ut_bot_signal(symbol, timeframe, shift)

    # Normalize using historical mean/std from config
    stats = FEATURE_STATS
    
    def norm(val, key):
        if key in stats:
            return robust_normalize(val, stats[key]["mean"], stats[key]["std"])
        return robust_normalize(val, 0.0, 1.0)
    
    momentum = norm(momentum_raw, "momentum")
    volatility = norm(volatility_raw, "volatility")
    trend = norm(trend_raw, "trend")
    rsi = norm(rsi_raw, "rsi")
    sr_distance = norm(sr_distance_raw, "sr_distance")
    liquidity = norm(liquidity_raw, "liquidity")
    volume_delta = norm(volume_delta_raw, "volume_delta")
    spread_norm = norm(spread_norm_raw, "spread_norm")
    bar_range_ratio = norm(bar_range_ratio_raw, "bar_range_ratio")
    
    macd = norm(macd_raw, "macd")
    stochastic = norm(stoch_raw, "stochastic")
    adx = norm(adx_raw, "adx")
    bollinger_prox = norm(bb_prox_raw, "bollinger_prox")
    smc = norm(smc_raw, "smc")
    ut_bot = norm(ut_bot_raw, "ut_bot")

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
        macd,
        stochastic,
        adx,
        bollinger_prox,
        smc,
        ut_bot,
    )
