"""
Ichimoku Kinko Hyo Indicator
---------------------------
Calculates Tenkan-sen, Kijun-sen, and Senkou Spans A/B.
Used for trend confirmation and support/resistance cloud detection.
"""

import MetaTrader5 as mt5
import numpy as np

def calculate_ichimoku(symbol: str, timeframe: int, shift: int = 0, 
                       tenkan_period: int = 9, kijun_period: int = 26, 
                       senkou_period: int = 52) -> dict:
    """
    Returns Ichimoku values at the given shift.
    - tenkan:  Conversion Line
    - kijun:   Base Line
    - span_a:  Leading Span A (at current price, i.e. shifted back 26)
    - span_b:  Leading Span B (at current price)
    """
    # We need enough history for the spans + displacement
    count = shift + senkou_period + 26 + 10
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) < count:
        return {}

    rates_rev = rates[::-1] # 0 = current
    
    def get_hl_avg(period, start_idx):
        subset = rates_rev[start_idx : start_idx + period]
        high = np.max(subset["high"])
        low = np.min(subset["low"])
        return (high + low) / 2.0

    # Current values
    tenkan = get_hl_avg(tenkan_period, shift)
    kijun = get_hl_avg(kijun_period, shift)
    
    # Spans are displaced forward by 26 bars. 
    # To get the cloud at the CURRENT price, we look at calculations from 26 bars ago.
    span_a_raw = (get_hl_avg(tenkan_period, shift + 26) + get_hl_avg(kijun_period, shift + 26)) / 2.0
    span_b_raw = get_hl_avg(senkou_period, shift + 26)

    return {
        "tenkan": tenkan,
        "kijun": kijun,
        "span_a": span_a_raw,
        "span_b": span_b_raw
    }
