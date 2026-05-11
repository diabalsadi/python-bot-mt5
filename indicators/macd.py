"""
MACD (Moving Average Convergence Divergence)
-------------------------------------------
Standard 12, 26, 9 settings.
"""

import MetaTrader5 as mt5
from indicators.ema import calculate_ema

def calculate_macd(symbol: str, timeframe: int, shift: int = 0,
                   fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple:
    """
    Returns (macd_main, macd_signal) at the given shift.
    """
    # Macd Main = EMA(Fast) - EMA(Slow)
    # We need a series of Macd Main values to calculate the Signal (EMA of Main)
    
    macd_series = []
    # To calculate EMA(Signal) of length 9, we need ~20-30 macd values
    for i in range(shift, shift + signal_period + 20):
        fast_ema = calculate_ema(symbol, timeframe, i, fast_period)
        slow_ema = calculate_ema(symbol, timeframe, i, slow_period)
        if fast_ema == 0 or slow_ema == 0:
            macd_series.append(0.0)
        else:
            macd_series.append(fast_ema - slow_ema)
            
    if not macd_series or len(macd_series) < signal_period:
        return 0.0, 0.0
        
    # Macd Main is the first value (at shift)
    macd_main = macd_series[0]
    
    # Macd Signal is EMA of the macd_series
    # Lightweight EMA calculation
    alpha = 2.0 / (signal_period + 1.0)
    # Start with SMA of the last signal_period values as seed
    seed_subset = macd_series[-(signal_period):]
    ema_signal = sum(seed_subset) / len(seed_subset)
    
    # Iterate from oldest to newest (reverse macd_series)
    for val in reversed(macd_series[:-signal_period]):
        ema_signal = (val - ema_signal) * alpha + ema_signal
        
    return macd_main, ema_signal
