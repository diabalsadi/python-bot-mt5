"""
Bollinger Bands Proximity
-------------------------
Measures how close the current price is to the Bollinger Bands.
"""

import MetaTrader5 as mt5
import numpy as np

def calculate_bollinger_proximity(symbol: str, timeframe: int, shift: int, period: int = 20, std_dev: float = 2.0) -> float:
    """
    Returns the normalized position of the price within the Bollinger Bands.
    0.0 = on the lower band
    0.5 = on the moving average
    1.0 = on the upper band
    
    We subtract 0.5 and multiply by 2 to center it at 0.0 with range [-1.0, 1.0].
    Values > 1.0 mean price is above the upper band.
    """
    count = shift + period + 5
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)

    if rates is None or len(rates) < count:
        return 0.0

    closes = rates["close"]

    target_idx = len(closes) - 1 - shift
    window = closes[target_idx - period + 1 : target_idx + 1]
    
    ma = np.mean(window)
    std = np.std(window)
    
    if std == 0:
        return 0.0
        
    upper_band = ma + (std * std_dev)
    lower_band = ma - (std * std_dev)
    
    current_close = closes[target_idx]
    
    position = (current_close - lower_band) / (upper_band - lower_band)
    
    # Normalize to [-1, 1]
    return float((position - 0.5) * 2.0)
