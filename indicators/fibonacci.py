import MetaTrader5 as mt5

def get_m30_fibo_levels(symbol: str) -> dict:
    """
    Fetch the last closed M30 bar and calculate Fibonacci retracement levels.
    
    Returns a dictionary of levels, or an empty dict if data is unavailable.
    """
    # Fetch the last completely closed M30 bar (index 1)
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M30, 1, 1)
    if rates is None or len(rates) == 0:
        return {}
        
    bar = rates[0]
    high = float(bar['high'])
    low = float(bar['low'])
    open_price = float(bar['open'])
    close_price = float(bar['close'])
    
    diff = high - low
    if diff == 0:
        return {}
        
    # Determine the direction of the bar to know which way to draw the fibo
    is_bullish = close_price >= open_price
    
    levels = {}
    fibo_ratios = [0.236, 0.382, 0.500, 0.618, 0.786]
    
    if is_bullish:
        # For a bullish bar, we retrace down from the High.
        # Support levels are calculated below the High.
        for ratio in fibo_ratios:
            levels[f"pullback_{int(ratio*100)}"] = high - (diff * ratio)
    else:
        # For a bearish bar, we retrace up from the Low.
        # Resistance levels are calculated above the Low.
        for ratio in fibo_ratios:
            levels[f"pullback_{int(ratio*100)}"] = low + (diff * ratio)
            
    # Also return the raw high and low for reference
    levels["high"] = high
    levels["low"] = low
    levels["is_bullish"] = is_bullish
    
    return levels
