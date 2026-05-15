"""
ADX (Average Directional Index)
-------------------------------
Measures the strength of a trend.
"""

import MetaTrader5 as mt5
import numpy as np

def calculate_adx(symbol: str, timeframe: int, shift: int, period: int = 14) -> float:
    """
    Returns ADX value normalized to [-1, 1].
    ADX ranges from 0 to 100. 
    > 25 is considered trending.
    We normalize it by subtracting 25 and dividing by 25.
    """
    count = shift + period * 3 + 10
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)

    if rates is None or len(rates) < count:
        return 0.0

    highs = rates["high"]
    lows = rates["low"]
    closes = rates["close"]

    tr = np.zeros(len(closes))
    pdm = np.zeros(len(closes))
    ndm = np.zeros(len(closes))

    for i in range(1, len(closes)):
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i-1])
        tr3 = abs(lows[i] - closes[i-1])
        tr[i] = max(tr1, tr2, tr3)

        up_move = highs[i] - highs[i-1]
        down_move = lows[i-1] - lows[i]

        if up_move > down_move and up_move > 0:
            pdm[i] = up_move
        else:
            pdm[i] = 0

        if down_move > up_move and down_move > 0:
            ndm[i] = down_move
        else:
            ndm[i] = 0

    def smooth(data, period):
        smoothed = np.zeros(len(data))
        smoothed[period] = np.sum(data[1:period+1])
        for i in range(period+1, len(data)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + data[i]
        return smoothed

    tr_smoothed = smooth(tr, period)
    pdm_smoothed = smooth(pdm, period)
    ndm_smoothed = smooth(ndm, period)

    # Avoid division by zero
    tr_smoothed[tr_smoothed == 0] = 1e-10

    pdi = 100 * (pdm_smoothed / tr_smoothed)
    ndi = 100 * (ndm_smoothed / tr_smoothed)

    dx = 100 * (abs(pdi - ndi) / (pdi + ndi + 1e-10))
    
    adx = np.zeros(len(dx))
    adx[period*2] = np.mean(dx[period+1 : period*2+1])
    for i in range(period*2 + 1, len(dx)):
        adx[i] = ((adx[i-1] * (period - 1)) + dx[i]) / period

    target_idx = len(closes) - 1 - shift
    adx_val = adx[target_idx]
    
    # Normalize: center around 25. Return range roughly [-1, 2]
    return float((adx_val - 25.0) / 25.0)
