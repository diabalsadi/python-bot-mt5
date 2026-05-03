'''
To add an Exponential Moving Average (EMA) on MT5, go to Insert > Indicators > Trend > Moving Average,
set the "MA Method" to Exponential, and adjust the period (e.g., 50 or 200).
The EMA highlights trends by giving more weight to recent prices, with popular settings including the 8, 20, 50, and 200-day periods
'''
import MetaTrader5 as mt5


def calculate_ema(symbol: str, timeframe: int, shift: int, period: int) -> float:
    """
    Return the EMA value `shift` bars ago.

    The EMA is seeded with a simple average of the first `period` closes
    (oldest data), then the exponential multiplier k = 2 / (period + 1)
    is applied for every subsequent bar up to the target bar.

    Args:
        symbol:    Trading symbol
        timeframe: MT5 timeframe constant
        shift:     Bar offset (1 = last closed bar)
        period:    EMA period (MQL5 EMAPeriod, typically 50)

    Returns:
        EMA price, or 0.0 on data error.
    """
    # Extra warmup so the exponential smoothing is well-settled by the time
    # we reach the target bar.
    warmup = period * 3
    count  = shift + warmup + period + 2
    rates  = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)

    if rates is None or len(rates) < period + shift + 1:
        return 0.0

    # We work oldest → newest (natural MT5 order).
    # The bar we want is at index  (len - 1 - shift).
    closes     = rates["close"]
    target_idx = len(closes) - 1 - shift

    if target_idx < period:
        return 0.0

    k   = 2.0 / (period + 1)

    # Seed: SMA of first `period` bars
    ema = float(closes[:period].mean())

    # Smooth from bar `period` to `target_idx`
    for i in range(period, target_idx + 1):
        ema = closes[i] * k + ema * (1.0 - k)

    return ema
