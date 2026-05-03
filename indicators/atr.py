'''
# indicators/atr.py
The Average True Range (ATR) in MetaTrader 5 (MT5) is a built-in,
non-directional technical indicator used to measure market volatility by averaging the true range of price movement over a specified period, typically 14.
High ATR values indicate increased volatility, while low values suggest a calmer market, allowing traders to set volatility-based stop-losses and take-profit levels
'''
import MetaTrader5 as mt5
import numpy as np

def calculate_atr(symbol: str, timeframe: int, period: int, shift: int = 0) -> float:
    """
    Return the Average True Range for `period` bars, evaluated at `shift`
    bars ago.  True Range = max(H-L, |H-prevC|, |L-prevC|).

    Args:
        symbol:    Trading symbol
        timeframe: MT5 timeframe constant
        period:    ATR period (MQL5 ATRPeriod, typically 14)
        shift:     Bar offset (0 = current/forming bar)

    Returns:
        ATR value in price units, or 0.0 on data error.
    """
    # We need one extra bar for the previous close
    count = shift + period + 2
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)

    if rates is None or len(rates) < period + 2:
        return 0.0

    # Reverse: index 0 = current bar (shift 0)
    r = rates[::-1]

    true_ranges = []
    for i in range(shift, shift + period):
        high       = r[i]["high"]
        low        = r[i]["low"]
        prev_close = r[i + 1]["close"]          # one bar older
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)

    return float(np.mean(true_ranges)) if true_ranges else 0.0


def check_high_volatility(
    symbol: str,
    timeframe: int,
    atr_period: int,
    volatility_multiplier: float,
    avg_window: int = 50,
) -> tuple[bool, float, float]:
    """
    Compare the current ATR against its `avg_window`-bar rolling average.
    If current ATR > average * multiplier → high-volatility regime.

    Avoids N individual MT5 calls by fetching one large rate array and
    sliding the ATR window across it.

    Args:
        symbol:               Trading symbol
        timeframe:            MT5 timeframe constant
        atr_period:           ATR period
        volatility_multiplier: Threshold multiplier (MQL5 VolatilityMultiplier)
        avg_window:           Bars used for the rolling ATR average (default 50)

    Returns:
        (is_high_volatility, current_atr, average_atr)
    """
    # Fetch enough bars for both the current ATR and the avg_window
    count = avg_window + atr_period + 5
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)

    if rates is None or len(rates) < count:
        return False, 0.0, 0.0

    r = rates[::-1]   # index 0 = most recent bar

    def _atr_at(shift: int) -> float:
        trs = []
        for i in range(shift, shift + atr_period):
            if i + 1 >= len(r):
                break
            h, l, pc = r[i]["high"], r[i]["low"], r[i + 1]["close"]
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        return float(np.mean(trs)) if trs else 0.0

    current_atr = _atr_at(0)

    historical_atrs = [_atr_at(s) for s in range(avg_window)]
    average_atr     = float(np.mean(historical_atrs)) if historical_atrs else 0.0

    threshold   = average_atr * volatility_multiplier
    is_high_vol = current_atr > threshold

    return is_high_vol, current_atr, average_atr
