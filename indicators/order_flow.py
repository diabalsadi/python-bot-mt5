"""
Order Flow Features
-------------------
Tick-volume delta and spread width — two features the price chart
alone cannot provide, giving XGBoost signal the market chart doesn't.

tick_volume_delta:
  Difference between up-tick volume and down-tick volume over a window.
  Positive = more buying pressure, negative = more selling pressure.
  MT5 provides tick_volume per bar (proxy for real volume on CFDs).

spread_width_norm:
  Current spread as a fraction of ATR(14).
  High spread relative to ATR = broker is pricing in uncertainty.
  Very useful as a "don't trade this noise" signal.

session_volatility_ratio:
  Current bar's range vs. the 20-bar average range.
  > 2.0 = abnormal spike (news/manipulation), < 0.3 = dead market.
"""

from __future__ import annotations
import numpy as np
import MetaTrader5 as mt5


def calculate_volume_delta(
    symbol: str,
    timeframe: int,
    shift: int = 0,
    window: int = 14,
) -> float:
    """
    Compute normalised tick-volume delta over `window` bars.

    Returns value in [-1, 1]:
      +1 = all volume was on up-bars (strong buying)
      -1 = all volume was on down-bars (strong selling)
       0 = balanced

    Args:
        symbol:    Trading symbol
        timeframe: MT5 timeframe constant
        shift:     Bar offset (0 = forming bar)
        window:    Look-back window
    """
    n = shift + window + 5
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if rates is None or len(rates) < window:
        return 0.0

    closes = rates["close"][::-1]
    ticks  = rates["tick_volume"][::-1].astype(np.float64)

    up_vol   = 0.0
    down_vol = 0.0
    for i in range(shift, shift + window):
        if i + 1 >= len(closes):
            break
        if closes[i] > closes[i + 1]:        # up bar
            up_vol   += ticks[i]
        elif closes[i] < closes[i + 1]:      # down bar
            down_vol += ticks[i]

    total = up_vol + down_vol
    if total == 0:
        return 0.0
    return float(np.clip((up_vol - down_vol) / total, -1.0, 1.0))


def calculate_spread_norm(
    symbol: str,
    timeframe: int,
    shift: int = 0,
    atr_period: int = 14,
) -> float:
    """
    Spread width normalised by ATR(14).

    Returns value in [0, 5]:
      < 0.1  = very tight spread (good liquidity)
      > 1.0  = spread is wider than ATR (dangerous to trade)

    Args:
        symbol:    Trading symbol
        timeframe: MT5 timeframe constant
        shift:     Bar offset
        atr_period: ATR period for normalisation
    """
    tick = mt5.symbol_info_tick(symbol)
    sym  = mt5.symbol_info(symbol)
    if tick is None or sym is None:
        return 0.0

    spread = (tick.ask - tick.bid)
    point  = sym.point

    n = atr_period + shift + 5
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if rates is None or len(rates) < atr_period:
        return 0.0

    h = rates["high"].astype(np.float64)
    l = rates["low"].astype(np.float64)
    c = rates["close"].astype(np.float64)
    trs = [max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
           for i in range(1, len(h))]
    atr = float(np.mean(trs[-atr_period:]))

    if atr < point:
        return 0.0
    return float(np.clip(spread / atr, 0.0, 5.0))


def calculate_bar_range_ratio(
    symbol: str,
    timeframe: int,
    shift: int = 0,
    avg_period: int = 20,
) -> float:
    """
    Current bar's high-low range relative to 20-bar average range.

    > 2.0 = abnormal spike — news event or manipulation → avoid
    < 0.3 = dead market → avoid (spread dominates)
    ~1.0  = normal conditions

    Returns value clipped to [0, 5].
    """
    n = avg_period + shift + 5
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if rates is None or len(rates) < avg_period:
        return 1.0

    h = rates["high"][::-1].astype(np.float64)
    l = rates["low"][::-1].astype(np.float64)

    current_range = float(h[shift] - l[shift])
    avg_range     = float(np.mean(h[shift+1:shift+avg_period+1] -
                                   l[shift+1:shift+avg_period+1]))
    if avg_range < 1e-9:
        return 1.0
    return float(np.clip(current_range / avg_range, 0.0, 5.0))
