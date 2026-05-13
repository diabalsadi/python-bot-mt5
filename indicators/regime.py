"""
Regime Detector
---------------
Classifies the current market into one of three regimes:
  TRENDING_UP   — directional upward move
  TRENDING_DOWN — directional downward move
  RANGING       — choppy / sideways

How it works:
  1. ATR ratio:  current_ATR(14) / slow_ATR(50)
     > 1.3  → trending (expanding volatility)
     < 0.7  → ranging  (compressed volatility)
  2. ADX(14): > 25 confirms trend, < 20 confirms range
  3. EMA slope: direction of short EMA vs long EMA

All three are combined into a single confidence-weighted output.

Why this matters:
  - In RANGING markets, momentum/trend signals are noise — skip entries
  - In TRENDING markets, counter-trend signals are dangerous — filter harder
  - The trail_sltp liquidity gate only tightens in TRENDING regimes
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
import numpy as np
import MetaTrader5 as mt5


class Regime(Enum):
    TRENDING_UP   = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING       = "RANGING"
    UNKNOWN       = "UNKNOWN"


def detect_regime(
    symbol: str,
    timeframe: int,
    atr_fast: int  = 14,
    atr_slow: int  = 50,
    adx_period: int = 14,
    ema_fast: int  = 9,
    ema_slow: int  = 21,
) -> tuple[Regime, float]:
    """
    Detect the current market regime.

    Returns:
        (regime, confidence)
        confidence: 0.0–1.0 — how strongly the regime signal is
    """
    n = atr_slow + adx_period + ema_slow + 10
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if rates is None or len(rates) < atr_slow + 10:
        return Regime.UNKNOWN, 0.0

    highs  = rates["high"].astype(np.float64)
    lows   = rates["low"].astype(np.float64)
    closes = rates["close"].astype(np.float64)

    # ── 1. ATR ratio ──────────────────────────────────────────────────
    def _atr(period: int) -> float:
        trs = np.array([
            max(highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i]  - closes[i - 1]))
            for i in range(1, len(closes))
        ])
        return float(np.mean(trs[-period:])) if len(trs) >= period else float(np.mean(trs))

    atr_f = _atr(atr_fast)
    atr_s = _atr(atr_slow)
    atr_ratio = atr_f / atr_s if atr_s > 0 else 1.0

    # ── 2. ADX ────────────────────────────────────────────────────────
    def _adx(period: int) -> tuple[float, float, float]:
        """Returns (adx, +DI, -DI)"""
        plus_dm  = np.zeros(len(highs))
        minus_dm = np.zeros(len(highs))
        tr_arr   = np.zeros(len(highs))

        for i in range(1, len(highs)):
            up   = highs[i]  - highs[i - 1]
            down = lows[i - 1] - lows[i]
            plus_dm[i]  = up   if up > down and up > 0 else 0
            minus_dm[i] = down if down > up and down > 0 else 0
            tr_arr[i]   = max(highs[i] - lows[i],
                              abs(highs[i] - closes[i - 1]),
                              abs(lows[i]  - closes[i - 1]))

        def _smooth(arr, p):
            result = np.zeros(len(arr))
            result[p] = np.sum(arr[1:p + 1])
            for i in range(p + 1, len(arr)):
                result[i] = result[i - 1] - result[i - 1] / p + arr[i]
            return result

        atr14    = _smooth(tr_arr, period)
        pdm14    = _smooth(plus_dm, period)
        mdm14    = _smooth(minus_dm, period)

        with np.errstate(divide="ignore", invalid="ignore"):
            pdi = np.where(atr14 > 0, 100 * pdm14 / atr14, 0)
            mdi = np.where(atr14 > 0, 100 * mdm14 / atr14, 0)
            dx  = np.where((pdi + mdi) > 0,
                           100 * np.abs(pdi - mdi) / (pdi + mdi), 0)

        adx_val = float(np.mean(dx[-period:]))
        return adx_val, float(pdi[-1]), float(mdi[-1])

    adx, pdi, mdi = _adx(adx_period)

    # ── 3. EMA slope ──────────────────────────────────────────────────
    def _ema(arr: np.ndarray, period: int) -> np.ndarray:
        k = 2 / (period + 1)
        result = np.zeros(len(arr))
        result[0] = arr[0]
        for i in range(1, len(arr)):
            result[i] = arr[i] * k + result[i - 1] * (1 - k)
        return result

    ema_f  = _ema(closes, ema_fast)[-1]
    ema_s  = _ema(closes, ema_slow)[-1]
    ema_bullish = ema_f > ema_s

    # ── Combine ───────────────────────────────────────────────────────
    is_trending = atr_ratio > 1.15 and adx > 22
    is_ranging  = atr_ratio < 0.85 and adx < 22

    if is_ranging:
        confidence = min(1.0, (0.85 - atr_ratio) / 0.3 * 0.5 + (22 - adx) / 22 * 0.5)
        return Regime.RANGING, round(confidence, 3)

    if is_trending:
        confidence = min(1.0, (atr_ratio - 1.15) / 0.5 * 0.4 + (adx - 22) / 30 * 0.6)
        regime = Regime.TRENDING_UP if (ema_bullish and pdi > mdi) else Regime.TRENDING_DOWN
        return regime, round(confidence, 3)

    # Weak signal — use EMA to give best guess at low confidence
    regime = Regime.TRENDING_UP if ema_bullish else Regime.TRENDING_DOWN
    return regime, 0.3
