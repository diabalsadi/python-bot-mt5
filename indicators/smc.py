"""
Smart Money Concepts (SMC) Indicator
-------------------------------------
Institutional order-flow analysis tools:

1. Order Blocks (OB)
   - Bullish OB: last bearish candle before a strong bullish move
   - Bearish OB: last bullish candle before a strong bearish move
   - Returns proximity score to nearest OB (good entry = price at OB)

2. Fair Value Gaps (FVG) / Imbalances
   - 3-candle pattern where candle 1 high < candle 3 low (bullish FVG)
   - or candle 1 low > candle 3 high (bearish FVG)
   - Unfilled FVGs act as magnets for price

3. Break of Structure (BOS) / Change of Character (CHoCH)
   - BOS: price breaks a swing high/low in the direction of trend → continuation
   - CHoCH: price breaks a swing high/low AGAINST the trend → reversal signal

4. calculate_smc_score()
   - Composite score [-1, 1] combining all SMC concepts
   - Positive = bullish institutional bias, Negative = bearish
"""

from __future__ import annotations

import MetaTrader5 as mt5
import numpy as np


# ── Swing Detection Helper ────────────────────────────────────────────

def _find_swing_points(
    highs: np.ndarray, lows: np.ndarray, strength: int = 3
) -> tuple[list[dict], list[dict]]:
    """
    Identify swing highs and swing lows using left/right bar comparison.

    Args:
        highs:    Array of high prices (index 0 = most recent)
        lows:     Array of low prices (index 0 = most recent)
        strength: Number of bars on each side that must be lower/higher

    Returns:
        (swing_highs, swing_lows) — lists of {"index": int, "price": float}
    """
    swing_highs = []
    swing_lows = []

    for i in range(strength, len(highs) - strength):
        # Swing High: current high is higher than `strength` bars on each side
        is_swing_high = True
        for j in range(1, strength + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_swing_high = False
                break
        if is_swing_high:
            swing_highs.append({"index": i, "price": float(highs[i])})

        # Swing Low: current low is lower than `strength` bars on each side
        is_swing_low = True
        for j in range(1, strength + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_swing_low = False
                break
        if is_swing_low:
            swing_lows.append({"index": i, "price": float(lows[i])})

    return swing_highs, swing_lows


# ── Order Blocks ──────────────────────────────────────────────────────

def detect_order_blocks(
    symbol: str,
    timeframe: int,
    shift: int = 0,
    lookback: int = 50,
    min_impulse_multiplier: float = 1.5,
) -> list[dict]:
    """
    Detect Order Blocks — the last opposing candle before a strong impulse move.

    A Bullish OB is the last bearish (red) candle before a strong bullish move.
    A Bearish OB is the last bullish (green) candle before a strong bearish move.

    Args:
        symbol:     Trading symbol
        timeframe:  MT5 timeframe constant
        shift:      Bar offset
        lookback:   Number of bars to scan
        min_impulse_multiplier: Minimum impulse size as multiple of average range

    Returns:
        List of OB dicts: {"type": "bullish"|"bearish", "top": float, "bottom": float,
                           "index": int, "mitigated": bool}
    """
    count = shift + lookback + 10
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) < lookback + 5:
        return []

    opens = rates["open"][::-1].astype(np.float64)
    highs = rates["high"][::-1].astype(np.float64)
    lows = rates["low"][::-1].astype(np.float64)
    closes = rates["close"][::-1].astype(np.float64)

    # Average candle range for impulse threshold
    ranges = highs[shift:shift + lookback] - lows[shift:shift + lookback]
    avg_range = float(np.mean(ranges)) if len(ranges) > 0 else 0.0
    impulse_threshold = avg_range * min_impulse_multiplier

    order_blocks = []

    for i in range(shift + 1, shift + lookback - 2):
        if i + 2 >= len(closes):
            break

        candle_body = closes[i] - opens[i]
        next_body = closes[i - 1] - opens[i - 1]
        next_range = highs[i - 1] - lows[i - 1]

        # Bullish OB: bearish candle (close < open) followed by strong bullish move
        if candle_body < 0 and next_body > 0 and next_range > impulse_threshold:
            # Check if OB has been mitigated (price returned to OB zone)
            ob_top = max(opens[i], closes[i])
            ob_bottom = min(opens[i], closes[i])
            mitigated = any(lows[j] <= ob_top for j in range(shift, i))

            order_blocks.append({
                "type": "bullish",
                "top": float(ob_top),
                "bottom": float(ob_bottom),
                "index": i,
                "mitigated": mitigated,
            })

        # Bearish OB: bullish candle (close > open) followed by strong bearish move
        elif candle_body > 0 and next_body < 0 and next_range > impulse_threshold:
            ob_top = max(opens[i], closes[i])
            ob_bottom = min(opens[i], closes[i])
            mitigated = any(highs[j] >= ob_bottom for j in range(shift, i))

            order_blocks.append({
                "type": "bearish",
                "top": float(ob_top),
                "bottom": float(ob_bottom),
                "index": i,
                "mitigated": mitigated,
            })

    return order_blocks[:10]  # Return most recent 10


def calculate_ob_proximity(
    symbol: str, timeframe: int, shift: int = 0, lookback: int = 50
) -> float:
    """
    Calculate proximity to the nearest unmitigated Order Block.

    Returns:
        Score in [-1, 1]:
        +1 = price sitting right on a bullish OB (strong buy zone)
        -1 = price sitting right on a bearish OB (strong sell zone)
         0 = no nearby OB or price is far from any OB
    """
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, shift + 5)
    if rates is None or len(rates) < shift + 1:
        return 0.0

    current_price = float(rates["close"][::-1][shift])

    obs = detect_order_blocks(symbol, timeframe, shift, lookback)
    if not obs:
        return 0.0

    best_score = 0.0

    for ob in obs:
        if ob["mitigated"]:
            continue  # Skip already-tested OBs

        ob_mid = (ob["top"] + ob["bottom"]) / 2.0
        ob_range = ob["top"] - ob["bottom"]

        if ob_range < 1e-10:
            continue

        distance = abs(current_price - ob_mid)
        # Normalize: 0 = right at OB, 1 = one OB-width away
        proximity = max(0.0, 1.0 - distance / (ob_range * 3.0))

        if ob["type"] == "bullish" and current_price <= ob["top"]:
            score = proximity  # positive = bullish OB nearby below price
        elif ob["type"] == "bearish" and current_price >= ob["bottom"]:
            score = -proximity  # negative = bearish OB nearby above price
        else:
            score = 0.0

        if abs(score) > abs(best_score):
            best_score = score

    return float(np.clip(best_score, -1.0, 1.0))


# ── Fair Value Gaps (FVG) ─────────────────────────────────────────────

def detect_fvg(
    symbol: str, timeframe: int, shift: int = 0, lookback: int = 50
) -> list[dict]:
    """
    Detect Fair Value Gaps (imbalances) in price action.

    Bullish FVG: candle[i+1].high < candle[i-1].low  (gap up)
    Bearish FVG: candle[i+1].low  > candle[i-1].high (gap down)

    Returns:
        List of FVG dicts: {"type": "bullish"|"bearish", "top": float,
                            "bottom": float, "index": int, "filled": bool}
    """
    count = shift + lookback + 10
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) < lookback + 5:
        return []

    highs = rates["high"][::-1].astype(np.float64)
    lows = rates["low"][::-1].astype(np.float64)

    fvgs = []

    for i in range(shift + 1, shift + lookback - 1):
        if i + 1 >= len(highs) or i - 1 < 0:
            break

        # Bullish FVG: gap between candle before and candle after
        # The low of the candle after (i-1) is above the high of the candle before (i+1)
        if lows[i - 1] > highs[i + 1]:
            fvg_top = float(lows[i - 1])
            fvg_bottom = float(highs[i + 1])

            # Check if FVG has been filled (price entered the gap)
            filled = any(
                lows[j] <= fvg_top and highs[j] >= fvg_bottom
                for j in range(shift, i - 1)
            )

            fvgs.append({
                "type": "bullish",
                "top": fvg_top,
                "bottom": fvg_bottom,
                "index": i,
                "filled": filled,
            })

        # Bearish FVG
        elif highs[i - 1] < lows[i + 1]:
            fvg_top = float(lows[i + 1])
            fvg_bottom = float(highs[i - 1])

            filled = any(
                highs[j] >= fvg_bottom and lows[j] <= fvg_top
                for j in range(shift, i - 1)
            )

            fvgs.append({
                "type": "bearish",
                "top": fvg_top,
                "bottom": fvg_bottom,
                "index": i,
                "filled": filled,
            })

    return fvgs[:10]


def calculate_fvg_score(
    symbol: str, timeframe: int, shift: int = 0, lookback: int = 50
) -> float:
    """
    Score based on proximity to unfilled Fair Value Gaps.

    Returns:
        Score in [-1, 1]:
        +1 = price near unfilled bullish FVG (magnet pulling price up)
        -1 = price near unfilled bearish FVG (magnet pulling price down)
         0 = no relevant FVGs
    """
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, shift + 5)
    if rates is None or len(rates) < shift + 1:
        return 0.0

    current_price = float(rates["close"][::-1][shift])
    fvgs = detect_fvg(symbol, timeframe, shift, lookback)

    if not fvgs:
        return 0.0

    best_score = 0.0

    for fvg in fvgs:
        if fvg["filled"]:
            continue

        fvg_mid = (fvg["top"] + fvg["bottom"]) / 2.0
        fvg_size = fvg["top"] - fvg["bottom"]

        if fvg_size < 1e-10:
            continue

        distance = abs(current_price - fvg_mid)
        proximity = max(0.0, 1.0 - distance / (fvg_size * 5.0))

        if fvg["type"] == "bullish":
            # Bullish FVG below price = magnet pulling price down to fill,
            # then bounce up → bullish bias
            score = proximity if current_price >= fvg_mid else proximity * 0.5
        else:
            score = -proximity if current_price <= fvg_mid else -proximity * 0.5

        if abs(score) > abs(best_score):
            best_score = score

    return float(np.clip(best_score, -1.0, 1.0))


# ── Break of Structure / Change of Character ─────────────────────────

def detect_structure_breaks(
    symbol: str, timeframe: int, shift: int = 0, lookback: int = 50
) -> dict:
    """
    Detect Break of Structure (BOS) and Change of Character (CHoCH).

    BOS:   Price breaks the most recent swing high/low in trend direction
           → trend continuation signal
    CHoCH: Price breaks the most recent swing high/low AGAINST trend
           → potential reversal signal

    Returns:
        {
            "bos_bullish": bool,    # Bullish BOS detected
            "bos_bearish": bool,    # Bearish BOS detected
            "choch_bullish": bool,  # Bullish CHoCH (reversal from bearish)
            "choch_bearish": bool,  # Bearish CHoCH (reversal from bullish)
            "structure_score": float  # Composite score [-1, 1]
        }
    """
    count = shift + lookback + 10
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) < lookback + 5:
        return {
            "bos_bullish": False, "bos_bearish": False,
            "choch_bullish": False, "choch_bearish": False,
            "structure_score": 0.0,
        }

    highs = rates["high"][::-1].astype(np.float64)
    lows = rates["low"][::-1].astype(np.float64)
    closes = rates["close"][::-1].astype(np.float64)

    swing_highs, swing_lows = _find_swing_points(highs, lows, strength=3)

    result = {
        "bos_bullish": False, "bos_bearish": False,
        "choch_bullish": False, "choch_bearish": False,
        "structure_score": 0.0,
    }

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return result

    # Determine recent trend from swing structure
    # Bullish trend: higher highs and higher lows
    # Bearish trend: lower highs and lower lows
    recent_hh = swing_highs[0]["price"] > swing_highs[1]["price"]
    recent_hl = swing_lows[0]["price"] > swing_lows[1]["price"]
    recent_lh = swing_highs[0]["price"] < swing_highs[1]["price"]
    recent_ll = swing_lows[0]["price"] < swing_lows[1]["price"]

    bullish_trend = recent_hh and recent_hl
    bearish_trend = recent_lh and recent_ll

    current_price = float(closes[shift])

    # Check for BOS / CHoCH at current price
    nearest_swing_high = swing_highs[0]["price"]
    nearest_swing_low = swing_lows[0]["price"]

    # BOS Bullish: price breaks above recent swing high in a bullish trend
    if current_price > nearest_swing_high:
        if bullish_trend:
            result["bos_bullish"] = True
        else:
            # Breaking high against bearish trend = CHoCH bullish (reversal)
            result["choch_bullish"] = True

    # BOS Bearish: price breaks below recent swing low in a bearish trend
    if current_price < nearest_swing_low:
        if bearish_trend:
            result["bos_bearish"] = True
        else:
            # Breaking low against bullish trend = CHoCH bearish (reversal)
            result["choch_bearish"] = True

    # Composite score
    score = 0.0
    if result["bos_bullish"]:
        score += 0.5   # Trend continuation = moderate signal
    if result["bos_bearish"]:
        score -= 0.5
    if result["choch_bullish"]:
        score += 1.0   # Reversal = strong signal
    if result["choch_bearish"]:
        score -= 1.0

    result["structure_score"] = float(np.clip(score, -1.0, 1.0))
    return result


# ── Composite SMC Score ──────────────────────────────────────────────

def calculate_smc_score(
    symbol: str, timeframe: int, shift: int = 0, lookback: int = 50
) -> float:
    """
    Composite Smart Money Concepts score combining:
    - Order Block proximity (40% weight)
    - Fair Value Gap proximity (30% weight)
    - Structure breaks/CHoCH (30% weight)

    Returns:
        Score in [-1, 1]:
        +1 = strong bullish institutional bias
        -1 = strong bearish institutional bias
         0 = neutral / no clear SMC signal
    """
    ob_score = calculate_ob_proximity(symbol, timeframe, shift, lookback)
    fvg_score = calculate_fvg_score(symbol, timeframe, shift, lookback)
    structure = detect_structure_breaks(symbol, timeframe, shift, lookback)
    struct_score = structure["structure_score"]

    composite = ob_score * 0.4 + fvg_score * 0.3 + struct_score * 0.3

    return float(np.clip(composite, -1.0, 1.0))
