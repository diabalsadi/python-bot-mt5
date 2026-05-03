"""
Strategy
--------
All core trading logic ported from gold-scalper-v3.mq5, organized into
standalone functions that are called from the main event loop.

Covers:
  - Lot sizing (CalculateLotSize)
  - ML-based entry levels (GetMLEntryLevels)
  - ML-based SL/TP distances (GetMLSLTP)
  - Trailing stop + trailing TP (TrailSLTP)
  - Stale-order cancellation (CancelStaleOrders)
  - Pending-order deduplication (HasOrderNearPrice)
  - Latency guard (CheckLatency)
  - Market-session detection (GetMarketSession)
  - US-open protection (CheckUSSessionExclusion)
  - Order execution (executeBuyLimit / executeSellLimit)
  - Utility: findHigh / findLow
"""

from __future__ import annotations

import time
import logging
from datetime import datetime

import MetaTrader5 as mt5

from indicators.volatility import calculate_volatility
from indicators.trend      import get_technical_trend
from ml.model              import LinearRegressionModel


# ══════════════════════════════════════════════════════════════════════
# Lot Sizing
# ══════════════════════════════════════════════════════════════════════

def calculate_lot_size(symbol: str, sl_points: int, risk_percent: float) -> float:
    """
    Risk-based position sizing.  Returns the lot size that risks exactly
    `risk_percent` of account balance over a stop loss of `sl_points`.

    MQL5 equivalent: CalculateLotSize()
    """
    balance    = mt5.account_info().balance
    risk_amt   = balance * risk_percent / 100.0

    sym        = mt5.symbol_info(symbol)
    tick_value = sym.trade_tick_value
    tick_size  = sym.trade_tick_size

    lot = risk_amt / (sl_points * (tick_value / tick_size))

    lot_step = sym.volume_step
    lot = (lot // lot_step) * lot_step              # floor to nearest step

    lot = max(sym.volume_min, min(sym.volume_max, lot))
    return round(lot, 2)


# ══════════════════════════════════════════════════════════════════════
# ML Entry Levels
# ══════════════════════════════════════════════════════════════════════

def get_ml_entry_levels(
    symbol: str,
    timeframe: int,
    trend_bars: int,
    feature_window: int,
    rsi_period: int,
    ema_period: int,
    model: LinearRegressionModel,
) -> tuple[float, float]:
    """
    Return (buy_level, sell_level) based on technical trend + ML confirmation.

    Buy  signal → entry at the lowest low of the last `trend_bars` bars.
    Sell signal → entry at the highest high of the last `trend_bars` bars.
    Returns -1 for the inactive side.

    MQL5 equivalent: GetMLEntryLevels()
    """
    trend      = get_technical_trend(symbol, timeframe, trend_bars, ema_period, rsi_period)
    prediction = model.predict(symbol, timeframe, feature_window, rsi_period)

    buy_level  = -1.0
    sell_level = -1.0

    count = trend_bars + 5
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, count)  # skip forming bar
    if rates is None or len(rates) == 0:
        return buy_level, sell_level

    highs  = rates["high"]
    lows   = rates["low"]

    if trend == 1 and prediction > 0:
        buy_level = float(lows.min())
        print(f"📈 TREND BUY  | Low[{trend_bars}]={buy_level:.5f}  pred={prediction:.5f}")

    elif trend == -1 and prediction < 0:
        sell_level = float(highs.max())
        print(f"📉 TREND SELL | High[{trend_bars}]={sell_level:.5f}  pred={prediction:.5f}")

    return buy_level, sell_level


# ══════════════════════════════════════════════════════════════════════
# ML SL / TP Distances
# ══════════════════════════════════════════════════════════════════════

def get_ml_sltp(
    symbol: str,
    timeframe: int,
    sl_points_default: int,
    feature_window: int,
    rsi_period: int,
    model: LinearRegressionModel,
) -> tuple[float, float]:
    """
    Return (sl_distance, tp_distance) in price units (not points).

    When the model is trained:
      SL = volatility × 1.3, clamped between 50 % and 180 % of the default.
      TP = SL × multiplier, where multiplier varies 2.0 – 3.0 based on
           predicted magnitude vs SL size.

    Falls back to static values when the model is not yet trained.

    MQL5 equivalent: GetMLSLTP()
    """
    sym   = mt5.symbol_info(symbol)
    point = sym.point

    if not model.is_trained:
        sl = sl_points_default * point
        tp = sl_points_default * 3 * point
        return sl, tp

    current_vol   = calculate_volatility(symbol, timeframe, 1, feature_window)
    prediction    = model.predict(symbol, timeframe, feature_window, rsi_period)
    pred_points   = abs(prediction) / point

    # SL based on volatility (in points)
    ml_sl_points = current_vol * 1.3
    min_sl = sl_points_default * 0.5
    max_sl = sl_points_default * 1.8
    ml_sl_points = max(min_sl, min(max_sl, ml_sl_points))

    # TP multiplier
    if pred_points > ml_sl_points * 2.5:
        tp_mult = 3.0
    elif pred_points < ml_sl_points * 0.8:
        tp_mult = 2.0
    else:
        tp_mult = 2.5

    ml_tp_points = ml_sl_points * tp_mult

    sl_distance = ml_sl_points * point
    tp_distance = ml_tp_points * point

    print(
        f"📊 ML SL/TP | vol={current_vol:.1f}  pred={pred_points:.1f}pt  "
        f"SL={ml_sl_points:.1f}pt  TP={ml_tp_points:.1f}pt  RR=1:{tp_mult:.1f}"
    )
    return sl_distance, tp_distance


# ══════════════════════════════════════════════════════════════════════
# Trailing Stop + Trailing TP
# ══════════════════════════════════════════════════════════════════════

def trail_sltp(
    symbol: str,
    timeframe: int,
    trailing_step_points: int,
    sl_points: int,
    feature_window: int,
) -> None:
    """
    Move SL (and TP) forward as the position gains profit.
    Step size is dynamic: proportional to current volatility but always at
    least `trailing_step_points`.

    MQL5 equivalent: TrailSLTP()
    """
    total = mt5.positions_total()
    if total == 0:
        return

    sym   = mt5.symbol_info(symbol)
    point = sym.point

    # Dynamic trailing step
    current_vol = calculate_volatility(symbol, timeframe, 1, feature_window)
    if (current_vol * point) > 2.0:
        step_points = trailing_step_points
    else:
        step_points = int(current_vol * 0.5)
        step_points = max(trailing_step_points, step_points)
        step_points = min(trailing_step_points * 4, step_points)

    stops_level = int(mt5.symbol_info_integer(symbol, mt5.SYMBOL_TRADE_STOPS_LEVEL))
    safety_buf  = 20 * point
    min_dist    = stops_level * point + safety_buf

    for i in range(total):
        pos = mt5.positions_get()[i]
        if pos is None or pos.symbol != symbol:
            continue

        entry = pos.price_open
        sl    = pos.sl
        tp    = pos.tp

        bid = mt5.symbol_info_tick(symbol).bid
        ask = mt5.symbol_info_tick(symbol).ask

        if pos.type == mt5.POSITION_TYPE_BUY:
            price         = bid
            profit_points = (price - entry) / point
        else:
            price         = ask
            profit_points = (entry - price) / point

        if profit_points < step_points:
            continue

        steps = int(profit_points // step_points)

        if pos.type == mt5.POSITION_TYPE_BUY:
            new_sl = entry + steps * step_points * point
            new_tp = new_sl + sl_points * 3 * point
        else:
            new_sl = entry - steps * step_points * point
            new_tp = new_sl - sl_points * 3 * point

        digits = sym.digits
        new_sl = round(new_sl, digits)
        new_tp = round(new_tp, digits)

        # Only move SL in profitable direction
        improve_sl = (
            (pos.type == mt5.POSITION_TYPE_BUY  and new_sl > sl) or
            (pos.type == mt5.POSITION_TYPE_SELL and new_sl < sl)
        )
        if not improve_sl:
            continue

        if abs(new_sl - sl) < 0.5 * point:
            continue

        if abs(new_sl - price) < min_dist or abs(new_tp - price) < min_dist:
            continue

        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "position": pos.ticket,
            "symbol":   symbol,
            "sl":       new_sl,
            "tp":       new_tp,
        }
        t0 = time.perf_counter()
        result = mt5.order_send(request)
        lat_ms = int((time.perf_counter() - t0) * 1000)
        logging.info(f"ACTION | TrailSLTP #{pos.ticket} | Latency: {lat_ms}ms")
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            err = mt5.last_error() if result is None else result.comment
            print(f"⚠️  TrailSLTP failed on #{pos.ticket}: {err}")


# ══════════════════════════════════════════════════════════════════════
# Stale-Order Cancellation
# ══════════════════════════════════════════════════════════════════════

def cancel_stale_orders(symbol: str, current_prediction: float) -> None:
    """
    Cancel pending orders whose direction contradicts the current ML prediction.

    Buy  orders cancelled when prediction is negative (bearish shift).
    Sell orders cancelled when prediction is positive (bullish shift).

    MQL5 equivalent: CancelStaleOrders()
    """
    orders = mt5.orders_get(symbol=symbol)
    if not orders:
        return

    for order in orders:
        order_type   = order.type
        is_buy_side  = order_type in (mt5.ORDER_TYPE_BUY_STOP,  mt5.ORDER_TYPE_BUY_LIMIT)
        is_sell_side = order_type in (mt5.ORDER_TYPE_SELL_STOP, mt5.ORDER_TYPE_SELL_LIMIT)

        cancel = (
            (is_buy_side  and current_prediction < 0) or
            (is_sell_side and current_prediction > 0)
        )

        if cancel:
            sym = mt5.symbol_info(symbol)
            pred_pts = current_prediction / sym.point if sym else current_prediction
            direction = "bearish" if current_prediction < 0 else "bullish"
            print(f"🚫 Cancel order #{order.ticket} — market shifted {direction} "
                  f"(pred={pred_pts:.2f} pts)")
            t0 = time.perf_counter()
            mt5.order_send({
                "action": mt5.TRADE_ACTION_REMOVE,
                "order":  order.ticket,
            })
            lat_ms = int((time.perf_counter() - t0) * 1000)
            logging.info(f"ACTION | CancelOrder #{order.ticket} | Latency: {lat_ms}ms")


def cancel_all_orders(symbol: str) -> None:
    """Cancel every pending order on `symbol`. MQL5: CancelAllOrders()"""
    orders = mt5.orders_get(symbol=symbol)
    if not orders:
        return
    for order in orders:
        t0 = time.perf_counter()
        result = mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": order.ticket})
        lat_ms = int((time.perf_counter() - t0) * 1000)
        logging.info(f"ACTION | CancelOrder #{order.ticket} | Latency: {lat_ms}ms")
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"🗑️  Cancelled order #{order.ticket}")
        else:
            err = mt5.last_error()
            print(f"❌ Could not cancel order #{order.ticket}: {err}")


def close_all_positions(symbol: str) -> None:
    """Close every open position on `symbol`. MQL5: CloseAllPositions()"""
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return
    tick = mt5.symbol_info_tick(symbol)
    for pos in positions:
        price  = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
        action = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        request = {
            "action":    mt5.TRADE_ACTION_DEAL,
            "position":  pos.ticket,
            "symbol":    symbol,
            "volume":    pos.volume,
            "type":      action,
            "price":     price,
            "deviation": 20,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        t0 = time.perf_counter()
        result = mt5.order_send(request)
        lat_ms = int((time.perf_counter() - t0) * 1000)
        logging.info(f"ACTION | ClosePosition #{pos.ticket} | Latency: {lat_ms}ms")
        
        ok = result and result.retcode == mt5.TRADE_RETCODE_DONE
        print(f"{'✅' if ok else '❌'} Close position #{pos.ticket}: "
              f"{'done' if ok else mt5.last_error()}")


# ══════════════════════════════════════════════════════════════════════
# Order Deduplication
# ══════════════════════════════════════════════════════════════════════

def has_order_near_price(
    symbol: str,
    target_price: float,
    order_type: int,
    min_distance_points: int = 50,
) -> bool:
    """
    Return True if a same-direction pending order already exists within
    `min_distance_points` of `target_price`.

    MQL5 equivalent: HasOrderNearPrice()
    """
    sym = mt5.symbol_info(symbol)
    if sym is None:
        return False

    min_dist = min_distance_points * sym.point

    buy_types  = (mt5.ORDER_TYPE_BUY_STOP,  mt5.ORDER_TYPE_BUY_LIMIT)
    sell_types = (mt5.ORDER_TYPE_SELL_STOP, mt5.ORDER_TYPE_SELL_LIMIT)

    is_buy_side  = order_type in buy_types
    is_sell_side = order_type in sell_types

    orders = mt5.orders_get(symbol=symbol)
    if not orders:
        return False

    for order in orders:
        existing_type  = order.type
        existing_price = order.price_open

        same_side = (
            (is_buy_side  and existing_type in buy_types) or
            (is_sell_side and existing_type in sell_types)
        )

        if same_side and abs(existing_price - target_price) < min_dist:
            dist_pts = abs(existing_price - target_price) / sym.point
            print(
                f"⚠️  Duplicate order near {target_price:.5f} "
                f"(existing {existing_price:.5f}, dist={dist_pts:.1f} pts)"
            )
            return True

    return False


# ══════════════════════════════════════════════════════════════════════
# Latency Guard
# ══════════════════════════════════════════════════════════════════════

_latency_state: dict = {"last_check": 0.0, "paused": False, "latency_ms": 0}


def check_latency(symbol: str, max_latency_ms: int = 2000) -> tuple[bool, int]:
    """
    Measure round-trip time for a symbol_info_tick() call and compare it
    against the threshold.  Only re-checks every 5 seconds.

    Returns:
        (trading_allowed, measured_latency_ms)

    MQL5 equivalent: CheckLatency()
    """
    now = time.time()
    state = _latency_state

    if now - state["last_check"] < 5.0:
        return not state["paused"], state["latency_ms"]

    state["last_check"] = now

    t0 = time.perf_counter()
    mt5.symbol_info_tick(symbol)           # representative MT5 API call
    latency_ms = int((time.perf_counter() - t0) * 1000)
    state["latency_ms"] = latency_ms

    if latency_ms > max_latency_ms:
        if not state["paused"]:
            print(f"⚠️  HIGH LATENCY: {latency_ms} ms (max {max_latency_ms} ms)")
            print("🛑 PAUSING TRADING — closing positions and cancelling orders …")
            close_all_positions(symbol)
            cancel_all_orders(symbol)
            state["paused"] = True
        return False, latency_ms
    else:
        if state["paused"]:
            print(f"✅ Latency normalised: {latency_ms} ms — RESUMING TRADING")
            state["paused"] = False
        return True, latency_ms


# ══════════════════════════════════════════════════════════════════════
# Market Session Detection
# ══════════════════════════════════════════════════════════════════════

def get_market_session(
    asia_start: int = 0,   asia_end: int = 9,
    europe_start: int = 10, europe_end: int = 18,
    us_start: int = 15,    us_end: int = 23,
) -> str:
    """
    Return a human-readable string of the active trading sessions.
    MQL5 equivalent: GetMarketSession()
    """
    hour = datetime.utcnow().hour
    parts = []
    if asia_start   <= hour < asia_end:   parts.append("Asia")
    if europe_start <= hour < europe_end: parts.append("Europe")
    if us_start     <= hour < us_end:     parts.append("USA")
    return " | ".join(parts) if parts else "Quiet"


_us_protection_state: dict = {"active": False}


def check_us_session_exclusion(
    us_start_hour: int = 15,
    protection_hours: int = 2,
) -> bool:
    """
    Return True (trading paused) during the first `protection_hours` of the
    US session opening.

    MQL5 equivalent: CheckUSSessionExclusion()
    """
    state = _us_protection_state
    hour  = datetime.utcnow().hour
    end   = us_start_hour + protection_hours

    if us_start_hour <= hour < end:
        if not state["active"]:
            print(f"🛑 US Open Protection active for {protection_hours} hours")
            state["active"] = True
        return True

    if state["active"]:
        print("✅ US Open Protection expired — RESUMING TRADING")
        state["active"] = False

    return False


# ══════════════════════════════════════════════════════════════════════
# Order Execution Helpers
# ══════════════════════════════════════════════════════════════════════

def _build_expiration(symbol: str, timeframe: int, expiration_hours: int) -> int:
    """Return Unix timestamp for order expiration."""
    current_bar_time = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1)
    if current_bar_time is None:
        return 0
    bar_seconds = mt5.period_seconds(timeframe)
    return int(current_bar_time[0]["time"]) + expiration_hours * bar_seconds


def execute_buy_limit(
    symbol: str,
    timeframe: int,
    entry_level: float,
    sl_points_default: int,
    feature_window: int,
    rsi_period: int,
    risk_percent: float,
    expiration_hours: int,
    min_order_distance_points: int,
    model: LinearRegressionModel,
) -> None:
    """
    Place a Buy Limit order at or below the current ask.
    MQL5 equivalent: executeBuyLimit()
    """
    sym     = mt5.symbol_info(symbol)
    point   = sym.point
    digits  = sym.digits
    stops   = int(mt5.symbol_info_integer(symbol, mt5.SYMBOL_TRADE_STOPS_LEVEL))

    ask   = mt5.symbol_info_tick(symbol).ask
    entry = round(entry_level, digits)

    min_dist = stops * point
    if ask - entry < min_dist:
        entry = round(ask - min_dist - 2 * point, digits)

    if has_order_near_price(symbol, entry, mt5.ORDER_TYPE_BUY_LIMIT, min_order_distance_points):
        return

    sl_dist, tp_dist = get_ml_sltp(symbol, timeframe, sl_points_default,
                                    feature_window, rsi_period, model)
    sl = round(entry - sl_dist, digits)
    tp = round(entry + tp_dist, digits)

    sl_pts  = int(sl_dist / point)
    lot     = calculate_lot_size(symbol, sl_pts, risk_percent)
    expiry  = _build_expiration(symbol, timeframe, expiration_hours)

    print(f"🟦 BUY LIMIT  → entry={entry:.5f}  SL={sl:.5f}  TP={tp:.5f}  lot={lot}")

    request = {
        "action":       mt5.TRADE_ACTION_PENDING,
        "symbol":       symbol,
        "volume":       lot,
        "type":         mt5.ORDER_TYPE_BUY_LIMIT,
        "price":        entry,
        "sl":           sl,
        "tp":           tp,
        "type_time":    mt5.ORDER_TIME_SPECIFIED,
        "expiration":   expiry,
        "comment":      "ML buy limit",
    }
    t0 = time.perf_counter()
    result = mt5.order_send(request)
    lat_ms = int((time.perf_counter() - t0) * 1000)
    logging.info(f"ACTION | ExecuteBuyLimit {symbol} | Latency: {lat_ms}ms")
    
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        err = mt5.last_error() if result is None else result.comment
        print(f"❌ BuyLimit failed: {err}")


def execute_sell_limit(
    symbol: str,
    timeframe: int,
    entry_level: float,
    sl_points_default: int,
    feature_window: int,
    rsi_period: int,
    risk_percent: float,
    expiration_hours: int,
    min_order_distance_points: int,
    model: LinearRegressionModel,
) -> None:
    """
    Place a Sell Limit order at or above the current bid.
    MQL5 equivalent: executeSellLimit()
    """
    sym     = mt5.symbol_info(symbol)
    point   = sym.point
    digits  = sym.digits
    stops   = int(mt5.symbol_info_integer(symbol, mt5.SYMBOL_TRADE_STOPS_LEVEL))

    bid   = mt5.symbol_info_tick(symbol).bid
    entry = round(entry_level, digits)

    min_dist = stops * point
    if entry - bid < min_dist:
        entry = round(bid + min_dist + 2 * point, digits)

    if has_order_near_price(symbol, entry, mt5.ORDER_TYPE_SELL_LIMIT, min_order_distance_points):
        return

    sl_dist, tp_dist = get_ml_sltp(symbol, timeframe, sl_points_default,
                                    feature_window, rsi_period, model)
    sl = round(entry + sl_dist, digits)
    tp = round(entry - tp_dist, digits)

    sl_pts = int(sl_dist / point)
    lot    = calculate_lot_size(symbol, sl_pts, risk_percent)
    expiry = _build_expiration(symbol, timeframe, expiration_hours)

    print(f"🟥 SELL LIMIT → entry={entry:.5f}  SL={sl:.5f}  TP={tp:.5f}  lot={lot}")

    request = {
        "action":       mt5.TRADE_ACTION_PENDING,
        "symbol":       symbol,
        "volume":       lot,
        "type":         mt5.ORDER_TYPE_SELL_LIMIT,
        "price":        entry,
        "sl":           sl,
        "tp":           tp,
        "type_time":    mt5.ORDER_TIME_SPECIFIED,
        "expiration":   expiry,
        "comment":      "ML sell limit",
    }
    t0 = time.perf_counter()
    result = mt5.order_send(request)
    lat_ms = int((time.perf_counter() - t0) * 1000)
    logging.info(f"ACTION | ExecuteSellLimit {symbol} | Latency: {lat_ms}ms")
    
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        err = mt5.last_error() if result is None else result.comment
        print(f"❌ SellLimit failed: {err}")


# ══════════════════════════════════════════════════════════════════════
# Utility: Find Highest High / Lowest Low
# ══════════════════════════════════════════════════════════════════════

def find_high(symbol: str, timeframe: int, bars_n: int) -> float:
    """Return the highest high of the last `bars_n` bars. MQL5: findHigh()"""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, bars_n)
    return float(rates["high"].max()) if rates is not None and len(rates) > 0 else -1.0


def find_low(symbol: str, timeframe: int, bars_n: int) -> float:
    """Return the lowest low of the last `bars_n` bars. MQL5: findLow()"""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, bars_n)
    return float(rates["low"].min()) if rates is not None and len(rates) > 0 else -1.0
