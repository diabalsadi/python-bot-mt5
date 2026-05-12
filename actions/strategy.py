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
  - Multi-timeframe quick profit (15m lookahead + small-timeframe S/R)
  - Utility: findHigh / findLow
"""

from __future__ import annotations

import time
import logging
from datetime import datetime

import MetaTrader5 as mt5

from indicators.volatility import calculate_volatility
from indicators.trend import get_technical_trend
from ml.model import LinearRegressionModel
from tools.trade_logger import TradeLogger

_trade_logger = TradeLogger()   # on_close callback set by main.py via set_trade_logger_callback()


def set_trade_logger_callback(cb) -> None:
    """Called from main.py to register the consecutive-loss circuit breaker."""
    _trade_logger._on_close = cb

# Module-level state (replaces function-attribute hacks)
_trail_debug_cnt: dict[int, int] = {}
_trail_last_err: str = ""
_scale_in_last_log_ts: float = 0.0


# ══════════════════════════════════════════════════════════════════════
# Lot Sizing
# ══════════════════════════════════════════════════════════════════════


def calculate_lot_size(symbol: str, sl_points: int, risk_percent: float) -> float:
    """
    Risk-based position sizing.  Returns the lot size that risks exactly
    `risk_percent` of account balance over a stop loss of `sl_points`.

    MQL5 equivalent: CalculateLotSize()
    """
    account = mt5.account_info()
    if account is None:
        print("❌ calculate_lot_size: account_info unavailable — returning min lot")
        sym = mt5.symbol_info(symbol)
        return sym.volume_min if sym else 0.01

    balance = account.balance
    risk_amt = balance * risk_percent / 100.0

    sym = mt5.symbol_info(symbol)
    if sym is None:
        print("❌ calculate_lot_size: symbol_info unavailable — returning 0.01")
        return 0.01
    tick_value = sym.trade_tick_value
    tick_size = sym.trade_tick_size

    lot = risk_amt / (sl_points * (tick_value / tick_size))

    lot_step = sym.volume_step
    lot = (lot // lot_step) * lot_step  # floor to nearest step

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
    prediction: float,
    verbose: bool = True,
) -> tuple[float, float]:
    """
    Return (buy_level, sell_level) based on technical trend + ML confirmation.

    Buy  signal → entry at the lowest low of the last `trend_bars` bars.
    Sell signal → entry at the highest high of the last `trend_bars` bars.
    Returns -1 for the inactive side.

    MQL5 equivalent: GetMLEntryLevels()
    """
    trend = get_technical_trend(symbol, timeframe, trend_bars, ema_period, rsi_period)
    # prediction is now passed as an argument to ensure synchronization

    buy_level = -1.0
    sell_level = -1.0

    count = trend_bars + 5
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)  # include forming bar
    if rates is None or len(rates) == 0:
        return buy_level, sell_level

    highs = rates["high"]
    lows = rates["low"]

    # ── VOLATILITY FILTER ──────────────────────────────────────────
    from indicators.volatility import calculate_volatility
    current_vol = calculate_volatility(symbol, timeframe, 1, feature_window)
    min_strength = current_vol * 0.4 # Need prediction > 40% of avg range
    
    # ── FIBONACCI PULLBACK FILTER ──────────────────────────────────
    swing_high = float(highs.max())
    swing_low = float(lows.min())
    swing_range = swing_high - swing_low
    
    # Buy pullback target: price should be in the lower half of the swing (below 50%)
    fib_50 = swing_low + swing_range * 0.50
    fib_61 = swing_low + swing_range * 0.382 # 61.8% retracement from high
    
    # Current price
    tick = mt5.symbol_info_tick(symbol)
    curr_price = tick.bid if trend == 1 else tick.ask

    if trend == 1 and prediction > min_strength:
        # Only buy if price is at or below 50% retracement (Pullback)
        if curr_price <= fib_50:
            buy_level = swing_low # Enter at recent low
            if verbose:
                print(f"📈 TREND BUY  | Fib Pullback OK (Price {curr_price:.5f} <= 50% Fib {fib_50:.5f})")
        else:
            if verbose: print(f"⏳ BUY Skipped | Price {curr_price:.5f} too high (above 50% Fib {fib_50:.5f})")

    elif trend == -1 and prediction < -min_strength:
        # Only sell if price is at or above 50% retracement
        fib_50_sell = swing_high - swing_range * 0.50
        if curr_price >= fib_50_sell:
            sell_level = swing_high
            if verbose:
                print(f"📉 TREND SELL | Fib Pullback OK (Price {curr_price:.5f} >= 50% Fib {fib_50_sell:.5f})")
        else:
            if verbose: print(f"⏳ SELL Skipped | Price {curr_price:.5f} too low (below 50% Fib {fib_50_sell:.5f})")

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
    # ── BROKER LIMITS ──────────────────────────────────────────────
    sym = mt5.symbol_info(symbol)
    if sym is None:
        # Fallback if symbol info is unavailable
        return sl_points_default * 0.0001, sl_points_default * 3 * 0.0001
        
    point = sym.point
    stops_level = int(sym.trade_stops_level)
    freeze_level = int(sym.trade_freeze_level)
    min_dist_pts = max(stops_level, freeze_level) + 20 
    digits = sym.digits

    if not model.is_trained:
        sl = max(min_dist_pts, sl_points_default) * point
        tp = max(min_dist_pts, sl_points_default * 3) * point
        return round(sl, digits), round(tp, digits)

    current_vol = calculate_volatility(symbol, timeframe, 1, feature_window)
    prediction = model.predict(symbol, timeframe, feature_window, rsi_period)
    pred_points = abs(prediction) / point

    # SL based on volatility (in points)
    ml_sl_points = current_vol * 1.3
    # Don't go below broker minimum
    min_sl = max(min_dist_pts, sl_points_default * 0.5)
    max_sl = sl_points_default * 1.8
    ml_sl_points = max(min_sl, min(max_sl, ml_sl_points))

    # TP multiplier
    tp_mult = 3.0 if pred_points > ml_sl_points * 2.5 else 2.0 if pred_points < ml_sl_points * 0.8 else 2.5
    ml_tp_points = max(min_dist_pts, ml_sl_points * tp_mult)

    sl_distance = round(ml_sl_points * point, digits)
    tp_distance = round(ml_tp_points * point, digits)

    print(
        f"📊 ML SL/TP | vol={current_vol:.1f}  pred={pred_points:.1f}pt  "
        f"SL={ml_sl_points:.1f}pt  TP={ml_tp_points:.1f}pt  RR=1:{ml_tp_points/ml_sl_points:.1f}"
    )
    return sl_distance, tp_distance


# ══════════════════════════════════════════════════════════════════════
# Position Management (Track all trades including manual ones)
# ══════════════════════════════════════════════════════════════════════


def get_symbol_positions(symbol: str) -> list:
    """
    Get all open positions for a specific symbol, including manual trades.

    Safely handles account with mixed EA and manual trades by filtering
    by symbol instead of assuming fixed indices.

    Args:
        symbol: Trading symbol

    Returns:
        List of position objects for the symbol (may be empty)
    """
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
        return []

    return list(positions)


# ══════════════════════════════════════════════════════════════════════
# Trailing Stop + Trailing TP
# ══════════════════════════════════════════════════════════════════════


def trail_sltp(
    symbol: str,
    timeframe: int,
    trailing_step_points: int,
    sl_points: int,
    feature_window: int,
    rsi_period: int,
    model: LinearRegressionModel,
) -> None:
    """
    Move SL (and TP) forward as the position gains profit.
    """
    positions = get_symbol_positions(symbol)
    if not positions:
        return

    sym = mt5.symbol_info(symbol)
    if sym is None:
        return

    point = sym.point
    digits = sym.digits
    
    # ── VOLATILITY & ML CONTEXT ────────────────────────────────────
    current_vol = calculate_volatility(symbol, timeframe, 1, feature_window)
    vol_dist = current_vol * point # 1.0 ATR in price units
    
    # Get current prediction to adjust TP
    prediction = model.predict(symbol, timeframe, feature_window, rsi_period)
    pred_mag = abs(prediction)

    # ── MIN DISTANCE (Broker's Limit) ──────────────────────────────
    stops_level = int(sym.trade_stops_level)
    freeze_level = int(sym.trade_freeze_level)
    min_dist = (max(stops_level, freeze_level) + 20) * point 

    # Iterate through actual positions
    for pos in positions:
        if pos is None or pos.symbol != symbol:
            continue

        entry = pos.price_open
        sl = pos.sl
        tp = pos.tp

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            continue

        price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
        profit_points = (price - entry) / point if pos.type == mt5.POSITION_TYPE_BUY else (entry - price) / point
        
        # ── 3-STAGE TRAILING LOGIC ──────────────────────────────────
        # Stage thresholds based on ATR (current_vol)
        stage = "Normal"
        
        if profit_points > current_vol * 3.0:
            # Stage 3: Aggressive Harvest (Lock in 70% of profit)
            # Use tight 0.8x ATR distance
            effective_dist = vol_dist * 0.8
            stage = "Stage 3 (Harvest)"
        elif profit_points > current_vol * 1.2:
            # Stage 2: Trend Following (Give room to breathe)
            # Use wider 1.5x ATR distance
            effective_dist = vol_dist * 1.5
            stage = "Stage 2 (Trend)"
        elif profit_points > current_vol * 0.7:
            # Stage 1: 50% Profit Reserve (Lock in half of the gains)
            # Instead of just breaking even, we take half the profit off the table
            reserve_pts = profit_points * 0.5
            if pos.type == mt5.POSITION_TYPE_BUY:
                target_sl = round(entry + reserve_pts * point, digits)
            else:
                target_sl = round(entry - reserve_pts * point, digits)
            
            effective_dist = abs(price - target_sl)
            stage = "Stage 1 (50% Reserve)"
        else:
            # Default: initial SL distance
            effective_dist = sl_points * point
            stage = "Initial"

        # Ensure effective_dist is valid
        effective_dist = max(effective_dist, min_dist)
        
        # ── CALCULATE TARGETS ────────────────────────────────────────
        # Stage 1 already computed target_sl directly; all other stages derive it from price
        if stage != "Stage 1 (50% Reserve)":
            if pos.type == mt5.POSITION_TYPE_BUY:
                target_sl = round(price - effective_dist, digits)
            else:
                target_sl = round(price + effective_dist, digits)

        # Dynamic TP: If prediction weakens, pull TP closer to lock in profit
        # If prediction mag is < 50% of the average range, use a tighter TP
        tp_mult = 3.0 if pred_mag > current_vol * 0.5 else 1.5
        tp_dist = effective_dist * tp_mult

        if pos.type == mt5.POSITION_TYPE_BUY:
            target_tp = round(price + tp_dist, digits)
            # Ensure TP only moves forward or stays
            if tp > 0 and target_tp < tp: target_tp = tp 
            is_better_sl = (target_sl > sl) or (sl == 0)
        else:
            target_tp = round(price - tp_dist, digits)
            if tp > 0 and target_tp > tp: target_tp = tp
            is_better_sl = (target_sl < sl) or (sl == 0)

        # ── SAFETY: BROKER LIMITS ───────────────────────────────────
        if pos.type == mt5.POSITION_TYPE_BUY:
            if target_sl > price - min_dist: target_sl = round(price - min_dist, digits)
            if target_tp < price + min_dist: target_tp = round(price + min_dist, digits)
        else:
            if target_sl < price + min_dist: target_sl = round(price + min_dist, digits)
            if target_tp > price - min_dist: target_tp = round(price - min_dist, digits)

        # DEBUG: Throttled feedback
        cnt = _trail_debug_cnt.get(pos.ticket, 0)
        if profit_points > 10 and cnt % 100 == 0:
            print(f"🔄 {stage} #{pos.ticket} | Prof: {profit_points:.1f}pt | "
                  f"SL: {sl:.5f}->{target_sl:.5f} | Better: {is_better_sl}")
        _trail_debug_cnt[pos.ticket] = cnt + 1

        # ── EXECUTE ─────────────────────────────────────────────────
        if profit_points > 0 and is_better_sl:
            step_pts = abs(target_sl - sl) / point if sl > 0 else 9999
            
            if step_pts >= trailing_step_points:
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": pos.ticket,
                    "symbol": symbol,
                    "sl": target_sl,
                    "tp": target_tp,
                }
                
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logging.info(f"ACTION | TrailSLTP {stage} #{pos.ticket} | SL: {target_sl:.5f}")
                    print(f"✅ {stage} Updated #{pos.ticket} | SL: {target_sl:.5f}")
                else:
                    global _trail_last_err
                    err = mt5.last_error() if result is None else result.comment
                    if _trail_last_err != err:
                        logging.error(f"FAIL | TrailSLTP #{pos.ticket} | Error: {err}")
                        print(f"❌ TrailSLTP Fail: {err}")
                        _trail_last_err = err






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
        order_type = order.type
        is_buy_side = order_type in (mt5.ORDER_TYPE_BUY_STOP, mt5.ORDER_TYPE_BUY_LIMIT)
        is_sell_side = order_type in (
            mt5.ORDER_TYPE_SELL_STOP,
            mt5.ORDER_TYPE_SELL_LIMIT,
        )

        cancel = (is_buy_side and current_prediction < 0) or (
            is_sell_side and current_prediction > 0
        )

        if cancel:
            sym = mt5.symbol_info(symbol)
            pred_pts = current_prediction / sym.point if sym else current_prediction
            direction = "bearish" if current_prediction < 0 else "bullish"
            print(
                f"🚫 Cancel order #{order.ticket} — market shifted {direction} "
                f"(pred={pred_pts:.2f} pts)"
            )
            t0 = time.perf_counter()
            mt5.order_send(
                {
                    "action": mt5.TRADE_ACTION_REMOVE,
                    "order": order.ticket,
                }
            )
            lat_ms = int((time.perf_counter() - t0) * 1000)
            logging.info(f"ACTION | CancelOrder #{order.ticket} | Latency: {lat_ms}ms")


def cancel_all_orders(symbol: str) -> None:
    """Cancel every pending order on `symbol`. MQL5: CancelAllOrders()"""
    orders = mt5.orders_get(symbol=symbol)
    if not orders:
        return
    for order in orders:
        t0 = time.perf_counter()
        result = mt5.order_send(
            {"action": mt5.TRADE_ACTION_REMOVE, "order": order.ticket}
        )
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
        price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
        action = (
            mt5.ORDER_TYPE_SELL
            if pos.type == mt5.POSITION_TYPE_BUY
            else mt5.ORDER_TYPE_BUY
        )
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": pos.ticket,
            "symbol": symbol,
            "volume": pos.volume,
            "type": action,
            "price": price,
            "deviation": 20,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        t0 = time.perf_counter()
        result = mt5.order_send(request)
        lat_ms = int((time.perf_counter() - t0) * 1000)
        logging.info(f"ACTION | ClosePosition #{pos.ticket} | Latency: {lat_ms}ms")

        ok = result and result.retcode == mt5.TRADE_RETCODE_DONE
        print(
            f"{'✅' if ok else '❌'} Close position #{pos.ticket}: "
            f"{'done' if ok else mt5.last_error()}"
        )


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

    buy_types = (mt5.ORDER_TYPE_BUY_STOP, mt5.ORDER_TYPE_BUY_LIMIT)
    sell_types = (mt5.ORDER_TYPE_SELL_STOP, mt5.ORDER_TYPE_SELL_LIMIT)

    is_buy_side = order_type in buy_types
    is_sell_side = order_type in sell_types

    orders = mt5.orders_get(symbol=symbol)
    if not orders:
        return False

    for order in orders:
        existing_type = order.type
        existing_price = order.price_open

        same_side = (is_buy_side and existing_type in buy_types) or (
            is_sell_side and existing_type in sell_types
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
    mt5.symbol_info_tick(symbol)  # representative MT5 API call
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
    asia_start: int = 0,
    asia_end: int = 9,
    europe_start: int = 10,
    europe_end: int = 18,
    us_start: int = 15,
    us_end: int = 23,
) -> str:
    """
    Return a human-readable string of the active trading sessions.
    MQL5 equivalent: GetMarketSession()
    """
    hour = datetime.utcnow().hour
    parts = []
    if asia_start <= hour < asia_end:
        parts.append("Asia")
    if europe_start <= hour < europe_end:
        parts.append("Europe")
    if us_start <= hour < us_end:
        parts.append("USA")
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
    hour = datetime.utcnow().hour
    end = us_start_hour + protection_hours

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


def _get_timeframe_seconds(timeframe: int) -> int:
    mapping = {
        mt5.TIMEFRAME_M1: 60,
        mt5.TIMEFRAME_M2: 120,
        mt5.TIMEFRAME_M3: 180,
        mt5.TIMEFRAME_M4: 240,
        mt5.TIMEFRAME_M5: 300,
        mt5.TIMEFRAME_M6: 360,
        mt5.TIMEFRAME_M10: 600,
        mt5.TIMEFRAME_M12: 720,
        mt5.TIMEFRAME_M15: 900,
        mt5.TIMEFRAME_M20: 1200,
        mt5.TIMEFRAME_M30: 1800,
        mt5.TIMEFRAME_H1: 3600,
        mt5.TIMEFRAME_H2: 7200,
        mt5.TIMEFRAME_H3: 10800,
        mt5.TIMEFRAME_H4: 14400,
        mt5.TIMEFRAME_H6: 21600,
        mt5.TIMEFRAME_H8: 28800,
        mt5.TIMEFRAME_H12: 43200,
        mt5.TIMEFRAME_D1: 86400,
        mt5.TIMEFRAME_W1: 604800,
        mt5.TIMEFRAME_MN1: 2592000,
    }
    return mapping.get(timeframe, 60)


def _build_expiration(symbol: str, timeframe: int, expiration_hours: int) -> int:
    """Return Unix timestamp for order expiration."""
    current_bar_time = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1)
    if current_bar_time is None:
        return 0
    bar_seconds = _get_timeframe_seconds(timeframe)
    return int(current_bar_time[0]["time"]) + expiration_hours * bar_seconds


def has_open_position(symbol: str, position_type: int) -> bool:
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return False
    for pos in positions:
        if pos.type == position_type:
            return True
    return False


def execute_buy_market(
    symbol: str,
    timeframe: int,
    sl_points_default: int,
    feature_window: int,
    rsi_period: int,
    risk_percent: float,
    model: LinearRegressionModel,
    allow_multiple: bool = False,
) -> None:
    """
    Execute a Market Buy order at current ask.
    """
    sym = mt5.symbol_info(symbol)
    point = sym.point
    digits = sym.digits

    if not allow_multiple and has_open_position(symbol, mt5.POSITION_TYPE_BUY):
        return

    # Anti-hedge guard: never open a BUY while a SELL is still open
    if has_open_position(symbol, mt5.POSITION_TYPE_SELL):
        print(f"🚫 Anti-hedge: BUY blocked — SELL position already open on {symbol}")
        return

    # Portfolio-level guard
    from position_manager import get_position_manager
    ok, reason = get_position_manager().can_open_position(symbol)
    if not ok:
        print(f"🚫 PositionManager: BUY blocked — {reason}")
        return

    ask = mt5.symbol_info_tick(symbol).ask
    entry = ask

    sl_dist, tp_dist = get_ml_sltp(
        symbol, timeframe, sl_points_default, feature_window, rsi_period, model
    )
    sl = round(entry - sl_dist, digits)
    tp = round(entry + tp_dist, digits)

    sl_pts = int(sl_dist / point)
    lot = calculate_lot_size(symbol, sl_pts, risk_percent)

    print(f"🟦 BUY MARKET → entry={entry:.5f}  SL={sl:.5f}  TP={tp:.5f}  lot={lot}")

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY,
        "price": entry,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
        "comment": "ML buy market",
    }
    t0 = time.perf_counter()
    result = mt5.order_send(request)
    lat_ms = int((time.perf_counter() - t0) * 1000)
    logging.info(f"ACTION | ExecuteBuyMarket {symbol} | Latency: {lat_ms}ms")

    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        err = mt5.last_error() if result is None else result.comment
        print(f"❌ Buy Market failed: {err}")
    else:
        _trade_logger.log_open(result, symbol, "BUY", entry, sl, tp, lot, "ML buy market")


def execute_sell_market(
    symbol: str,
    timeframe: int,
    sl_points_default: int,
    feature_window: int,
    rsi_period: int,
    risk_percent: float,
    model: LinearRegressionModel,
    allow_multiple: bool = False,  # ✅ FIXED
) -> None:
    """
    Execute a Market Sell order at current bid.
    """
    sym = mt5.symbol_info(symbol)
    point = sym.point
    digits = sym.digits

    if not allow_multiple and has_open_position(symbol, mt5.POSITION_TYPE_SELL):
        return

    # Anti-hedge guard: never open a SELL while a BUY is still open
    if has_open_position(symbol, mt5.POSITION_TYPE_BUY):
        print(f"🚫 Anti-hedge: SELL blocked — BUY position already open on {symbol}")
        return

    # Portfolio-level guard
    from position_manager import get_position_manager
    ok, reason = get_position_manager().can_open_position(symbol)
    if not ok:
        print(f"🚫 PositionManager: SELL blocked — {reason}")
        return

    bid = mt5.symbol_info_tick(symbol).bid
    entry = bid

    sl_dist, tp_dist = get_ml_sltp(
        symbol, timeframe, sl_points_default, feature_window, rsi_period, model
    )

    sl = round(entry + sl_dist, digits)
    tp = round(entry - tp_dist, digits)

    sl_pts = int(sl_dist / point)
    lot = calculate_lot_size(symbol, sl_pts, risk_percent)

    print(f"🟥 SELL MARKET → entry={entry:.5f}  SL={sl:.5f}  TP={tp:.5f}  lot={lot}")

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_SELL,
        "price": entry,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
        "comment": "ML sell market",
    }

    t0 = time.perf_counter()
    result = mt5.order_send(request)
    lat_ms = int((time.perf_counter() - t0) * 1000)
    logging.info(f"ACTION | ExecuteSellMarket {symbol} | Latency: {lat_ms}ms")

    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        err = mt5.last_error() if result is None else result.comment
        print(f"❌ Sell Market failed: {err}")
    else:
        _trade_logger.log_open(result, symbol, "SELL", entry, sl, tp, lot, "ML sell market")


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


# ══════════════════════════════════════════════════════════════════════
# Multi-Timeframe Quick Profit Targets (15m + S/R based)
# ══════════════════════════════════════════════════════════════════════


def get_quick_profit_levels(
    symbol: str,
    timeframe: int,
    trend_direction: int,  # 1 = buy, -1 = sell
    model: LinearRegressionModel,
    feature_window: int,
    rsi_period: int,
) -> tuple[float, float]:
    """
    Calculate quick profit levels based on:
      1. 15-minute lookahead ML prediction
      2. 15-minute support/resistance zones
      3. Current small-timeframe swing highs/lows

    Returns (tight_tp, extended_tp) for scalping two-level profit targets.

    Args:
        symbol:           Trading symbol
        timeframe:        Primary timeframe (e.g., M1)
        trend_direction:  1 for buy, -1 for sell
        model:            Trained ML model
        feature_window:   Feature window size
        rsi_period:       RSI period

    Returns:
        (tight_tp, extended_tp) — two profit target levels in price units
    """
    from indicators.support_resistance import identify_sr_levels

    sym_info = mt5.symbol_info(symbol)
    if sym_info is None:
        return -1.0, -1.0

    # Get current price
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return -1.0, -1.0

    current_price = tick.bid if trend_direction == 1 else tick.ask

    # 1. Get 15-minute lookahead prediction
    lookahead_pred = model.predict_ahead(
        symbol, timeframe, feature_window, rsi_period, horizon=15
    )

    # 2. Get S/R levels from 15m timeframe
    support_levels, resistance_levels = identify_sr_levels(
        symbol, mt5.TIMEFRAME_M15, lookback_bars=50
    )

    # 3. Find nearest S/R in direction of trade
    tight_tp = -1.0
    extended_tp = -1.0

    if trend_direction == 1:  # BUY
        # Look for resistances above current price
        resistances_above = [r for r in resistance_levels if r > current_price]
        if resistances_above:
            # Tight TP = nearest resistance (quick scalp)
            tight_tp = min(resistances_above)
            # Extended TP = 2nd nearest or lookahead target
            if len(resistances_above) > 1:
                extended_tp = resistances_above[1]
            else:
                # Use lookahead if positive
                extended_tp = (
                    current_price + lookahead_pred
                    if lookahead_pred > 0
                    else tight_tp * 1.002
                )

    else:  # SELL (trend_direction == -1)
        # Look for supports below current price
        supports_below = [s for s in support_levels if s < current_price]
        if supports_below:
            # Tight TP = nearest support (quick scalp)
            tight_tp = max(supports_below)
            # Extended TP = 2nd nearest or lookahead target
            if len(supports_below) > 1:
                extended_tp = supports_below[-2]
            else:
                # Use lookahead if negative
                extended_tp = (
                    current_price + lookahead_pred
                    if lookahead_pred < 0
                    else tight_tp * 0.998
                )

    return tight_tp, extended_tp


def get_15m_lookahead_confirmation(
    symbol: str,
    timeframe: int,
    model: LinearRegressionModel,
    feature_window: int,
    rsi_period: int,
) -> tuple[float, str]:
    """
    Get 15-minute lookahead price target and direction confirmation.

    Returns (target_price, direction) where direction is "BUY", "SELL", or "NEUTRAL".
    Useful for multi-timeframe confirmation before entering trades.

    Args:
        symbol:         Trading symbol
        timeframe:      Primary timeframe
        model:          Trained ML model
        feature_window: Feature window size
        rsi_period:     RSI period

    Returns:
        (target_price, direction_string)
    """
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        # Fallback to last known rates if tick fails
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1)
        if rates is not None and len(rates) > 0:
            current_price = rates[0]["close"]
        else:
            return 0.0, "NEUTRAL"
    else:
        current_price = tick.bid
    prediction_15m = model.predict_ahead(
        symbol, timeframe, feature_window, rsi_period, horizon=15
    )

    target_price = current_price + prediction_15m

    if prediction_15m > 0:
        direction = "BUY"
    elif prediction_15m < 0:
        direction = "SELL"
    else:
        direction = "NEUTRAL"

    return target_price, direction


# ══════════════════════════════════════════════════════════════════════
# S/R-Based Entry Strategy
# ══════════════════════════════════════════════════════════════════════


def execute_sr_entries(
    symbol: str,
    timeframe: int,
    sl_points_default: int,
    feature_window: int,
    rsi_period: int,
    risk_percent: float,
    model,
    confirmed_prediction: float,
    proximity_points: int = 150,
    sr_lookback_bars: int = 50,
) -> None:
    from indicators.support_resistance import identify_sr_levels

    sym = mt5.symbol_info(symbol)
    if sym is None:
        return

    point = sym.point

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return

    bid = tick.bid
    ask = tick.ask

    # last closed candle
    last_bar = mt5.copy_rates_from_pos(symbol, timeframe, 1, 2)
    if last_bar is None or len(last_bar) < 2:
        return

    last_close = float(last_bar["close"][0])
    prev_close = float(last_bar["close"][1])

    # 🔥 MOMENTUM (simple but powerful)
    price_velocity = last_close - prev_close
    is_rising = price_velocity > 0
    is_falling = price_velocity < 0

    support_levels, resistance_levels = identify_sr_levels(
        symbol, timeframe, sr_lookback_bars
    )

    proximity = proximity_points * point

    # ── RESISTANCE ─────────────────────────────
    for res in resistance_levels:

        # SELL (mean reversion) → price rising into resistance
        if (
            is_rising
            and ask <= res
            and (res - ask) <= proximity
            and confirmed_prediction < 0
        ):
            if not has_open_position(symbol, mt5.POSITION_TYPE_SELL):
                print(f"📉 SELL @ resistance (momentum confirmed)")
                execute_sell_market(
                    symbol,
                    timeframe,
                    sl_points_default,
                    feature_window,
                    rsi_period,
                    risk_percent,
                    model,
                )
            break

        # BUY breakout
        if last_close > res and confirmed_prediction > 0:
            if not has_open_position(symbol, mt5.POSITION_TYPE_BUY):
                print(f"🚀 BUY breakout resistance")
                execute_buy_market(
                    symbol,
                    timeframe,
                    sl_points_default,
                    feature_window,
                    rsi_period,
                    risk_percent,
                    model,
                )
            break

    # ── SUPPORT ───────────────────────────────
    for sup in support_levels:

        # BUY (mean reversion) → price falling into support
        if (
            is_falling
            and bid >= sup
            and (bid - sup) <= proximity
            and confirmed_prediction > 0
        ):
            if not has_open_position(symbol, mt5.POSITION_TYPE_BUY):
                print(f"📈 BUY @ support (momentum confirmed)")
                execute_buy_market(
                    symbol,
                    timeframe,
                    sl_points_default,
                    feature_window,
                    rsi_period,
                    risk_percent,
                    model,
                )
            break

        # SELL breakout
        if last_close < sup and confirmed_prediction < 0:
            if not has_open_position(symbol, mt5.POSITION_TYPE_SELL):
                print(f"💥 SELL breakout support")
                execute_sell_market(
                    symbol,
                    timeframe,
                    sl_points_default,
                    feature_window,
                    rsi_period,
                    risk_percent,
                    model,
                )
            break


# ══════════════════════════════════════════════════════════════════════
# Scale-In Cooldown State
# ══════════════════════════════════════════════════════════════════════
import time as _time

_scale_in_last_time: dict = {}  # symbol -> last scale-in unix timestamp


# ══════════════════════════════════════════════════════════════════════
# Scale-In Management (averaging down with risk-based position cap)
# ══════════════════════════════════════════════════════════════════════


def manage_scale_in(
    symbol: str,
    timeframe: int,
    sl_points_default: int,
    feature_window: int,
    rsi_period: int,
    risk_percent: float,
    model,
    max_total_risk_percent: float,
    volatility_multiplier: float,
    cooldown_seconds: float = 60.0,
) -> None:
    """
    Averaging-down (scale-in) strategy.

    If a position is losing beyond a volatility-based threshold AND the ML
    prediction still favours the same direction, we:
      1. Move the original position's SL further out (give it room).
      2. Open a new position at the current better price.

    Position cap is derived from the risk budget so it automatically adjusts
    as RISK_PERCENT changes:
        max_positions = floor(max_total_risk_percent / risk_percent)

    A cooldown prevents the function from firing more than once per
    `cooldown_seconds` for the same symbol, stopping the runaway-open bug.
    """
    # ── Cooldown guard ────────────────────────────────────────────────
    now = _time.time()
    last = _scale_in_last_time.get(symbol, 0.0)
    if now - last < cooldown_seconds:
        return
    # ─────────────────────────────────────────────────────────────────

    positions = get_symbol_positions(symbol)
    if not positions:
        return

    buy_positions = [p for p in positions if p.type == mt5.POSITION_TYPE_BUY]
    sell_positions = [p for p in positions if p.type == mt5.POSITION_TYPE_SELL]

    sym = mt5.symbol_info(symbol)
    if not sym:
        return

    point = sym.point
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return

    # Dynamic cap from risk budget
    max_positions = max(1, int(max_total_risk_percent / risk_percent))
    
    # Throttle Scale-In status log to 30 seconds
    now = _time.time()
    global _scale_in_last_log_ts
    if now - _scale_in_last_log_ts > 30:
        print(
            f"⚖️  Scale-In | cap={max_positions} "
            f"({max_total_risk_percent:.0f}% / {risk_percent:.1f}% per trade)"
        )
        _scale_in_last_log_ts = now

    # Dynamic loss threshold from current volatility
    current_vol = calculate_volatility(symbol, timeframe, 1, feature_window)
    loss_threshold_pts = max(50.0, current_vol * volatility_multiplier)

    prediction = model.predict(symbol, timeframe, feature_window, rsi_period)

    # ── BUY positions ─────────────────────────────────────────────────
    max_adverse_pts = sl_points_default * 2.0  # same ceiling as SELL block

    if buy_positions and len(buy_positions) < max_positions and prediction > 0:
        for pos in buy_positions:
            profit_points = (tick.bid - pos.price_open) / point

            # SAFETY: never scale in beyond 2× SL distance
            adverse_pts = -profit_points
            if adverse_pts >= max_adverse_pts:
                print(
                    f"🛑 Scale-In BLOCKED #{pos.ticket}: adverse move {adverse_pts:.1f} pts "
                    f">= ceiling {max_adverse_pts:.0f} pts"
                )
                break

            # Case A: Averaging Down (Scale-In)
            if profit_points < -loss_threshold_pts:
                print(f"📉 Scale-In BUY #{pos.ticket} losing {-profit_points:.1f} pts — trend still UP")
                # Do NOT widen the original SL
                execute_buy_market(symbol, timeframe, sl_points_default, feature_window, rsi_period, risk_percent, model, allow_multiple=True)
                _scale_in_last_time[symbol] = _time.time()
                break

            # Case B: Pyramiding (Scaling Up) - Adding when in profit if "even more sure"
            elif profit_points > current_vol * 1.5 and prediction > current_vol * 0.8:
                print(f"🚀 Pyramiding BUY #{pos.ticket} profiting {profit_points:.1f} pts — trend is STRONG")
                execute_buy_market(symbol, timeframe, sl_points_default, feature_window, rsi_period, risk_percent, model, allow_multiple=True)
                _scale_in_last_time[symbol] = _time.time()
                break

    # ── SELL positions ────────────────────────────────────────────────
    # Hard ceiling: never scale in if any existing SELL is losing more than
    # 2× the default SL distance.  This prevents runaway averaging-down
    # during a trend surge (e.g. the $13 XAUUSDm bull move scenario).
    MAX_ADVERSE_SCALE_MULTIPLIER = 2.0
    max_adverse_pts = sl_points_default * MAX_ADVERSE_SCALE_MULTIPLIER

    if sell_positions and len(sell_positions) < max_positions and prediction < 0:
        for pos in sell_positions:
            profit_points = (pos.price_open - tick.ask) / point

            # SAFETY: if adverse move exceeds hard ceiling, skip ALL scale-in
            adverse_pts = -profit_points  # positive = losing
            if adverse_pts >= max_adverse_pts:
                print(
                    f"🛑 Scale-In BLOCKED #{pos.ticket}: adverse move {adverse_pts:.1f} pts "
                    f">= ceiling {max_adverse_pts:.0f} pts — not averaging into a trend move"
                )
                break

            # Case A: Averaging Down (Scale-In) — only within the safe zone
            if profit_points < -loss_threshold_pts:
                print(f"📈 Scale-In SELL #{pos.ticket} losing {-profit_points:.1f} pts — trend still DOWN")
                # Do NOT widen the original SL — let each position stand on its own
                execute_sell_market(symbol, timeframe, sl_points_default, feature_window, rsi_period, risk_percent, model, allow_multiple=True)
                _scale_in_last_time[symbol] = _time.time()
                break

            # Case B: Pyramiding (Scaling Up)
            elif profit_points > current_vol * 1.5 and abs(prediction) > current_vol * 0.8:
                print(f"🚀 Pyramiding SELL #{pos.ticket} profiting {profit_points:.1f} pts — trend is STRONG")
                execute_sell_market(symbol, timeframe, sl_points_default, feature_window, rsi_period, risk_percent, model, allow_multiple=True)
                _scale_in_last_time[symbol] = _time.time()
                break


# ══════════════════════════════════════════════════════════════════════
# Reversal Cut-Loss (close positions when trend confirmed to have flipped)
# ══════════════════════════════════════════════════════════════════════


def manage_reversal_cut_loss(
    symbol: str, prediction: float, loss_threshold_pct: float = 0.7
) -> None:
    """
    Close all open positions that are now against the confirmed ML direction.

    - BUY positions closed when prediction turns negative (bearish).
    - SELL positions closed when prediction turns positive (bullish).

    Both the short-term (M1) and long-term (H1) models must agree on the
    reversal before closing (call this with the combined signal from main.py).
    """
    positions = get_symbol_positions(symbol)
    if not positions:
        return

    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return

    for pos in positions:
        # Calculate loss as percentage of initial SL
        sl_dist = abs(pos.price_open - pos.sl) if pos.sl > 0 else 0
        current_loss = 0.0
        if pos.type == mt5.POSITION_TYPE_BUY:
            current_loss = pos.price_open - tick.bid
        else:
            current_loss = tick.ask - pos.price_open

        loss_pct = (current_loss / sl_dist) if sl_dist > 0 else 1.0

        if pos.type == mt5.POSITION_TYPE_BUY and prediction < 0:
            if loss_pct >= loss_threshold_pct:
                print(
                    f"\U0001f6d1 Cut-Loss: closing BUY #{pos.ticket} "
                    f"— market shifted bearish (pred={prediction:.5f}) & loss={loss_pct:.1%}"
                )
                mt5.order_send(
                    {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "position": pos.ticket,
                        "symbol": symbol,
                        "volume": pos.volume,
                        "type": mt5.ORDER_TYPE_SELL,
                        "price": tick.bid,
                        "deviation": 20,
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                )

        elif pos.type == mt5.POSITION_TYPE_SELL and prediction > 0:
            if loss_pct >= loss_threshold_pct:
                print(
                    f"\U0001f6d1 Cut-Loss: closing SELL #{pos.ticket} "
                    f"— market shifted bullish (pred={prediction:.5f}) & loss={loss_pct:.1%}"
                )
                mt5.order_send(
                    {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "position": pos.ticket,
                        "symbol": symbol,
                        "volume": pos.volume,
                        "type": mt5.ORDER_TYPE_BUY,
                        "price": tick.ask,
                        "deviation": 20,
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                )


# ══════════════════════════════════════════════════════════════════════
# Market Status & Dynamic Trailing Utils
# ══════════════════════════════════════════════════════════════════════


def check_symbol_trading_status(symbol: str) -> tuple[bool, str]:
    """
    Check if the symbol is currently tradeable.
    Returns (is_tradeable, status_message)
    """
    sym = mt5.symbol_info(symbol)
    if sym is None:
        return False, "SYMBOL NOT FOUND"

    if sym.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
        return False, "DISABLED"
    elif sym.trade_mode == mt5.SYMBOL_TRADE_MODE_CLOSEONLY:
        return False, "CLOSE-ONLY"
    elif sym.trade_mode == mt5.SYMBOL_TRADE_MODE_LONGONLY:
        return True, "LONG-ONLY"
    elif sym.trade_mode == mt5.SYMBOL_TRADE_MODE_SHORTONLY:
        return True, "SHORT-ONLY"

    return True, "FULL"


def get_dynamic_trailing_step(symbol: str, timeframe: int, feature_window: int) -> int:
    """
    Calculate dynamic trailing step based on volatility.
    Range [10, 50]. High Vol -> 10, Low Vol -> 50.
    """
    cur_vol = calculate_volatility(symbol, timeframe, 0, feature_window)
    avg_vol = calculate_volatility(symbol, timeframe, 1, 50)  # 50-bar rolling average

    if avg_vol <= 0:
        return 10  # Default to conservative/tight if no data

    ratio = cur_vol / avg_vol

    # Linear interpolation:
    # ratio 2.0 -> step 10
    # ratio 0.5 -> step 50
    # Formula: step = 50 - (ratio - 0.5) * (40 / 1.5)
    step = 50 - (ratio - 0.5) * (40.0 / 1.5)

    return max(10, min(50, int(step)))



# ══════════════════════════════════════════════════════════════════════
# Expose trade logger for polling from main loop
# ══════════════════════════════════════════════════════════════════════

def poll_trade_log() -> None:
    """Call periodically from main loop to flush closed deals to trades.csv."""
    _trade_logger.poll_closed_deals()
