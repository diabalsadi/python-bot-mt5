"""
Gold Scalper — Python Edition
=================================
Main entry point.  Mirrors the MQL5 EA's OnInit() + OnTick() loop.

Run:
    python main.py
"""

import time
import os
import logging

import MetaTrader5 as mt5

logging.basicConfig(
    filename="latency.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from actions.connection import initialize_connection
from actions.strategy import (
    cancel_all_orders,
    cancel_stale_orders,
    check_latency,
    check_us_session_exclusion,
    execute_buy_market,
    execute_sell_market,
    get_market_session,
    get_ml_entry_levels,
    get_quick_profit_levels,
    get_15m_lookahead_confirmation,
    manage_scale_in,
    manage_reversal_cut_loss,
    trail_sltp,
)
from indicators.atr import check_high_volatility
from indicators.fibonacci import get_m30_fibo_levels
from ml.model import LinearRegressionModel
from mt5_tool.symbol import get_symbol, stream_ticks
from tools.print import pretty_print

# ──────────────────────────────────────────────────────────────────────
# Configuration  (mirrors MQL5 input block)
# ──────────────────────────────────────────────────────────────────────
SYMBOL = os.getenv("MT5_SYMBOL", "XAUUSDm")
TIMEFRAME = mt5.TIMEFRAME_M1

# ML
ML_TRAINING_BARS = 150
ML_FEATURE_WINDOW = 12
ML_RETRAIN_INTERVAL = 10  # bars between retrains
ML_PREDICTION_HORIZON = 15

# Indicators
RSI_PERIOD = 14
EMA_PERIOD = 50
ATR_PERIOD = 14
TREND_BARS = 5

# Risk / order management
SL_POINTS = 2000
RISK_PERCENT = 10.0
TRAILING_STEP_POINTS = 100
EXPIRATION_HOURS = 50
MIN_ORDER_DISTANCE_PTS = 500
RESET_ORDERS_INTERVAL = 30  # minutes

# Protection
AVOID_HIGH_VOLATILITY = False
VOLATILITY_MULTIPLIER = 2.0
MAX_LATENCY_MS = 2000
ENABLE_LATENCY_CHECK = True
USE_US_OPEN_PROTECTION = False
US_OPEN_PROTECTION_HRS = 2
US_START_HOUR = 15

# Scale-In / Cut-Loss strategy
ENABLE_SCALE_IN = True  # Open new trades in same direction when losing but trend holds
MAX_SCALE_IN_POSITIONS = (
    3  # Max open positions per direction before scale-in is blocked
)
SCALE_IN_VOL_MULTIPLIER = 1.0  # Loss threshold = current volatility × this multiplier
ENABLE_REVERSAL_CUT_LOSS = True  # Close all positions when ML flips direction


# ──────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────
model = LinearRegressionModel()
total_bars = 0
ml_train_counter = 0
last_deletion_ts = 0.0
last_tick_time_msc = 0
last_logged_bid = 0.0

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _bars_available() -> int:
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 1)
    if rates is None:
        return 0
    return int(
        mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 5000).shape[0]
    )  # approximate


# ──────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────


def on_tick() -> None:
    """Called on every price tick — mirrors MQL5 OnTick()."""
    global total_bars, ml_train_counter, last_deletion_ts, last_tick_time_msc, last_logged_bid

    # Measure tick latency
    t0 = time.perf_counter()
    tick = mt5.symbol_info_tick(SYMBOL)
    app_terminal_speed_ms = int((time.perf_counter() - t0) * 1000)

    if tick is not None and tick.time_msc != last_tick_time_msc:
        last_tick_time_msc = tick.time_msc

        if abs(tick.bid - last_logged_bid) >= 1.0:
            last_logged_bid = tick.bid

            terminal_info = mt5.terminal_info()
            broker_ping_ms = int(terminal_info.ping_last / 1000) if terminal_info else 0

            logging.info(
                f"PRICE UPDATE | {SYMBOL} Bid: {tick.bid:.5f} Ask: {tick.ask:.5f} | Broker Ping: {broker_ping_ms}ms | App-Terminal Speed: {app_terminal_speed_ms}ms"
            )
            if model.is_trained:
                print(
                    f"🤖 ML Model Trained | "
                    f"β0={model.beta0:.4f}  β1={model.beta1:.4f}  "
                    f"β2={model.beta2:.4f}  β3={model.beta3:.4f}  β4={model.beta4:.4f}  "
                    f"β5={model.beta5:.4f}  β6={model.beta6:.4f}"
                )

    # 1. US open protection (highest priority)
    if USE_US_OPEN_PROTECTION and check_us_session_exclusion(
        US_START_HOUR, US_OPEN_PROTECTION_HRS
    ):
        session = get_market_session(us_start=US_START_HOUR)
        print(f"⚠️  PAUSED (US Open protection) | Session: {session}")
        return

    # 2. Latency guard
    if ENABLE_LATENCY_CHECK:
        ok, latency = check_latency(SYMBOL, MAX_LATENCY_MS)
        if not ok:
            print(f"⚠️  PAUSED (latency {latency} ms > {MAX_LATENCY_MS} ms)")
            return

    # 3. High-volatility guard
    if AVOID_HIGH_VOLATILITY:
        is_high, cur_atr, avg_atr = check_high_volatility(
            SYMBOL, TIMEFRAME, ATR_PERIOD, VOLATILITY_MULTIPLIER
        )
        if is_high:
            print(f"⚠️  PAUSED (high volatility) | ATR={cur_atr:.5f} avg={avg_atr:.5f}")
            return

    # 4. Periodic order reset
    now = time.time()
    if last_deletion_ts == 0:
        last_deletion_ts = now
    elif (now - last_deletion_ts) >= RESET_ORDERS_INTERVAL * 60:
        print(f"🗑️  Resetting all pending orders ({RESET_ORDERS_INTERVAL} min interval)")
        cancel_all_orders(SYMBOL)
        last_deletion_ts = now

    # 6a. Model retraining (every tick)
    model.train(
        SYMBOL,
        TIMEFRAME,
        ML_TRAINING_BARS,
        ML_PREDICTION_HORIZON,
        ML_FEATURE_WINDOW,
        RSI_PERIOD,
    )

    # 6a.5 Get 15-minute lookahead confirmation
    target_15m, direction_15m = get_15m_lookahead_confirmation(
        SYMBOL, TIMEFRAME, model, ML_FEATURE_WINDOW, RSI_PERIOD
    )
    print(f"📊 15M Lookahead | Target: {target_15m:.5f} | Direction: {direction_15m}")

    # 6b. Get ML entry levels
    buy_level, sell_level = get_ml_entry_levels(
        SYMBOL,
        TIMEFRAME,
        TREND_BARS,
        ML_FEATURE_WINDOW,
        RSI_PERIOD,
        EMA_PERIOD,
        model,
    )

    # 6b.5 Get quick-profit levels for tight TP (15m S/R based)
    buy_tight_tp, buy_extended_tp = (
        get_quick_profit_levels(
            SYMBOL, TIMEFRAME, 1, model, ML_FEATURE_WINDOW, RSI_PERIOD
        )
        if buy_level > 0
        else (-1.0, -1.0)
    )

    sell_tight_tp, sell_extended_tp = (
        get_quick_profit_levels(
            SYMBOL, TIMEFRAME, -1, model, ML_FEATURE_WINDOW, RSI_PERIOD
        )
        if sell_level > 0
        else (-1.0, -1.0)
    )

    if buy_level > 0:
        print(
            f"💰 BUY Quick Profit | Tight TP: {buy_tight_tp:.5f} | Extended TP: {buy_extended_tp:.5f}"
        )
    if sell_level > 0:
        print(
            f"💰 SELL Quick Profit | Tight TP: {sell_tight_tp:.5f} | Extended TP: {sell_extended_tp:.5f}"
        )

    # 6c. Cancel orders that contradict the current prediction
    prediction = model.predict(SYMBOL, TIMEFRAME, ML_FEATURE_WINDOW, RSI_PERIOD)
    predicted_price = tick.bid + prediction
    print(
        f"ML Prediction: {prediction:+.5f} | Predicted Target Price: {predicted_price:.5f}"
    )
    cancel_stale_orders(SYMBOL, prediction)

    # 6c.1 Scale-In: add to position at better price if trend holds & trade is losing
    if ENABLE_SCALE_IN:
        manage_scale_in(
            SYMBOL,
            TIMEFRAME,
            SL_POINTS,
            ML_FEATURE_WINDOW,
            RSI_PERIOD,
            RISK_PERCENT,
            model,
            MAX_SCALE_IN_POSITIONS,
            SCALE_IN_VOL_MULTIPLIER,
        )

    # 6c.2 Reversal Cut-Loss: close all positions when ML confirms trend has flipped
    if ENABLE_REVERSAL_CUT_LOSS:
        manage_reversal_cut_loss(SYMBOL, prediction)

    # 6d. Place new orders
    if buy_level > 0:
        execute_buy_market(
            SYMBOL,
            TIMEFRAME,
            SL_POINTS,
            ML_FEATURE_WINDOW,
            RSI_PERIOD,
            RISK_PERCENT,
            model,
        )

    if sell_level > 0:
        execute_sell_market(
            SYMBOL,
            TIMEFRAME,
            SL_POINTS,
            ML_FEATURE_WINDOW,
            RSI_PERIOD,
            RISK_PERCENT,
            model,
        )

    # 6e. Place Fibonacci orders
    fibo_levels = get_m30_fibo_levels(SYMBOL)
    if fibo_levels:
        is_bullish = fibo_levels.get("is_bullish", False)
        # If prediction is positive and M30 bar was bullish (retracing down for support)
        if prediction > 0 and is_bullish:
            for level_name in ["pullback_50", "pullback_61"]:
                if level_name in fibo_levels:
                    execute_buy_market(
                        SYMBOL,
                        TIMEFRAME,
                        SL_POINTS,
                        ML_FEATURE_WINDOW,
                        RSI_PERIOD,
                        RISK_PERCENT / 2.0,  # Risk half on fibo entries
                        model,
                    )

        # If prediction is negative and M30 bar was bearish (retracing up for resistance)
        elif prediction < 0 and not is_bullish:
            for level_name in ["pullback_50", "pullback_61"]:
                if level_name in fibo_levels:
                    execute_sell_market(
                        SYMBOL,
                        TIMEFRAME,
                        SL_POINTS,
                        ML_FEATURE_WINDOW,
                        RSI_PERIOD,
                        RISK_PERCENT / 2.0,  # Risk half on fibo entries
                        model,
                    )

    # 7. Trail SL/TP every tick
    trail_sltp(SYMBOL, TIMEFRAME, TRAILING_STEP_POINTS, SL_POINTS, ML_FEATURE_WINDOW)


def main() -> None:
    global model

    initialize_connection()
    pretty_print(mt5.account_info(), "Account Info")

    symbol = get_symbol(SYMBOL)
    if not symbol:
        mt5.shutdown()
        return

    # Initial model training
    req_bars = (
        ML_TRAINING_BARS + ML_PREDICTION_HORIZON + ML_FEATURE_WINDOW + RSI_PERIOD + 10
    )
    available = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, req_bars)
    if available is not None and len(available) >= req_bars:
        model.train(
            symbol,
            TIMEFRAME,
            ML_TRAINING_BARS,
            ML_PREDICTION_HORIZON,
            ML_FEATURE_WINDOW,
            RSI_PERIOD,
        )
    else:
        print(
            f"⚠️  Not enough bars for initial training (need {req_bars}). "
            "Model will train after enough bars accumulate."
        )

    print(f"\n🚀 Gold Scalper v3 running on {symbol} {TIMEFRAME} (Ctrl+C to stop)\n")

    try:
        while True:
            on_tick()
            time.sleep(0.1)  # ~10 ticks/sec; adjust as needed
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    finally:
        mt5.shutdown()
        print("👋 MT5 disconnected")


if __name__ == "__main__":
    main()
