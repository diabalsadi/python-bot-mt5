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
from dotenv import load_dotenv

load_dotenv()

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
    execute_sr_entries,
    get_market_session,
    get_ml_entry_levels,
    get_quick_profit_levels,
    get_15m_lookahead_confirmation,
    manage_scale_in,
    manage_reversal_cut_loss,
    trail_sltp,
    check_symbol_trading_status,
    get_dynamic_trailing_step,
)
from indicators.atr import check_high_volatility
from indicators.fibonacci import get_m30_fibo_levels
from ml.model import LinearRegressionModel
from mt5_tool.symbol import get_symbol, stream_ticks
from tools.print import pretty_print

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
# Read symbols from .env (comma-separated if multiple)
SYMBOLS = [s.strip() for s in os.getenv("MT5_SYMBOL", "XAUUSDm").split(",") if s.strip()]
TIMEFRAME = mt5.TIMEFRAME_M1
TIMEFRAME_M5 = mt5.TIMEFRAME_M5

# ML — short-term model (M1)
ML_TRAINING_BARS = 150
ML_FEATURE_WINDOW = 12
ML_RETRAIN_INTERVAL = 10 
ML_PREDICTION_HORIZON = 15

# ML — long-term trend model (H1)
LONG_TIMEFRAME = mt5.TIMEFRAME_H1
LONG_TRAINING_BARS = 2160
LONG_RETRAIN_INTERVAL = 60

# Indicators
RSI_PERIOD = 14
EMA_PERIOD = 50
ATR_PERIOD = 14
TREND_BARS = 5

# Risk / order management
SL_POINTS = 2000
RISK_PERCENT = 2.0
TRAILING_STEP_POINTS = 50
EXPIRATION_HOURS = 50
MIN_ORDER_DISTANCE_PTS = 500
RESET_ORDERS_INTERVAL = 30

# Protection
AVOID_HIGH_VOLATILITY = False
VOLATILITY_MULTIPLIER = 2.0
MAX_LATENCY_MS = 2000
ENABLE_LATENCY_CHECK = True
USE_US_OPEN_PROTECTION = False
US_OPEN_PROTECTION_HRS = 2
US_START_HOUR = 15

# Strategy features
ENABLE_SCALE_IN = True
ENABLE_REVERSAL_CUT_LOSS = True
ENABLE_SR_ENTRIES = True
SR_LOOKBACK_BARS = 50

# ──────────────────────────────────────────────────────────────────────
# State (Multi-Symbol Dictionaries)
# ──────────────────────────────────────────────────────────────────────
models = {s: LinearRegressionModel() for s in SYMBOLS}
m5_models = {s: LinearRegressionModel() for s in SYMBOLS}
long_models = {s: LinearRegressionModel() for s in SYMBOLS}
last_bar_times = {s: 0 for s in SYMBOLS}
last_bar_times_m5 = {s: 0 for s in SYMBOLS}
last_tick_time_msc = {s: 0 for s in SYMBOLS}
last_logged_bid = {s: 0.0 for s in SYMBOLS}
long_train_counters = {s: 0 for s in SYMBOLS}
last_deletion_ts = 0.0

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _bars_available(symbol: str) -> int:
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, 1)
    if rates is None:
        return 0
    return int(
        mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, 5000).shape[0]
    )  # approximate


# ──────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────


def on_tick(symbol: str) -> None:
    """Called on every price tick for a specific symbol."""
    global last_deletion_ts

    model = models[symbol]
    model_m5 = m5_models[symbol]
    long_model = long_models[symbol]

    # Measure tick latency
    t0 = time.perf_counter()
    tick = mt5.symbol_info_tick(symbol)
    app_terminal_speed_ms = int((time.perf_counter() - t0) * 1000)

    if tick is not None and tick.time_msc != last_tick_time_msc[symbol]:
        last_tick_time_msc[symbol] = tick.time_msc

        # 0. Market Status Guard
        is_open, status_msg = check_symbol_trading_status(symbol)
        if not is_open:
            # Throttled logging per symbol
            if not hasattr(on_tick, "_last_status_log"):
                on_tick._last_status_log = {}
            now = time.time()
            if symbol not in on_tick._last_status_log or now - on_tick._last_status_log[symbol] > 30:
                print(f"⚠️  {symbol} PAUSED (Market Status: {status_msg})")
                on_tick._last_status_log[symbol] = now
            return

        if abs(tick.bid - last_logged_bid[symbol]) >= 1.0:
            last_logged_bid[symbol] = tick.bid

            terminal_info = mt5.terminal_info()
            broker_ping_ms = int(terminal_info.ping_last / 1000) if terminal_info else 0

            logging.info(
                f"PRICE UPDATE | {symbol} Bid: {tick.bid:.5f} Ask: {tick.ask:.5f} | "
                f"Broker: {broker_ping_ms}ms | App: {app_terminal_speed_ms}ms"
            )

    # 1. US open protection (highest priority)
    if USE_US_OPEN_PROTECTION and check_us_session_exclusion(
        US_START_HOUR, US_OPEN_PROTECTION_HRS
    ):
        return

    # 2. Latency guard
    if ENABLE_LATENCY_CHECK:
        ok, latency = check_latency(symbol, MAX_LATENCY_MS)
        if not ok:
            return

    # 3. High-volatility guard
    if AVOID_HIGH_VOLATILITY:
        is_high, cur_atr, avg_atr = check_high_volatility(
            symbol, TIMEFRAME, ATR_PERIOD, VOLATILITY_MULTIPLIER
        )
        if is_high:
            return

    # 4. Periodic order reset (only once per loop, not per symbol)
    # This is handled outside the symbol loop in a real implementation usually, 
    # but here we can just do it once.
    now = time.time()
    if last_deletion_ts == 0:
        last_deletion_ts = now
    elif (now - last_deletion_ts) >= RESET_ORDERS_INTERVAL * 60:
        for s in SYMBOLS:
            cancel_all_orders(s)
        last_deletion_ts = now

    # 6a. Short-term model retraining (M1, only on NEW BAR)
    current_rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, 1)
    if current_rates is not None:
        this_bar_time = int(current_rates[0]["time"])
        if this_bar_time != last_bar_times[symbol] or not model.is_trained:
            model.train(
                symbol,
                TIMEFRAME,
                ML_TRAINING_BARS,
                ML_PREDICTION_HORIZON,
                ML_FEATURE_WINDOW,
                RSI_PERIOD,
            )
            last_bar_times[symbol] = this_bar_time

    # 6b. M5 model retraining (Only on NEW M5 BAR)
    rates_m5 = mt5.copy_rates_from_pos(symbol, TIMEFRAME_M5, 0, 1)
    if rates_m5 is not None:
        bar_time_m5 = int(rates_m5[0]["time"])
        if bar_time_m5 != last_bar_times_m5[symbol] or not model_m5.is_trained:
            model_m5.train(
                symbol,
                TIMEFRAME_M5,
                ML_TRAINING_BARS,
                ML_PREDICTION_HORIZON,
                ML_FEATURE_WINDOW,
                RSI_PERIOD,
            )
            last_bar_times_m5[symbol] = bar_time_m5

    # 6a.1 Long-term trend model retraining
    long_train_counters[symbol] += 1
    if long_train_counters[symbol] >= LONG_RETRAIN_INTERVAL or not long_model.is_trained:
        long_model.train(
            symbol,
            LONG_TIMEFRAME,
            LONG_TRAINING_BARS,
            ML_PREDICTION_HORIZON,
            ML_FEATURE_WINDOW,
            RSI_PERIOD,
        )
        long_train_counters[symbol] = 0

    # 6. LLM Agent Decision (Throttled per symbol)
    now_ts = time.time()
    if not hasattr(on_tick, "_last_llm_ts"):
        on_tick._last_llm_ts = {}
        
    if symbol not in on_tick._last_llm_ts or now_ts - on_tick._last_llm_ts[symbol] >= 60:
        from tools.market_context import get_market_snapshot, format_snapshot_for_llm
        from tools.ollama_agent import agent
        
        config = {
            'SR_LOOKBACK_BARS': SR_LOOKBACK_BARS,
            'ML_FEATURE_WINDOW': ML_FEATURE_WINDOW,
            'RSI_PERIOD': RSI_PERIOD,
            'LONG_TIMEFRAME': LONG_TIMEFRAME
        }
        
        snapshot = get_market_snapshot(symbol, TIMEFRAME, model, model_m5, long_model, config)
        if snapshot:
            prompt_context = format_snapshot_for_llm(snapshot)
            print(f"🧠 LLM is analyzing {symbol}...")
            decision = agent.get_decision(prompt_context)
            
            action = decision.get("action", "HOLD")
            reasoning = decision.get("reasoning", "No reasoning provided.")
            
            print(f"🤖 {symbol} AGENT DECISION: {action}")
            print(f"📝 REASONING: {reasoning}")
            
            if action == "BUY":
                execute_buy_market(symbol, TIMEFRAME, SL_POINTS, ML_FEATURE_WINDOW, RSI_PERIOD, RISK_PERCENT, model, reasoning=reasoning)
            elif action == "SELL":
                execute_sell_market(symbol, TIMEFRAME, SL_POINTS, ML_FEATURE_WINDOW, RSI_PERIOD, RISK_PERCENT, model, reasoning=reasoning)
            elif action == "CLOSE_ALL":
                from actions.strategy import close_all_positions, cancel_all_orders
                close_all_positions(symbol)
                cancel_all_orders(symbol)
            
            on_tick._last_llm_ts[symbol] = now_ts

    # 6c.2 Reversal Cut-Loss
    if ENABLE_REVERSAL_CUT_LOSS:
        prediction = model.predict(symbol, TIMEFRAME, ML_FEATURE_WINDOW, RSI_PERIOD)
        manage_reversal_cut_loss(symbol, prediction)

    # 7. Trail SL/TP
    trail_sltp(
        symbol,
        TIMEFRAME,
        TRAILING_STEP_POINTS,
        SL_POINTS,
        ML_FEATURE_WINDOW,
        RSI_PERIOD,
        model,
    )

    # 8. Persistent Market Data
    if not hasattr(on_tick, "_last_chroma_ts"):
        on_tick._last_chroma_ts = {}
    if symbol not in on_tick._last_chroma_ts or now_ts - on_tick._last_chroma_ts[symbol] >= 300:
        from tools.chroma_db import db
        from indicators.support_resistance import identify_sr_levels
        from indicators.liquidity_zones import get_liquidity_zones
        supports, resistances = identify_sr_levels(symbol, TIMEFRAME, SR_LOOKBACK_BARS)
        zones = get_liquidity_zones(symbol, TIMEFRAME, SR_LOOKBACK_BARS)
        db.save_sr_levels(symbol, supports, resistances)
        db.save_liquidity_zones(symbol, zones)
        on_tick._last_chroma_ts[symbol] = now_ts


def main() -> None:
    initialize_connection()
    pretty_print(mt5.account_info(), "Account Info")

    for symbol_name in SYMBOLS:
        symbol = get_symbol(symbol_name)
        if not symbol:
            print(f"❌ Symbol {symbol_name} not found!")
            continue

        # Initial model training (M1, M5)
        model = models[symbol_name]
        model_m5 = m5_models[symbol_name]
        req_bars = (
            ML_TRAINING_BARS + ML_PREDICTION_HORIZON + ML_FEATURE_WINDOW + RSI_PERIOD + 10
        )
        # Train M1
        available = mt5.copy_rates_from_pos(symbol_name, TIMEFRAME, 0, req_bars)
        if available is not None and len(available) >= req_bars:
            model.train(symbol_name, TIMEFRAME, ML_TRAINING_BARS, ML_PREDICTION_HORIZON, ML_FEATURE_WINDOW, RSI_PERIOD)
        
        # Train M5
        available_m5 = mt5.copy_rates_from_pos(symbol_name, TIMEFRAME_M5, 0, req_bars)
        if available_m5 is not None and len(available_m5) >= req_bars:
            model_m5.train(symbol_name, TIMEFRAME_M5, ML_TRAINING_BARS, ML_PREDICTION_HORIZON, ML_FEATURE_WINDOW, RSI_PERIOD)
        else:
            print(f"⚠️  {symbol_name}: Not enough M5 bars for initial training.")

    print(f"\n🚀 Multi-Symbol Scalper running on {SYMBOLS} (Ctrl+C to stop)\n")

    try:
        while True:
            for symbol in SYMBOLS:
                on_tick(symbol)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    finally:
        mt5.shutdown()
        print("👋 MT5 disconnected")


if __name__ == "__main__":
    main()
