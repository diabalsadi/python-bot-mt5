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
from datetime import datetime, timezone
from typing import Optional

import MetaTrader5 as mt5
import numpy as np

logging.basicConfig(
    filename="latency.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from actions.connection import initialize_connection, ensure_connected
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
    poll_trade_log,
    set_trade_logger_callback,
)
from indicators.atr import check_high_volatility
from indicators.volatility import calculate_volatility
from indicators.fibonacci import get_m30_fibo_levels
from ml.model import LinearRegressionModel
from ml.rl_agent import RLAgent, build_state, ACTIONS
from ml.features import get_features
from indicators.regime import detect_regime, Regime
from mt5_tool.symbol import get_symbol
from tools.print import pretty_print
from tools.trade_logger import TradeLogger

# ──────────────────────────────────────────────────────────────────────
# Configuration  (mirrors MQL5 input block)
# ──────────────────────────────────────────────────────────────────────
SYMBOL = os.getenv("MT5_SYMBOL", "XAUUSDm")
TIMEFRAME = mt5.TIMEFRAME_M1

# ML — short-term model (M1, last 150 bars for fast adaptation)
ML_TRAINING_BARS = 1440  # 24 hour of m1 candles
ML_FEATURE_WINDOW = 12
ML_RETRAIN_INTERVAL = 10  # bars between retrains
ML_PREDICTION_HORIZON = 15

# ML — medium-term model (M15, last 1200 bars = ~12 days)
MID_TIMEFRAME = mt5.TIMEFRAME_M15
MID_TRAINING_BARS = 1200  
MID_RETRAIN_INTERVAL = 45  # retrain mid model every 45 short ticks

# ML — long-term trend model (H1, ~3 months = 2160 bars)
LONG_TIMEFRAME = mt5.TIMEFRAME_H1
LONG_TRAINING_BARS = 2160  # ~3 months of H1 candles
LONG_RETRAIN_INTERVAL = 60  # retrain long model every 60 short ticks

# Indicators
RSI_PERIOD = 14
EMA_PERIOD = 50
ATR_PERIOD = 14
TREND_BARS = 15

# Risk / order management
SL_POINTS = 2000
RISK_PERCENT = 10.0
TRAILING_STEP_POINTS = 300
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
ENABLE_SCALE_IN = True  # Average-down when trade loses but trend holds
MAX_TOTAL_SCALE_RISK_PERCENT = 30.0  # Total risk budget; cap = this / RISK_PERCENT
SCALE_IN_VOL_MULTIPLIER = 1.0  # Loss threshold = volatility × this value
SCALE_IN_COOLDOWN_SECS = 60.0  # Min seconds between scale-in trades per symbol
ENABLE_REVERSAL_CUT_LOSS = True  # Close positions when BOTH models confirm reversal

# S/R Entry strategy
ENABLE_SR_ENTRIES = True  # Trade bounces + breakouts from S/R levels
SR_PROXIMITY_POINTS = 150  # Points from level to trigger mean-reversion entry
SR_LOOKBACK_BARS = 50  # Bars used to detect S/R levels


# ──────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────
model = LinearRegressionModel()  # short-term M1 model
mid_model = LinearRegressionModel()  # medium-term M5 model
long_model = LinearRegressionModel()  # long-term H1 model (3 months)

# RL gating agent
rl_agent = RLAgent(weights_path="rl_weights.npz")
_last_rl_state: Optional[np.ndarray] = None
_last_rl_action: int = 0
total_bars = 0
ml_train_counter = 0
mid_train_counter = 0
long_train_counter = 0
last_deletion_ts = 0.0
last_tick_time_msc = 0
last_bar_time = 0
last_logged_bid = 0.0

# Throttle timestamps (replaces function-attribute hacks)
_last_status_log_ts = 0.0
_last_model_log_bar = 0
_last_analysis_ts = 0.0

# ── Consecutive loss circuit breaker ──────────────────────────────────
# If the bot takes MAX_CONSECUTIVE_LOSSES losses in the same direction
# without a win, it pauses that direction for LOSS_PAUSE_SECONDS.
MAX_CONSECUTIVE_LOSSES = 3
LOSS_PAUSE_SECONDS     = 300   # 5 minutes

_consecutive_sell_losses = 0
_consecutive_buy_losses  = 0
_sell_paused_until: float = 0.0
_buy_paused_until:  float = 0.0
_last_logged_direction: str = ""


def _record_trade_outcome(direction: str, profit: float) -> None:
    """Call after a position closes to update the loss counter and train RL."""
    global _consecutive_sell_losses, _consecutive_buy_losses
    global _sell_paused_until, _buy_paused_until

    # ── RL online update ──────────────────────────────────────────────
    if _last_rl_state is not None:
        action = 1 if direction == "BUY" else 2
        next_state = _last_rl_state.copy()  # best approximation available
        rl_agent.store(_last_rl_state, action, profit, next_state, done=False)
        loss = rl_agent.learn()
        if loss > 0:
            logging.info(f"RL online update | dir={direction} P&L=${profit:+.2f} loss={loss:.4f} ε={rl_agent.epsilon:.3f}")

        # Persist weights periodically (every 50 online updates)
        if rl_agent._steps > 0 and rl_agent._steps % 50 == 0:
            rl_agent.save("rl_weights.npz")

    # ── Consecutive loss circuit breaker ─────────────────────────────
    if direction == "SELL":
        if profit < 0:
            _consecutive_sell_losses += 1
            if _consecutive_sell_losses >= MAX_CONSECUTIVE_LOSSES:
                _sell_paused_until = time.time() + LOSS_PAUSE_SECONDS
                print(
                    f"⏸  SELL paused for {LOSS_PAUSE_SECONDS}s after "
                    f"{_consecutive_sell_losses} consecutive losses"
                )
        else:
            _consecutive_sell_losses = 0
    else:
        if profit < 0:
            _consecutive_buy_losses += 1
            if _consecutive_buy_losses >= MAX_CONSECUTIVE_LOSSES:
                _buy_paused_until = time.time() + LOSS_PAUSE_SECONDS
                print(
                    f"⏸  BUY paused for {LOSS_PAUSE_SECONDS}s after "
                    f"{_consecutive_buy_losses} consecutive losses"
                )
        else:
            _consecutive_buy_losses = 0


# Initialize trade logger with callback
_trade_logger = TradeLogger(on_close=_record_trade_outcome)


def _is_strong_trend_against(direction: str, symbol: str, timeframe: int, n_bars: int = 5) -> bool:
    """
    Return True if the last `n_bars` M1 candles show a strong directional move
    that is AGAINST the proposed entry direction.

    Logic:
      - Compute the net move over the last n_bars candles.
      - If proposing SELL but price has moved up by > 1× ATR → strong uptrend → block.
      - If proposing BUY  but price has moved dn by > 1× ATR → strong downtrend → block.

    This prevents the bot from entering counter-trend during a surge.
    """
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, n_bars + 14)
    if rates is None or len(rates) < n_bars + 1:
        return False

    closes = rates["close"]
    highs  = rates["high"]
    lows   = rates["low"]

    # ATR(14) on the fetched window
    trs = []
    for i in range(1, len(rates)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i]  - closes[i - 1]),
        )
        trs.append(tr)
    atr = sum(trs[-14:]) / 14 if len(trs) >= 14 else sum(trs) / max(len(trs), 1)

    net_move = closes[-1] - closes[-n_bars]   # positive = price moved up

    if direction == "SELL" and net_move > atr:
        print(f"🚫 Trend gate: SELL blocked — last {n_bars} bars moved UP {net_move:.3f} (ATR={atr:.3f})")
        return True
    if direction == "BUY"  and net_move < -atr:
        print(f"🚫 Trend gate: BUY blocked — last {n_bars} bars moved DN {-net_move:.3f} (ATR={atr:.3f})")
        return True

    return False


# ──────────────────────────────────────────────────────────────────────
# Session filter — block the NY open chaos window (19:00–20:30 UTC)
# Both live sessions blew up exactly in this window
# ──────────────────────────────────────────────────────────────────────

# Configurable: (hour_utc, minute_utc) pairs for block window
_SESSION_BLOCK_START = (19, 0)   # 19:00 UTC — London close / NY open overlap
_SESSION_BLOCK_END   = (20, 30)  # 20:30 UTC


def _is_session_blocked() -> bool:
    """Return True during the NY open overlap window (19:00–20:30 UTC)."""
    now_utc = datetime.now(timezone.utc)
    h, m = now_utc.hour, now_utc.minute
    cur_mins  = h * 60 + m
    start_mins = _SESSION_BLOCK_START[0] * 60 + _SESSION_BLOCK_START[1]
    end_mins   = _SESSION_BLOCK_END[0]   * 60 + _SESSION_BLOCK_END[1]
    return start_mins <= cur_mins <= end_mins


# ──────────────────────────────────────────────────────────────────────
# Regime cache — only re-detect every 5 minutes (expensive)
# ──────────────────────────────────────────────────────────────────────

_regime_cache: dict = {"regime": Regime.UNKNOWN, "confidence": 0.0, "ts": 0.0}
_REGIME_CACHE_TTL = 300.0   # seconds


def _get_regime() -> tuple[Regime, float]:
    """Return cached regime, refreshing every 5 minutes."""
    global _regime_cache
    if time.time() - _regime_cache["ts"] > _REGIME_CACHE_TTL:
        r, c = detect_regime(SYMBOL, TIMEFRAME)
        _regime_cache = {"regime": r, "confidence": c, "ts": time.time()}
        print(f"🔭 Regime: {r.value}  confidence={c:.2f}")
    return _regime_cache["regime"], _regime_cache["confidence"]


# ──────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────


def on_tick() -> None:
    """Called on every price tick — mirrors MQL5 OnTick()."""
    global total_bars, ml_train_counter, mid_train_counter, long_train_counter
    global last_deletion_ts, last_tick_time_msc, last_logged_bid, last_bar_time
    global _last_status_log_ts, _last_model_log_bar, _last_analysis_ts

    # Reconnection guard — skip tick if MT5 is unreachable
    if not ensure_connected():
        return

    # Measure tick latency
    t0 = time.perf_counter()
    tick = mt5.symbol_info_tick(SYMBOL)
    app_terminal_speed_ms = int((time.perf_counter() - t0) * 1000)

    # Reconnection guard — bail out if MT5 terminal is not reachable
    if not ensure_connected():
        return

    if tick is not None and tick.time_msc != last_tick_time_msc:
        last_tick_time_msc = tick.time_msc

        # 0. Market Status Guard
        is_open, status_msg = check_symbol_trading_status(SYMBOL)
        if not is_open:
            now = time.time()
            if now - _last_status_log_ts > 30:
                print(f"⚠️  PAUSED (Market Status: {status_msg})")
                _last_status_log_ts = now
            return

        if abs(tick.bid - last_logged_bid) >= 1.0:
            last_logged_bid = tick.bid

            terminal_info = mt5.terminal_info()
            broker_ping_ms = int(terminal_info.ping_last / 1000) if terminal_info else 0

            logging.info(
                f"PRICE UPDATE | {SYMBOL} Bid: {tick.bid:.5f} Ask: {tick.ask:.5f} | "
                f"Step: {TRAILING_STEP_POINTS} | Broker: {broker_ping_ms}ms | App: {app_terminal_speed_ms}ms"
            )
            if model.is_trained and _last_model_log_bar != last_bar_time:
                print(
                    f"🤖 ML Model Updated | "
                    f"β0={model.beta0:.4f}  β1={model.beta1:.4f}  "
                    f"β2={model.beta2:.4f}  β3={model.beta3:.4f}  β4={model.beta4:.4f}  "
                    f"β5={model.beta5:.4f}  β6={model.beta6:.4f}"
                )
                _last_model_log_bar = last_bar_time

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

    # 6a. Short-term model retraining (M1, only on NEW BAR to save CPU)
    current_rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 1)
    if current_rates is not None:
        this_bar_time = int(current_rates[0]["time"])
        if this_bar_time != last_bar_time or not model.is_trained:
            model.train(
                SYMBOL,
                TIMEFRAME,
                ML_TRAINING_BARS,
                ML_PREDICTION_HORIZON,
                ML_FEATURE_WINDOW,
                RSI_PERIOD,
            )
            last_bar_time = this_bar_time

    # 6a.1 Medium-term model retraining (M15)
    mid_train_counter += 1
    if mid_train_counter >= MID_RETRAIN_INTERVAL or not mid_model.is_trained:
        mid_model.train(SYMBOL, MID_TIMEFRAME, MID_TRAINING_BARS, ML_PREDICTION_HORIZON, ML_FEATURE_WINDOW, RSI_PERIOD)
        mid_train_counter = 0

    # 6a.2 Long-term trend model retraining (H1, ~3 months, less frequent)
    long_train_counter += 1
    if long_train_counter >= LONG_RETRAIN_INTERVAL or not long_model.is_trained:
        long_model.train(
            SYMBOL,
            LONG_TIMEFRAME,
            LONG_TRAINING_BARS,
            ML_PREDICTION_HORIZON,
            ML_FEATURE_WINDOW,
            RSI_PERIOD,
        )
        long_train_counter = 0

    # 6a.5 Analysis & Signal Logs (Throttled to reduce noise)
    now_ts = time.time()
    show_analysis = (now_ts - _last_analysis_ts > 30) or (
        abs(tick.bid - last_logged_bid) >= 5.0
    )

    target_15m, direction_15m = get_15m_lookahead_confirmation(
        SYMBOL, TIMEFRAME, model, ML_FEATURE_WINDOW, RSI_PERIOD
    )
    if show_analysis:
        print(
            f"📊 15M Lookahead | Target: {target_15m:.5f} | Direction: {direction_15m}"
        )

    # 6c. Predictions: short-term (M1) + medium-term (M15) + long-term (H1) combined signal
    prediction = model.predict(SYMBOL, TIMEFRAME, ML_FEATURE_WINDOW, RSI_PERIOD)
    mid_prediction = mid_model.predict(SYMBOL, MID_TIMEFRAME, ML_FEATURE_WINDOW, RSI_PERIOD)
    long_prediction = long_model.predict(
        SYMBOL, LONG_TIMEFRAME, ML_FEATURE_WINDOW, RSI_PERIOD
    )

    # ── CONFLUENCE FILTER ──────────────────────────────────────────────
    # M1 and M15 must agree AND M15 must not be overwhelmingly opposed.
    # If mid_prediction is >3× stronger than short in the opposite direction,
    # mid wins — this prevents entering BUY when mid screams SELL.
    mid_dominates_opposite = (
        prediction > 0 and mid_prediction < 0 and abs(mid_prediction) > abs(prediction) * 3
    ) or (
        prediction < 0 and mid_prediction > 0 and abs(mid_prediction) > abs(prediction) * 3
    )

    m1_m15_agree = (
        (prediction > 0 and mid_prediction > 0) or
        (prediction < 0 and mid_prediction < 0)
    ) and not mid_dominates_opposite

    # H1 soft filter: only blocks if strongly opposed (>50% of H1 volatility)
    vol_h1 = calculate_volatility(SYMBOL, LONG_TIMEFRAME, 1, ML_FEATURE_WINDOW)
    h1_opposed = (prediction > 0 and long_prediction < -vol_h1 * 0.5) or \
                 (prediction < 0 and long_prediction > vol_h1 * 0.5)

    same_direction = m1_m15_agree and not h1_opposed
    confirmed_prediction = prediction if same_direction else 0.0
    predicted_price = tick.bid + prediction

    if show_analysis:
        status_h1 = "OPPOSED (BLOCKED)" if h1_opposed else "OK"
        print(
            f"ML Short: {prediction:+.5f} | Mid: {mid_prediction:+.5f} | Long: {long_prediction:+.5f} ({status_h1}) | "
            f"Confirmed: {confirmed_prediction:+.5f} | Target: {predicted_price:.5f}"
        )
        _last_analysis_ts = now_ts
    # 6b. Get ML entry levels — only if confluence confirmed
    # If confirmed_prediction == 0 (models disagree), skip entirely
    if confirmed_prediction == 0.0:
        buy_level = sell_level = 0.0
    else:
        buy_level, sell_level = get_ml_entry_levels(
            SYMBOL,
            TIMEFRAME,
            TREND_BARS,
            ML_FEATURE_WINDOW,
            RSI_PERIOD,
            EMA_PERIOD,
            model,
            prediction=confirmed_prediction,
            verbose=show_analysis,
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

    if show_analysis:
        if buy_level > 0:
            print(
                f"💰 BUY Quick Profit | Tight TP: {buy_tight_tp:.5f} | Extended TP: {buy_extended_tp:.5f}"
            )
        if sell_level > 0:
            print(
                f"💰 SELL Quick Profit | Tight TP: {sell_tight_tp:.5f} | Extended TP: {sell_extended_tp:.5f}"
            )

    cancel_stale_orders(SYMBOL, confirmed_prediction)

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
            MAX_TOTAL_SCALE_RISK_PERCENT,
            SCALE_IN_VOL_MULTIPLIER,
            SCALE_IN_COOLDOWN_SECS,
        )

    # 6c.2 Reversal Cut-Loss: close all positions when ML confirms trend has flipped
    if ENABLE_REVERSAL_CUT_LOSS:
        # Only cut losses when BOTH models confirm the reversal
        manage_reversal_cut_loss(SYMBOL, confirmed_prediction)

    # 6c.3 S/R Entries: sell at resistance, buy at support (+ breakouts)
    if ENABLE_SR_ENTRIES:
        execute_sr_entries(
            SYMBOL,
            TIMEFRAME,
            SL_POINTS,
            ML_FEATURE_WINDOW,
            RSI_PERIOD,
            RISK_PERCENT,
            model,
            confirmed_prediction,
            SR_PROXIMITY_POINTS,
            SR_LOOKBACK_BARS,
        )

    # 6d. Build RL state and gate entry signals
    global _last_rl_state, _last_rl_action
    now_ts_entry = time.time()

    # SHAP dims from the short-term model (0.0 until first retrain completes)
    _raw_feats = get_features(SYMBOL, TIMEFRAME, 0, ML_FEATURE_WINDOW, RSI_PERIOD) \
        if model.is_trained else None
    shap_top_norm, shap_conflict_val = (
        model.shap.shap_rl_dims(np.array(_raw_feats), prediction)
        if (model.shap.is_fitted and _raw_feats is not None) else (0.0, 0.0)
    )

    _last_rl_state = build_state(
        momentum=float(prediction),
        volatility=float(calculate_volatility(SYMBOL, TIMEFRAME, 0, ML_FEATURE_WINDOW)),
        trend=float(long_prediction),
        rsi=50.0,
        consecutive_losses=max(_consecutive_sell_losses, _consecutive_buy_losses),
        open_time=datetime.now(timezone.utc),
        recent_pnl=float(np.mean(list(rl_agent._recent_pnl))) if rl_agent._recent_pnl else 0.0,
        shap_top_norm=shap_top_norm,
        shap_conflict=shap_conflict_val,
    )

    sell_allowed = (
        now_ts_entry > _sell_paused_until
        and not _is_strong_trend_against("SELL", SYMBOL, TIMEFRAME)
    )
    buy_allowed = (
        now_ts_entry > _buy_paused_until
        and not _is_strong_trend_against("BUY", SYMBOL, TIMEFRAME)
    )

    # ── Session filter: block NY open chaos window ─────────────────
    if _is_session_blocked():
        sell_allowed = buy_allowed = False
        if int(now_ts_entry) % 120 < 2:
            print("⏰ Session filter: no entries 19:00–20:30 UTC (NY open window)")

    # ── Regime filter ──────────────────────────────────────────────
    regime, regime_conf = _get_regime()
    if regime == Regime.RANGING and regime_conf > 0.5:
        sell_allowed = buy_allowed = False
        if int(now_ts_entry) % 120 < 2:
            print(f"🔭 Regime RANGING ({regime_conf:.2f}) — no new entries")
    elif regime == Regime.TRENDING_UP and regime_conf > 0.6:
        sell_allowed = False   # don't fight a strong trend
    elif regime == Regime.TRENDING_DOWN and regime_conf > 0.6:
        buy_allowed = False

    # ── Order-flow filters: spread and bar-range ───────────────────
    from indicators.order_flow import calculate_spread_norm, calculate_bar_range_ratio
    spread_norm = calculate_spread_norm(SYMBOL, TIMEFRAME)
    bar_range   = calculate_bar_range_ratio(SYMBOL, TIMEFRAME)
    if spread_norm > 1.0:
        sell_allowed = buy_allowed = False
        if int(now_ts_entry) % 60 < 2:
            print(f"🚫 Spread too wide ({spread_norm:.2f}× ATR) — skipping entry")
    if bar_range > 2.5:
        sell_allowed = buy_allowed = False
        if int(now_ts_entry) % 60 < 2:
            print(f"🚫 Abnormal bar range ({bar_range:.2f}×) — news/spike, skipping")

    if buy_level > 0 and buy_allowed:
        rl_action, rl_reason = rl_agent.act_verbose(_last_rl_state, ml_action=1)
        _last_rl_action = rl_action
        if rl_action == 1:
            execute_buy_market(SYMBOL, TIMEFRAME, SL_POINTS, ML_FEATURE_WINDOW, RSI_PERIOD, RISK_PERCENT, model)
        else:
            print(f"🤖 RL HOLD (blocked BUY) | {rl_reason}")

    if sell_level > 0 and sell_allowed:
        rl_action, rl_reason = rl_agent.act_verbose(_last_rl_state, ml_action=2)
        _last_rl_action = rl_action
        if rl_action == 2:
            execute_sell_market(SYMBOL, TIMEFRAME, SL_POINTS, ML_FEATURE_WINDOW, RSI_PERIOD, RISK_PERCENT, model)
        else:
            print(f"🤖 RL HOLD (blocked SELL) | {rl_reason}")

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
    positions = mt5.positions_get(symbol=SYMBOL)
    pos_count = len(positions) if positions else 0
    if pos_count > 0 and show_analysis:
        print(f"🔄 Trailing {pos_count} position(s) on {SYMBOL}...")

    trail_sltp(
        SYMBOL,
        TIMEFRAME,
        TRAILING_STEP_POINTS,
        SL_POINTS,
        ML_FEATURE_WINDOW,
        RSI_PERIOD,
        model,
    )

    # Flush any closed deals to trades.csv
    poll_trade_log()

    # Periodic portfolio status (every 5 min)
    from position_manager import get_position_manager
    if int(time.time()) % 300 < 2:
        get_position_manager().report_status()
        rl_agent.report()

    # 8. Poll for newly closed trades and log them to CSV
    _trade_logger.poll_closed_deals()


def main() -> None:
    global model

    initialize_connection()
    set_trade_logger_callback(_record_trade_outcome)

    # RL agent: warm-start from any existing trade history
    if os.path.exists("trades.csv"):
        rl_agent.train_from_csv("trades.csv")
    else:
        print("⚠️  trades.csv not found — RL agent starting cold (will learn online)")

    # Initialize portfolio-level position manager
    from position_manager import get_position_manager
    get_position_manager().initialize_session()

    pretty_print(mt5.account_info(), "Account Info")

    symbol = get_symbol(SYMBOL)
    if not symbol:
        mt5.shutdown()
        return

    # Initial model training
    req_bars = (
        ML_TRAINING_BARS + ML_PREDICTION_HORIZON + ML_FEATURE_WINDOW + RSI_PERIOD + 10
    )
    available = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, req_bars)
    if available is not None and len(available) >= req_bars:
        model.train(
            SYMBOL,
            TIMEFRAME,
            ML_TRAINING_BARS,
            ML_PREDICTION_HORIZON,
            ML_FEATURE_WINDOW,
            RSI_PERIOD,
        )
    else:
        print(
            f"⚠️  Not enough M1 bars for initial training (need {req_bars}). "
            "Short model will train after enough bars accumulate."
        )

    # Initial medium-term training
    req_mid_bars = (
        MID_TRAINING_BARS + ML_PREDICTION_HORIZON + ML_FEATURE_WINDOW + RSI_PERIOD + 10
    )
    available_mid = mt5.copy_rates_from_pos(SYMBOL, MID_TIMEFRAME, 0, req_mid_bars)
    if available_mid is not None and len(available_mid) >= req_mid_bars:
        mid_model.train(
            SYMBOL,
            MID_TIMEFRAME,
            MID_TRAINING_BARS,
            ML_PREDICTION_HORIZON,
            ML_FEATURE_WINDOW,
            RSI_PERIOD,
        )
    else:
        print(
            f"⚠️  Not enough M5 bars for initial training (need {req_mid_bars}). "
            "Mid model will train after enough bars accumulate."
        )

    print(f"\n🚀 Gold Scalper v3 running on {SYMBOL} {TIMEFRAME} (Ctrl+C to stop)\n")

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
