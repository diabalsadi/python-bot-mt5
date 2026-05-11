"""
Trade Logger
------------
Writes every trade open and close to trades.csv for post-session analysis.

Columns:
    ticket, symbol, direction, open_time, close_time,
    entry_price, exit_price, lot, sl, tp,
    profit_usd, profit_pips, duration_mins, comment

Usage:
    from tools.trade_logger import TradeLogger
    logger = TradeLogger()
    logger.log_open(result, symbol, direction, entry, sl, tp, lot)
    logger.poll_closed_deals()   # call periodically in on_tick
"""

from __future__ import annotations

import csv
import os
import time
from datetime import datetime, timezone
from typing import Optional

import MetaTrader5 as mt5

CSV_PATH = os.getenv("TRADE_LOG_PATH", "trades.csv")

_HEADERS = [
    "ticket", "symbol", "direction",
    "open_time", "close_time",
    "entry_price", "exit_price",
    "lot", "sl", "tp",
    "profit_usd", "profit_pips", "duration_mins",
    "comment",
]


class TradeLogger:
    """
    XGBoost-backed prediction model with StandardScaler normalisation.

    Falls back to OLS (via np.linalg.lstsq) if xgboost is not installed.
    Maintains the same public interface as the old LinearRegressionModel
    so all call-sites in main.py / strategy.py work unchanged.

    Public attributes kept for backward compatibility:
        is_trained, beta0..beta6 (mapped to feature importances when using XGB),
        prediction_horizon
    """

    def __init__(self, csv_path: str = CSV_PATH, on_close=None) -> None:
        """
        Args:
            csv_path: Path to the output CSV file.
            on_close: Optional callback(direction: str, profit: float) called
                      after every position close. Used by main.py to feed the
                      consecutive-loss circuit breaker.
        """
        self.csv_path = csv_path
        self._on_close = on_close  # callback(direction, profit)
        self._open_trades: dict[int, dict] = {}
        self._last_poll_ts: float = 0.0
        self._poll_interval: float = 30.0
        self._last_deal_ts: int = int(time.time())

        self._ensure_header()

    # ── Public API ────────────────────────────────────────────────────

    def log_open(
        self,
        result,           # mt5.OrderSendResult
        symbol: str,
        direction: str,   # "BUY" or "SELL"
        entry: float,
        sl: float,
        tp: float,
        lot: float,
        comment: str = "",
    ) -> None:
        """Call immediately after a successful order_send for a new position."""
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            return

        ticket = result.order
        self._open_trades[ticket] = {
            "ticket": ticket,
            "symbol": symbol,
            "direction": direction,
            "open_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "entry_price": entry,
            "sl": sl,
            "tp": tp,
            "lot": lot,
            "comment": comment,
        }
        print(f"📋 Trade logged open  #{ticket} | {direction} {symbol} @ {entry:.5f}")

    def poll_closed_deals(self) -> None:
        """
        Scan MT5 deal history since last check and write any newly closed
        positions to the CSV. Call this every ~30 s from on_tick.
        """
        now = time.time()
        if now - self._last_poll_ts < self._poll_interval:
            return
        self._last_poll_ts = now

        # Fetch deals from last known timestamp up to now
        from_ts = self._last_deal_ts
        to_ts = int(now) + 1

        deals = mt5.history_deals_get(from_ts, to_ts)
        if not deals:
            return

        for deal in deals:
            # Only care about position close deals (entry = 1 means out)
            if deal.entry != mt5.DEAL_ENTRY_OUT:
                continue
            if deal.time <= self._last_deal_ts:
                continue

            self._last_deal_ts = max(self._last_deal_ts, deal.time)
            self._write_close(deal)

    # ── Internal ──────────────────────────────────────────────────────

    def _write_close(self, deal) -> None:
        """Match a close deal to its open record and write a complete row."""
        ticket   = deal.position_id
        open_rec = self._open_trades.pop(ticket, None)

        sym_info = mt5.symbol_info(deal.symbol)
        point    = sym_info.point if sym_info else 0.0001

        entry_price = open_rec["entry_price"] if open_rec else 0.0
        profit_pips = round(deal.profit / (deal.volume * point * 10), 1) if deal.volume else 0.0

        open_dt_str  = open_rec["open_time"] if open_rec else ""
        close_dt_str = datetime.fromtimestamp(deal.time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        duration_mins = 0
        if open_rec:
            try:
                open_dt = datetime.strptime(open_rec["open_time"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                close_dt = datetime.fromtimestamp(deal.time, tz=timezone.utc)
                duration_mins = round((close_dt - open_dt).total_seconds() / 60, 1)
            except Exception:
                pass

        direction = open_rec["direction"] if open_rec else ("BUY" if deal.type == mt5.DEAL_TYPE_BUY else "SELL")

        row = {
            "ticket":        ticket,
            "symbol":        deal.symbol,
            "direction":     direction,
            "open_time":     open_dt_str,
            "close_time":    close_dt_str,
            "entry_price":   entry_price,
            "exit_price":    round(deal.price, 5),
            "lot":           deal.volume,
            "sl":            open_rec["sl"] if open_rec else 0.0,
            "tp":            open_rec["tp"] if open_rec else 0.0,
            "profit_usd":    round(deal.profit, 2),
            "profit_pips":   profit_pips,
            "duration_mins": duration_mins,
            "comment":       open_rec["comment"] if open_rec else deal.comment,
        }

        self._append_row(row)
        status = "✅" if deal.profit >= 0 else "❌"
        print(
            f"📋 Trade logged close {status} #{ticket} | "
            f"{direction} {deal.symbol} | P&L: ${deal.profit:+.2f} "
            f"({profit_pips:+.1f} pips) | {duration_mins} min"
        )

        # Notify caller (e.g. consecutive-loss circuit breaker in main.py)
        if self._on_close is not None:
            try:
                self._on_close(direction, deal.profit)
            except Exception as e:
                print(f"⚠️  TradeLogger on_close callback error: {e}")

    def _ensure_header(self) -> None:
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=_HEADERS).writeheader()

    def _append_row(self, row: dict) -> None:
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_HEADERS, extrasaction="ignore")
            writer.writerow(row)
