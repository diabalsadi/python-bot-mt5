"""
Backtester for Strategy Validation
-----------------------------------
Historical simulation framework that mirrors the live on_tick() logic.

Usage:
    python backtest.py                          # uses defaults
    python backtest.py --symbol XAUUSDm --days 30
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Tuple

import numpy as np

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


# ── Lazy MT5 import (so backtest can be imported in tests without MT5) ────────
def _get_mt5():
    try:
        import MetaTrader5 as mt5
        return mt5
    except ImportError:
        return None


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_time:       datetime
    entry_price:      float
    direction:        str        = "BUY"
    volume:           float      = 0.01
    exit_time:        Optional[datetime] = None
    exit_price:       Optional[float]    = None
    profit_loss:      float      = 0.0
    profit_pct:       float      = 0.0
    duration_minutes: float      = 0.0

    def close_trade(self, exit_time: datetime, exit_price: float) -> None:
        self.exit_time  = exit_time
        self.exit_price = exit_price
        self.duration_minutes = (exit_time - self.entry_time).total_seconds() / 60
        if self.direction == "BUY":
            raw = (exit_price - self.entry_price) * self.volume * 100
            self.profit_pct = (exit_price - self.entry_price) / self.entry_price * 100
        else:
            raw = (self.entry_price - exit_price) * self.volume * 100
            self.profit_pct = (self.entry_price - exit_price) / self.entry_price * 100
        self.profit_loss = raw

    def is_closed(self) -> bool:
        return self.exit_time is not None


@dataclass
class BacktestResults:
    total_trades:          int   = 0
    winning_trades:        int   = 0
    losing_trades:         int   = 0
    win_rate:              float = 0.0
    total_profit_loss:     float = 0.0
    gross_profit:          float = 0.0
    gross_loss:            float = 0.0
    profit_factor:         float = 0.0
    avg_trade_profit:      float = 0.0
    avg_winning_trade:     float = 0.0
    avg_losing_trade:      float = 0.0
    max_drawdown_pct:      float = 0.0
    max_consecutive_losses:int   = 0
    sharpe_ratio:          float = 0.0
    final_balance:         float = 0.0

    def print_summary(self) -> None:
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Total Trades:              {self.total_trades}")
        print(f"Winning Trades:            {self.winning_trades}")
        print(f"Losing Trades:             {self.losing_trades}")
        print(f"Win Rate:                  {self.win_rate:.2f}%")
        print(f"\nProfit & Loss:")
        print(f"  Total P&L:               ${self.total_profit_loss:,.2f}")
        print(f"  Gross Profit:            ${self.gross_profit:,.2f}")
        print(f"  Gross Loss:              ${self.gross_loss:,.2f}")
        print(f"  Profit Factor:           {self.profit_factor:.2f}")
        print(f"  Final Balance:           ${self.final_balance:,.2f}")
        print(f"\nAverage Metrics:")
        print(f"  Avg Trade Profit:        ${self.avg_trade_profit:,.2f}")
        print(f"  Avg Winner:              ${self.avg_winning_trade:,.2f}")
        print(f"  Avg Loser:               ${self.avg_losing_trade:,.2f}")
        print(f"\nRisk Metrics:")
        print(f"  Max Drawdown:            {self.max_drawdown_pct:.2f}%")
        print(f"  Max Consecutive Losses:  {self.max_consecutive_losses}")
        print(f"  Sharpe Ratio:            {self.sharpe_ratio:.2f}")
        print("="*60 + "\n")


# ── Backtester ────────────────────────────────────────────────────────────────

class Backtester:
    """
    Backtester that mirrors the live strategy:
      - momentum/volatility/trend/RSI features
      - ATR trend gate (blocks counter-trend entries)
      - mid_dominates_opposite filter (mirrors the v8 confluence fix)
      - anti-hedge (one direction at a time)
      - SL/TP applied on future bars
      - configurable risk sizing per trade
    """

    def __init__(
        self,
        symbol:             str   = "XAUUSDm",
        timeframe:          int   = 1,           # TIMEFRAME_M1 = 1
        start_date:         str   = "2024-01-01",
        end_date:           str   = "2024-03-31",
        initial_balance:    float = 10000.0,
        commission_percent: float = 0.0005,
        sl_points:          int   = 150,
        tp_ratio:           float = 2.0,         # TP = sl_points * tp_ratio
        risk_percent:       float = 5.0,
        trade_cooldown_minutes: float = 5.0,
    ):
        self.symbol             = symbol
        self.timeframe          = timeframe
        self.start_date         = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        self.end_date           = datetime.strptime(end_date,   "%Y-%m-%d").replace(tzinfo=timezone.utc)
        self.initial_balance    = initial_balance
        self.commission_percent = commission_percent
        self.sl_points          = sl_points
        self.tp_ratio           = tp_ratio
        self.risk_percent       = risk_percent
        self.trade_cooldown_minutes = trade_cooldown_minutes

        self.trades:       List[Trade]                 = []
        self.results       = BacktestResults()
        self.equity_curve: List[Tuple[datetime, float]] = []

        self.current_balance = initial_balance
        self._history:       List        = []
        self._min_lookback:  int         = 25
        self._sim_direction: Optional[str] = None
        self._open_trade:    Optional[Trade] = None
        self._cooldown_until: Optional[datetime] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> BacktestResults:
        print(f"\n🔄 Backtesting {self.symbol}  {self.start_date.date()} → {self.end_date.date()}")
        print(f"   Risk: {self.risk_percent}%/trade  SL: {self.sl_points}pt  TP: {self.sl_points * self.tp_ratio:.0f}pt")
        print(f"   Hard cooldown: {self.trade_cooldown_minutes:.1f} min after each close")

        bars = self._fetch_historical_data()
        if bars is None or len(bars) == 0:
            print("❌ No historical data — is MT5 connected and the symbol correct?")
            return self.results

        print(f"📊 {len(bars):,} bars loaded  ({len(bars)/1440:.1f} trading days)")

        for i, bar in enumerate(bars):
            self._history.append(bar)

            # Check SL/TP on open trade first
            closed_this_bar = False
            if self._open_trade and not self._open_trade.is_closed():
                closed_this_bar = self._check_sl_tp(bar)

            signal = None if closed_this_bar or self._in_cooldown(bar) else self._generate_signal(i)
            if signal:
                self._process_signal(signal, bar)

            if i % 10000 == 0 and i > 0:
                print(f"  … {i:,}/{len(bars):,} bars  trades={len(self.trades)}  balance=${self.current_balance:,.2f}")

        # Close any open trade at last bar
        if self._open_trade and not self._open_trade.is_closed():
            last = bars[-1]
            ts   = self._bar_time(last)
            self._open_trade.close_trade(ts, float(last["close"]))
            self.current_balance += self._open_trade.profit_loss
            self._start_cooldown(ts)

        self._calculate_results()
        return self.results

    def run_walk_forward(self, train_ratio: float = 0.70) -> BacktestResults:
        """
        Walk-forward validation.

        The first slice is used only to warm indicators/calibration state. Trades
        are opened only on the later unseen slice, which gives a cleaner
        overfitting check than reporting performance on the whole history.
        """
        print(f"\nWalk-forward backtest {self.symbol} {self.start_date.date()} -> {self.end_date.date()}")
        bars = self._fetch_historical_data()
        if bars is None or len(bars) == 0:
            print("No historical data; is MT5 connected and the symbol correct?")
            return self.results

        split = int(len(bars) * max(0.1, min(0.9, train_ratio)))
        train_bars = bars[:split]
        test_bars = bars[split:]
        if len(test_bars) < self._min_lookback:
            print("Walk-forward test slice is too small.")
            return self.results

        print(
            f"Loaded {len(bars):,} bars | warm/train={len(train_bars):,} "
            f"unseen/test={len(test_bars):,}"
        )
        print(f"Hard cooldown: {self.trade_cooldown_minutes:.1f} min after each close")

        for bar in train_bars:
            self._history.append(bar)

        for i, bar in enumerate(test_bars, start=split):
            self._history.append(bar)
            closed_this_bar = False
            if self._open_trade and not self._open_trade.is_closed():
                closed_this_bar = self._check_sl_tp(bar)

            signal = None if closed_this_bar or self._in_cooldown(bar) else self._generate_signal(i)
            if signal:
                self._process_signal(signal, bar)

            if (i - split) % 10000 == 0 and i > split:
                print(
                    f"  ... {i - split:,}/{len(test_bars):,} unseen bars "
                    f"trades={len(self.trades)} balance=${self.current_balance:,.2f}"
                )

        if self._open_trade and not self._open_trade.is_closed():
            last = test_bars[-1]
            close_time = self._bar_time(last)
            self._open_trade.close_trade(close_time, float(last["close"]))
            self.current_balance += self._open_trade.profit_loss
            self._start_cooldown(close_time)

        self._calculate_results()
        return self.results

    # ── Data fetch ────────────────────────────────────────────────────────────

    def _fetch_historical_data(self):
        mt5 = _get_mt5()
        if mt5 is None:
            print("❌ MetaTrader5 package not available")
            return None

        if load_dotenv is not None:
            load_dotenv()

        login_raw = os.getenv("LOGIN")
        kwargs = {}
        if os.getenv("MT5_PATH"):
            kwargs["path"] = os.getenv("MT5_PATH")
        if login_raw:
            kwargs["login"] = int(login_raw)
        if os.getenv("SERVER"):
            kwargs["server"] = os.getenv("SERVER")
        if os.getenv("PASSWORD"):
            kwargs["password"] = os.getenv("PASSWORD")

        if not mt5.initialize(**kwargs):
            print(f"❌ MT5 init failed: {mt5.last_error()}")
            print("   Make sure MT5 terminal is running and logged in.")
            return None

        account = mt5.account_info()
        if account is not None:
            print(f"Connected MT5 account={account.login} server={account.server}")

        if not mt5.symbol_select(self.symbol, True):
            print(f"Could not select symbol {self.symbol}.")
            self._print_symbol_suggestions(mt5)
            mt5.shutdown()
            return None

        rates = self._copy_rates_with_fallback(mt5)

        if rates is None or len(rates) == 0:
            print(f"❌ No data for {self.symbol}. Check symbol name (try 'XAUUSDm' or 'XAUUSD').")
            self._print_symbol_suggestions(mt5)
            mt5.shutdown()
            return None

        mt5.shutdown()
        return rates

    def _copy_rates_with_fallback(self, mt5):
        """Fetch bars by date range, then fall back to latest-position history."""
        rates = mt5.copy_rates_range(self.symbol, self.timeframe, self.start_date, self.end_date)
        if rates is not None and len(rates) > 0:
            print(f"Fetched {len(rates):,} bars by date range.")
            return rates

        err = mt5.last_error()
        print(f"Date-range fetch returned no bars for {self.symbol}; MT5 last_error={err}.")

        chunked = self._copy_rates_range_chunked(mt5)
        if chunked is not None and len(chunked) > 0:
            print(f"Fetched {len(chunked):,} bars by chunked date range.")
            return chunked

        tf_seconds = self._timeframe_seconds()
        requested_seconds = max(1.0, (self.end_date - self.start_date).total_seconds())
        requested_bars = int(requested_seconds / tf_seconds) + self._min_lookback + 500
        request_count = max(2000, min(250000, requested_bars))

        rates = self._copy_rates_from_pos_chunked(mt5, request_count)
        if rates is None or len(rates) == 0:
            return rates

        start_ts = int(self.start_date.timestamp())
        end_ts = int(self.end_date.timestamp())
        filtered = rates[(rates["time"] >= start_ts) & (rates["time"] <= end_ts)]

        if len(filtered) > 0:
            print(
                f"Fallback fetched {len(rates):,} latest bars; "
                f"{len(filtered):,} are inside requested dates."
            )
            return filtered

        first = datetime.fromtimestamp(int(rates[0]["time"]), tz=timezone.utc).date()
        last = datetime.fromtimestamp(int(rates[-1]["time"]), tz=timezone.utc).date()
        print(
            f"Fallback fetched {len(rates):,} latest bars, but they cover {first} -> {last}, "
            f"not {self.start_date.date()} -> {self.end_date.date()}."
        )
        print("Using the latest available bars instead so you can still validate behavior.")
        return rates

    def _copy_rates_range_chunked(self, mt5):
        """Fetch date ranges in MT5-friendly chunks."""
        chunks = []
        cursor = self.start_date
        max_span = timedelta(days=29)
        while cursor < self.end_date:
            chunk_end = min(cursor + max_span, self.end_date)
            part = mt5.copy_rates_range(self.symbol, self.timeframe, cursor, chunk_end)
            if part is not None and len(part) > 0:
                chunks.append(part)
            cursor = chunk_end + timedelta(seconds=self._timeframe_seconds())

        return self._merge_rate_chunks(chunks)

    def _copy_rates_from_pos_chunked(self, mt5, total_count: int):
        """Fetch latest bars in chunks because large MT5 requests can fail."""
        chunks = []
        chunk_size = 50000
        start_pos = 0
        while start_pos < total_count:
            count = min(chunk_size, total_count - start_pos)
            part = mt5.copy_rates_from_pos(self.symbol, self.timeframe, start_pos, count)
            if part is None or len(part) == 0:
                print(
                    f"Latest-bars chunk stopped at start_pos={start_pos}, "
                    f"count={count}; MT5 last_error={mt5.last_error()}."
                )
                break
            chunks.append(part)
            if len(part) < count:
                break
            start_pos += count

        return self._merge_rate_chunks(chunks)

    @staticmethod
    def _merge_rate_chunks(chunks):
        if not chunks:
            return None
        merged = np.concatenate(chunks)
        _, idx = np.unique(merged["time"], return_index=True)
        merged = merged[np.sort(idx)]
        return np.sort(merged, order="time")

    def _timeframe_seconds(self) -> int:
        mapping = {
            1: 60,
            2: 120,
            3: 180,
            4: 240,
            5: 300,
            6: 360,
            10: 600,
            12: 720,
            15: 900,
            20: 1200,
            30: 1800,
            16385: 3600,
            16386: 7200,
            16387: 10800,
            16388: 14400,
            16390: 21600,
            16392: 28800,
            16396: 43200,
            16408: 86400,
        }
        return mapping.get(self.timeframe, 60)

    def _print_symbol_suggestions(self, mt5) -> None:
        """Print broker symbols that look like gold/XAU for quick correction."""
        try:
            symbols = mt5.symbols_get()
        except Exception:
            symbols = None
        if not symbols:
            return
        matches = [
            s.name for s in symbols
            if "XAU" in s.name.upper() or "GOLD" in s.name.upper()
        ][:20]
        if matches:
            print("Gold-like symbols available from this terminal:")
            print("  " + ", ".join(matches))

    # ── Signal generation ─────────────────────────────────────────────────────

    def _generate_signal(self, index: int) -> Optional[str]:
        if index < self._min_lookback:
            return None

        window = self._history[max(0, index - self._min_lookback): index + 1]
        try:
            closes = np.array([float(b["close"]) for b in window])
            highs  = np.array([float(b["high"])  for b in window])
            lows   = np.array([float(b["low"])   for b in window])
        except (KeyError, TypeError):
            return None

        n = len(closes)
        if n < 20:
            return None

        # ── Features ──────────────────────────────────────────────────
        momentum   = (closes[-1] - closes[-5])  / max(closes[-5], 1e-9)
        volatility = float(np.std(closes[-14:]))
        trend      = (closes[-1] - closes[-20]) / max(closes[-20], 1e-9)

        deltas   = np.diff(closes[-15:])
        gains    = np.where(deltas > 0, deltas, 0.0)
        losses   = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = float(np.mean(gains))  if gains.any()  else 1e-9
        avg_loss = float(np.mean(losses)) if losses.any() else 1e-9
        rsi      = 100 - (100 / (1 + avg_gain / avg_loss))

        short_pred = momentum * 1.0 + volatility * -0.1 + trend * 0.8 + (rsi - 50) * 0.02

        # Approximate mid prediction (5-bar horizon)
        if n >= 10:
            mid_trend = (closes[-1] - closes[-10]) / max(closes[-10], 1e-9)
            mid_pred  = mid_trend * 0.8 + momentum * 0.5
        else:
            mid_pred = short_pred

        # ── Confluence filter (mirrors v8 main.py logic) ──────────────
        mid_dominates = (
            (short_pred > 0 and mid_pred < 0 and abs(mid_pred) > abs(short_pred) * 3) or
            (short_pred < 0 and mid_pred > 0 and abs(mid_pred) > abs(short_pred) * 3)
        )
        m1_m15_agree = (
            (short_pred > 0 and mid_pred > 0) or
            (short_pred < 0 and mid_pred < 0)
        ) and not mid_dominates

        if not m1_m15_agree:
            return None

        # ── ATR trend gate ────────────────────────────────────────────
        trs = [
            max(highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i]  - closes[i - 1]))
            for i in range(max(1, n - 14), n)
        ]
        atr      = float(np.mean(trs)) if trs else volatility
        net_move = closes[-1] - closes[-6]

        if short_pred > 0:
            if net_move < -atr:
                return None
            if self._sim_direction == "SELL":
                return None
            return "BUY"
        elif short_pred < 0:
            if net_move > atr:
                return None
            if self._sim_direction == "BUY":
                return None
            return "SELL"

        return None

    # ── Trade execution ───────────────────────────────────────────────────────

    def _process_signal(self, signal: str, bar) -> None:
        entry_time  = self._bar_time(bar)
        entry_price = float(bar["close"])

        # Close existing trade if direction flipped
        if self._open_trade and not self._open_trade.is_closed():
            self._open_trade.close_trade(entry_time, entry_price)
            pnl = self._open_trade.profit_loss * (1 - self.commission_percent)
            self.current_balance += pnl
            self.equity_curve.append((entry_time, self.current_balance))
            self._start_cooldown(entry_time)
            return

        risk_amount = self.current_balance * (self.risk_percent / 100)
        lot = max(0.01, round(risk_amount / (self.sl_points * 1.0), 2))

        sl_dist = self.sl_points * 0.01   # approximate price distance
        tp_dist = sl_dist * self.tp_ratio
        if signal == "BUY":
            sl = entry_price - sl_dist
            tp = entry_price + tp_dist
        else:
            sl = entry_price + sl_dist
            tp = entry_price - tp_dist

        trade = Trade(
            entry_time=entry_time,
            entry_price=entry_price,
            direction=signal,
            volume=lot,
        )
        trade._sl = sl
        trade._tp = tp

        self.trades.append(trade)
        self._open_trade    = trade
        self._sim_direction = signal
        self.equity_curve.append((entry_time, self.current_balance))

    def _check_sl_tp(self, bar) -> bool:
        """Check if SL or TP was hit on the current bar."""
        t = self._open_trade
        if t is None or t.is_closed():
            return False

        sl = getattr(t, "_sl", None)
        tp = getattr(t, "_tp", None)
        if sl is None or tp is None:
            return False

        high  = float(bar["high"])
        low   = float(bar["low"])
        btime = self._bar_time(bar)

        if t.direction == "BUY":
            if low <= sl:
                t.close_trade(btime, sl)
                pnl = t.profit_loss * (1 - self.commission_percent)
                self.current_balance += pnl
                self.equity_curve.append((btime, self.current_balance))
                self._sim_direction = None
                self._start_cooldown(btime)
                return True
            elif high >= tp:
                t.close_trade(btime, tp)
                pnl = t.profit_loss * (1 - self.commission_percent)
                self.current_balance += pnl
                self.equity_curve.append((btime, self.current_balance))
                self._sim_direction = None
                self._start_cooldown(btime)
                return True
        else:
            if high >= sl:
                t.close_trade(btime, sl)
                pnl = t.profit_loss * (1 - self.commission_percent)
                self.current_balance += pnl
                self.equity_curve.append((btime, self.current_balance))
                self._sim_direction = None
                self._start_cooldown(btime)
                return True
            elif low <= tp:
                t.close_trade(btime, tp)
                pnl = t.profit_loss * (1 - self.commission_percent)
                self.current_balance += pnl
                self.equity_curve.append((btime, self.current_balance))
                self._sim_direction = None
                self._start_cooldown(btime)
                return True

        return False

    def _start_cooldown(self, close_time: datetime) -> None:
        if self.trade_cooldown_minutes <= 0:
            self._cooldown_until = None
            return
        self._cooldown_until = close_time + timedelta(minutes=self.trade_cooldown_minutes)

    def _in_cooldown(self, bar) -> bool:
        if self._cooldown_until is None:
            return False
        return self._bar_time(bar) < self._cooldown_until

    # ── Metrics ───────────────────────────────────────────────────────────────

    def _calculate_results(self) -> None:
        closed = [t for t in self.trades if t.is_closed()]
        if not closed:
            print("⚠️  No closed trades during backtest period")
            return

        r = self.results
        r.total_trades   = len(closed)
        r.winning_trades = sum(1 for t in closed if t.profit_loss > 0)
        r.losing_trades  = sum(1 for t in closed if t.profit_loss <= 0)
        r.win_rate       = r.winning_trades / r.total_trades * 100
        r.final_balance  = self.current_balance

        pnls = [t.profit_loss for t in closed]
        r.total_profit_loss = sum(pnls)
        r.gross_profit = sum(p for p in pnls if p > 0)
        r.gross_loss   = abs(sum(p for p in pnls if p < 0))
        r.profit_factor = r.gross_profit / r.gross_loss if r.gross_loss else float("inf")

        r.avg_trade_profit   = r.total_profit_loss / r.total_trades
        r.avg_winning_trade  = r.gross_profit / r.winning_trades if r.winning_trades else 0
        r.avg_losing_trade   = r.gross_loss   / r.losing_trades  if r.losing_trades  else 0

        # Sharpe (annualised, assumes M1 bars)
        if len(pnls) > 1:
            mean_r = float(np.mean(pnls))
            std_r  = float(np.std(pnls))
            bars_per_year = 525600   # M1
            r.sharpe_ratio = (mean_r / std_r * np.sqrt(bars_per_year)) if std_r > 0 else 0.0

        # Max drawdown
        if self.equity_curve:
            equities = [e for _, e in self.equity_curve]
            peak = equities[0]
            max_dd = 0.0
            for eq in equities:
                peak  = max(peak, eq)
                dd    = (peak - eq) / peak * 100 if peak > 0 else 0
                max_dd = max(max_dd, dd)
            r.max_drawdown_pct = max_dd

        # Consecutive losses
        streak = max_streak = 0
        for t in closed:
            if t.profit_loss < 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        r.max_consecutive_losses = max_streak

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _bar_time(bar) -> datetime:
        """Convert MT5 bar time (Unix int) to datetime."""
        t = bar["time"]
        if isinstance(t, (int, float, np.integer)):
            return datetime.fromtimestamp(int(t), tz=timezone.utc).replace(tzinfo=None)
        return t

    def save_results(self, filename: str = "backtest_results.csv") -> None:
        with open(filename, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Entry Time","Entry Price","Exit Time","Exit Price",
                        "Direction","Volume","Profit/Loss","Profit %","Duration (min)"])
            for t in self.trades:
                if t.is_closed():
                    w.writerow([t.entry_time, f"{t.entry_price:.5f}",
                                t.exit_time,  f"{t.exit_price:.5f}",
                                t.direction, t.volume,
                                f"{t.profit_loss:.2f}", f"{t.profit_pct:.4f}",
                                f"{t.duration_minutes:.1f}"])
        print(f"Results saved -> {filename}")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest the XAUUSDm strategy")
    parser.add_argument("--symbol",   default="XAUUSDm",    help="Symbol (default: XAUUSDm)")
    parser.add_argument("--days",     type=int, default=30,  help="How many days back (default: 30)")
    parser.add_argument("--balance",  type=float, default=10000, help="Starting balance (default: 10000)")
    parser.add_argument("--risk",     type=float, default=5.0,   help="Risk %% per trade (default: 5)")
    parser.add_argument("--sl",       type=int,   default=150,   help="SL in points (default: 150)")
    parser.add_argument("--walk-forward", action="store_true", help="Trade only the unseen future split")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Warm/train split for walk-forward")
    parser.add_argument("--cooldown-minutes", type=float, default=5.0, help="Hard cooldown after each close")
    args = parser.parse_args()

    end   = datetime.now()
    start = end - timedelta(days=args.days)

    bt = Backtester(
        symbol=args.symbol,
        timeframe=1,   # M1
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
        initial_balance=args.balance,
        risk_percent=args.risk,
        sl_points=args.sl,
        trade_cooldown_minutes=args.cooldown_minutes,
    )

    results = bt.run_walk_forward(args.train_ratio) if args.walk_forward else bt.run()
    results.print_summary()
    bt.save_results()
