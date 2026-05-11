"""
Backtester for Strategy Validation
-----------------------------------
Historical simulation framework for testing strategy performance.

Example usage:
    from backtest import Backtester
    
    bt = Backtester(
        symbol="XAUUSD",
        timeframe=mt5.TIMEFRAME_M1,
        start_date="2024-01-01",
        end_date="2024-03-31",
        initial_balance=10000,
    )
    
    results = bt.run()
    bt.print_report()
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
import csv
import os

import MetaTrader5 as mt5
import numpy as np


@dataclass
class Trade:
    """Single trade record."""
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    direction: str = "BUY"  # BUY or SELL
    volume: float = 0.01
    profit_loss: float = 0.0
    profit_pct: float = 0.0
    duration_minutes: float = 0.0
    
    def close_trade(self, exit_time: datetime, exit_price: float) -> None:
        """Close a trade."""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.duration_minutes = (exit_time - self.entry_time).total_seconds() / 60
        
        if self.direction == "BUY":
            self.profit_loss = (exit_price - self.entry_price) * self.volume * 100
            self.profit_pct = ((exit_price - self.entry_price) / self.entry_price) * 100
        else:
            self.profit_loss = (self.entry_price - exit_price) * self.volume * 100
            self.profit_pct = ((self.entry_price - exit_price) / self.entry_price) * 100
    
    def is_closed(self) -> bool:
        """Check if trade is closed."""
        return self.exit_time is not None


@dataclass
class BacktestResults:
    """Backtest performance summary."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_profit_loss: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    avg_trade_profit: float = 0.0
    avg_winning_trade: float = 0.0
    avg_losing_trade: float = 0.0
    max_drawdown_pct: float = 0.0
    max_consecutive_losses: int = 0
    sharpe_ratio: float = 0.0
    
    def print_summary(self) -> None:
        """Print results summary."""
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
        print(f"\nAverage Metrics:")
        print(f"  Avg Trade Profit:        ${self.avg_trade_profit:,.2f}")
        print(f"  Avg Winner:              ${self.avg_winning_trade:,.2f}")
        print(f"  Avg Loser:               ${self.avg_losing_trade:,.2f}")
        print(f"\nRisk Metrics:")
        print(f"  Max Drawdown:            {self.max_drawdown_pct:.2f}%")
        print(f"  Max Consecutive Losses:  {self.max_consecutive_losses}")
        print(f"  Sharpe Ratio:            {self.sharpe_ratio:.2f}")
        print("="*60 + "\n")


class Backtester:
    """
    Backtester for strategy validation on historical data.
    
    Note: This is a skeleton framework. Actual implementation would:
    - Fetch historical candle data from MT5
    - Simulate strategy signals on each bar
    - Track opening/closing of trades
    - Calculate performance metrics
    """
    
    def __init__(
        self,
        symbol: str = "XAUUSD",
        timeframe: int = mt5.TIMEFRAME_M1,
        start_date: str = "2024-01-01",
        end_date: str = "2024-03-31",
        initial_balance: float = 10000.0,
        commission_percent: float = 0.0005,
    ):
        """
        Initialize backtester.
        
        Args:
            symbol: Trading symbol (default: XAUUSD)
            timeframe: MT5 timeframe (default: M1)
            start_date: Start date as YYYY-MM-DD
            end_date: End date as YYYY-MM-DD
            initial_balance: Starting account balance (default: $10,000)
            commission_percent: Commission as % of trade volume (default: 0.05%)
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.initial_balance = initial_balance
        self.commission_percent = commission_percent
        
        # Results
        self.trades: List[Trade] = []
        self.results = BacktestResults()
        self.equity_curve: List[Tuple[datetime, float]] = []
        
        # Current state
        self.current_balance = initial_balance
        self.current_equity = initial_balance
    
    def run(self) -> BacktestResults:
        """
        Run backtest on historical data.
        
        Returns:
            BacktestResults with performance metrics
        """
        print(f"🔄 Backtesting {self.symbol} from {self.start_date.date()} to {self.end_date.date()}")
        print(f"   Timeframe: {self.timeframe}, Initial Balance: ${self.initial_balance:,.2f}")
        
        # Fetch historical data
        bars = self._fetch_historical_data()
        
        if bars is None or bars.size == 0:
            print("❌ No historical data available")
            return self.results
        
        print(f"📊 Loaded {bars.size} bars")
        
        # Simulate strategy on each bar
        for i, bar in enumerate(bars):
            # TODO: Implement signal generation on bar
            # signal = self._generate_signal(bar, i)
            # if signal:
            #     self._process_signal(signal, bar)
            pass
        
        # Calculate metrics
        self._calculate_results()
        
        return self.results
    
    def _fetch_historical_data(self) -> np.ndarray:
        """
        Fetch historical bars from MT5.
        
        Returns:
            Array of OHLCV candles
        """
        # Calculate number of bars needed
        delta = self.end_date - self.start_date
        
        # Rough estimate: M1 = 60*24 bars per day
        if self.timeframe == mt5.TIMEFRAME_M1:
            bars_per_day = 1440
        elif self.timeframe == mt5.TIMEFRAME_M5:
            bars_per_day = 288
        elif self.timeframe == mt5.TIMEFRAME_M15:
            bars_per_day = 96
        elif self.timeframe == mt5.TIMEFRAME_H1:
            bars_per_day = 24
        else:
            bars_per_day = 24
        
        approx_bars = delta.days * bars_per_day + 500  # Extra buffer
        
        # Fetch from MT5
        try:
            rates = mt5.copy_rates_range(
                self.symbol,
                self.timeframe,
                self.start_date,
                self.end_date
            )
            return rates if rates is not None else np.array([])
        except Exception as e:
            print(f"❌ Error fetching historical data: {e}")
            return np.array([])
    
    def _generate_signal(self, bar: dict, index: int) -> Optional[str]:
        """
        Generate trading signal for current bar.
        
        Args:
            bar: OHLCV candle data
            index: Bar index in history
            
        Returns:
            "BUY", "SELL", or None
        """
        # TODO: Implement actual signal logic
        # This would replicate on_tick() logic with historical data
        return None
    
    def _process_signal(self, signal: str, bar: dict) -> None:
        """
        Process trade signal.
        
        Args:
            signal: "BUY" or "SELL"
            bar: OHLCV data
        """
        # TODO: Simulate order execution and management
        pass
    
    def _calculate_results(self) -> None:
        """Calculate backtest performance metrics."""
        if not self.trades:
            print("⚠️  No trades executed during backtest")
            return
        
        closed_trades = [t for t in self.trades if t.is_closed()]
        
        if not closed_trades:
            print("⚠️  No closed trades")
            return
        
        # Basic metrics
        self.results.total_trades = len(closed_trades)
        self.results.winning_trades = len([t for t in closed_trades if t.profit_loss > 0])
        self.results.losing_trades = len([t for t in closed_trades if t.profit_loss < 0])
        
        if self.results.total_trades > 0:
            self.results.win_rate = (self.results.winning_trades / self.results.total_trades) * 100
        
        # P&L metrics
        self.results.total_profit_loss = sum(t.profit_loss for t in closed_trades)
        self.results.gross_profit = sum(t.profit_loss for t in closed_trades if t.profit_loss > 0)
        self.results.gross_loss = abs(sum(t.profit_loss for t in closed_trades if t.profit_loss < 0))
        
        if self.results.gross_loss > 0:
            self.results.profit_factor = self.results.gross_profit / self.results.gross_loss
        
        if self.results.total_trades > 0:
            self.results.avg_trade_profit = self.results.total_profit_loss / self.results.total_trades
        
        if self.results.winning_trades > 0:
            self.results.avg_winning_trade = self.results.gross_profit / self.results.winning_trades
        
        if self.results.losing_trades > 0:
            self.results.avg_losing_trade = self.results.gross_loss / self.results.losing_trades
        
        # Drawdown
        self._calculate_drawdown()
        
        # Consecutive losses
        consecutive_losses = 0
        max_consecutive = 0
        for trade in closed_trades:
            if trade.profit_loss < 0:
                consecutive_losses += 1
                max_consecutive = max(max_consecutive, consecutive_losses)
            else:
                consecutive_losses = 0
        
        self.results.max_consecutive_losses = max_consecutive
    
    def _calculate_drawdown(self) -> None:
        """Calculate maximum drawdown percentage."""
        if not self.equity_curve or len(self.equity_curve) < 2:
            return
        
        equities = [eq for _, eq in self.equity_curve]
        peak = equities[0]
        max_dd = 0.0
        
        for eq in equities:
            if eq > peak:
                peak = eq
            
            dd = ((peak - eq) / peak) * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        self.results.max_drawdown_pct = max_dd
    
    def save_results(self, filename: str = "backtest_results.csv") -> None:
        """
        Save trade list to CSV.
        
        Args:
            filename: Output CSV filename
        """
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Entry Time', 'Entry Price', 'Exit Time', 'Exit Price',
                'Direction', 'Volume', 'Profit/Loss', 'Profit %', 'Duration (min)'
            ])
            
            for trade in self.trades:
                if trade.is_closed():
                    writer.writerow([
                        trade.entry_time,
                        f"{trade.entry_price:.5f}",
                        trade.exit_time,
                        f"{trade.exit_price:.5f}",
                        trade.direction,
                        trade.volume,
                        f"{trade.profit_loss:.2f}",
                        f"{trade.profit_pct:.4f}",
                        f"{trade.duration_minutes:.1f}",
                    ])
        
        print(f"✅ Results saved to {filename}")


# Example usage
if __name__ == "__main__":
    bt = Backtester(
        symbol="XAUUSD",
        timeframe=mt5.TIMEFRAME_M1,
        start_date="2024-01-01",
        end_date="2024-03-31",
        initial_balance=10000,
    )
    
    results = bt.run()
    results.print_summary()
    bt.save_results()
