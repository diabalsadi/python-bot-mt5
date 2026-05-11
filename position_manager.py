"""
Position Manager
----------------
Enforces portfolio-level risk limits to prevent excessive drawdowns
and position concentration.

Features:
  - Max concurrent positions limit
  - Daily loss hard-stop (circuit breaker)
  - Max position size per symbol
  - Drawdown tracking
  - Account equity monitoring
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Optional

import MetaTrader5 as mt5


class PositionManager:
    """
    Portfolio risk management layer.
    
    Prevents:
      - Too many concurrent positions
      - Excessive daily losses
      - Leverage runaway (position size too large)
    """
    
    def __init__(
        self,
        max_positions: int = 3,
        daily_loss_limit_percent: float = 5.0,
        max_position_size_percent: float = 2.0,
        check_interval_seconds: float = 1.0,
    ):
        """
        Initialize position manager.
        
        Args:
            max_positions: Max concurrent open positions (default: 3)
            daily_loss_limit_percent: Stop trading if daily loss > this % (default: 5%)
            max_position_size_percent: Max position size as % of balance (default: 2%)
            check_interval_seconds: How often to check limits (default: 1s)
        """
        self.max_positions = max_positions
        self.daily_loss_limit_percent = daily_loss_limit_percent
        self.max_position_size_percent = max_position_size_percent
        self.check_interval_seconds = check_interval_seconds
        
        # State tracking
        self._last_check_ts = 0.0
        self._session_start_balance = 0.0
        self._session_start_time = datetime.now()
        self._trading_disabled = False
        self._disable_reason = ""
        
    def initialize_session(self) -> bool:
        """
        Start a new trading session. Record opening balance.
        
        Returns:
            True if session initialized successfully
        """
        account = mt5.account_info()
        if account is None:
            print("❌ PositionManager: Cannot get account info")
            return False
        
        self._session_start_balance = account.balance
        self._session_start_time = datetime.now()
        self._trading_disabled = False
        self._disable_reason = ""
        
        print(f"✅ Session started | Balance: ${account.balance:,.2f}")
        return True
    
    def can_open_position(self, symbol: Optional[str] = None) -> tuple[bool, str]:
        """
        Check if a new position can be opened.
        
        Args:
            symbol: Symbol to check (optional)
            
        Returns:
            (allowed: bool, reason: str)
            - (True, ""): OK to trade
            - (False, reason): Trading blocked with explanation
        """
        now = time.time()
        
        # Throttle checks
        if now - self._last_check_ts < self.check_interval_seconds:
            return (not self._trading_disabled, self._disable_reason)
        
        self._last_check_ts = now
        
        # Check if trading is disabled
        if self._trading_disabled:
            return (False, f"🚫 Trading disabled: {self._disable_reason}")
        
        # Check position count
        positions = mt5.positions_get()
        if positions is None:
            positions = []
        
        if len(positions) >= self.max_positions:
            msg = f"Max positions ({self.max_positions}) reached"
            return (False, msg)
        
        # Check daily loss
        account = mt5.account_info()
        if account is None:
            return (False, "Cannot get account info")
        
        daily_loss = self._session_start_balance - account.balance
        daily_loss_pct = (daily_loss / self._session_start_balance * 100) if self._session_start_balance > 0 else 0
        
        if daily_loss_pct > self.daily_loss_limit_percent:
            msg = f"Daily loss limit ({self.daily_loss_limit_percent}%) exceeded: {daily_loss_pct:.2f}%"
            self._trading_disabled = True
            self._disable_reason = msg
            print(f"🚫 {msg}")
            return (False, msg)
        
        return (True, "")
    
    def get_position_count(self) -> int:
        """Get current number of open positions."""
        positions = mt5.positions_get()
        return len(positions) if positions else 0
    
    def get_daily_loss_percent(self) -> float:
        """Get current daily loss as percentage of starting balance."""
        account = mt5.account_info()
        if account is None or self._session_start_balance <= 0:
            return 0.0
        
        loss = self._session_start_balance - account.balance
        return (loss / self._session_start_balance) * 100
    
    def get_session_duration_minutes(self) -> float:
        """Get session duration in minutes."""
        elapsed = datetime.now() - self._session_start_time
        return elapsed.total_seconds() / 60
    
    def get_max_allowed_lot_size(self, symbol: str, sl_points: int) -> float:
        """
        Calculate maximum lot size based on % of balance.
        
        Args:
            symbol: Trading symbol
            sl_points: Stop loss in points
            
        Returns:
            Max lot size respecting position size limit
        """
        sym = mt5.symbol_info(symbol)
        account = mt5.account_info()
        
        if sym is None or account is None:
            return 0.01
        
        max_risk_amount = account.balance * (self.max_position_size_percent / 100)
        
        tick_value = sym.trade_tick_value
        tick_size = sym.trade_tick_size
        
        if tick_value <= 0 or tick_size <= 0:
            return sym.volume_min if sym else 0.01
        
        max_lot = max_risk_amount / (sl_points * (tick_value / tick_size))
        
        # Clamp to broker limits
        max_lot = max(sym.volume_min, min(sym.volume_max, max_lot))
        
        return round(max_lot, 2)
    
    def report_status(self) -> None:
        """Print current session status."""
        account = mt5.account_info()
        if account is None:
            print("❌ Cannot get account info")
            return
        
        positions = self.get_position_count()
        daily_loss_pct = self.get_daily_loss_percent()
        session_mins = self.get_session_duration_minutes()
        
        status = "🟢 TRADING" if not self._trading_disabled else "🔴 DISABLED"
        
        print(
            f"{status} | Positions: {positions}/{self.max_positions} | "
            f"Daily Loss: {daily_loss_pct:.2f}% ({self.daily_loss_limit_percent}% limit) | "
            f"Balance: ${account.balance:,.2f} | Session: {session_mins:.1f} min"
        )


# Global instance
_position_manager = PositionManager()


def get_position_manager() -> PositionManager:
    """Get the global position manager instance."""
    return _position_manager
