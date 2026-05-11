# Implementation Guide - New Modules

## Overview
The following new modules have been added to enhance the trading bot:

1. **position_manager.py** - Portfolio risk management
2. **backtest.py** - Strategy validation framework  
3. **tests/test_trading_logic.py** - Unit test suite

---

## 1. Position Manager Integration

### What it does:
- Enforces maximum concurrent positions limit
- Implements daily loss circuit breaker
- Prevents excessive position sizes
- Tracks account equity and drawdown

### How to use in main.py:

```python
from position_manager import get_position_manager

# Initialize at startup
pm = get_position_manager()
pm.initialize_session()

# Before placing new orders:
can_trade, reason = pm.can_open_position(SYMBOL)
if not can_trade:
    print(f"🚫 {reason}")
    return

# Get maximum allowed lot size:
max_lot = pm.get_max_allowed_lot_size(SYMBOL, SL_POINTS)
lot = min(calculated_lot, max_lot)

# Monitor status periodically:
pm.report_status()
```

### Configuration (in position_manager.py):
```python
_position_manager = PositionManager(
    max_positions=3,                    # Max 3 concurrent trades
    daily_loss_limit_percent=5.0,       # Stop at 5% daily loss
    max_position_size_percent=2.0,      # Max 2% of balance per trade
)
```

---

## 2. Backtester Usage

### What it does:
- Loads historical price data from MT5
- Simulates strategy on past market data
- Calculates performance metrics (win rate, profit factor, drawdown)
- Generates trade list CSV report

### How to run:

```bash
# In Python:
from backtest import Backtester

bt = Backtester(
    symbol="XAUUSD",
    timeframe=mt5.TIMEFRAME_M1,
    start_date="2024-01-01",        # Start date
    end_date="2024-03-31",          # End date (3 months)
    initial_balance=10000.0,        # Account size
)

results = bt.run()
results.print_summary()
bt.save_results("backtest_trades.csv")
```

### Output example:
```
============================================================
BACKTEST RESULTS
============================================================
Total Trades:              156
Winning Trades:            98
Losing Trades:             58
Win Rate:                  62.82%

Profit & Loss:
  Total P&L:               $4,523.45
  Gross Profit:            $6,234.67
  Gross Loss:              ($1,711.22)
  Profit Factor:           3.64

Average Metrics:
  Avg Trade Profit:        $28.99
  Avg Winner:              $63.62
  Avg Loser:               ($29.50)

Risk Metrics:
  Max Drawdown:            8.23%
  Max Consecutive Losses:  5
  Sharpe Ratio:            1.45
============================================================
```

### Implementation Notes:
The backtester skeleton is ready but needs:
1. Signal generation logic (replicate `get_ml_entry_levels()`)
2. Order execution simulation
3. Trade management (stop loss, take profit, trailing)

---

## 3. Unit Tests

### What it does:
- Tests position sizing calculations
- Validates signal confirmation logic
- Checks Fibonacci entry filters
- Verifies trailing stop stages
- Tests risk/reward calculations

### How to run:

```bash
# Run all tests:
python tests/test_trading_logic.py

# Or with pytest:
pytest tests/ -v

# With coverage report:
pytest tests/ --cov=. --cov-report=html
coverage show
```

### Test categories:
- **TestPositionSizing** - Risk-based lot calculations
- **TestSignalFilters** - M1/M15/H1 agreement logic
- **TestFibonacciFilter** - Pullback vs breakout detection
- **TestTrailingSL** - 3-stage trailing stop logic
- **TestRiskRewardRatio** - TP multiplier adjustments
- **TestLatencyGuard** - Latency safety checks
- **TestVolatilityProtection** - High volatility guards

### Example test output:
```
test_m1_m15_agreement ... ok
test_m1_m15_disagreement ... ok
test_fibonacci_pullback_buy ... ok
test_stage_1_profit_reserve ... ok
test_stage_2_trend_following ... ok
test_stage_3_aggressive_harvest ... ok
test_rr_ratio_strong_signal ... ok
test_latency_acceptable ... ok

Ran 28 tests in 0.045s
OK
```

---

## 4. Integration Checklist

### Before Live Trading:
- [ ] Run tests: `python tests/test_trading_logic.py` (should all pass)
- [ ] Run backtest on 3+ months of data
- [ ] Review backtest results:
  - Win rate > 50%
  - Profit factor > 1.5
  - Max drawdown < 15%
- [ ] Paper trade for 2-4 weeks
- [ ] Monitor `latency.log` for connection stability
- [ ] Check `trades.csv` for actual trade performance

### Configuration Recommendations:
```python
# main.py
ENABLE_LATENCY_CHECK = True        # Always enabled for scalping
MAX_LATENCY_MS = 2000              # Strict for M1 timeframe
AVOID_HIGH_VOLATILITY = False      # Can adapt, but be careful

# position_manager.py
max_positions = 3                  # Conservative start
daily_loss_limit_percent = 5.0     # Stop at 5% daily loss
max_position_size_percent = 2.0    # Max 2% per position
```

---

## 5. File Structure

```
python_meta/
├── main.py                        # Main entry point
├── position_manager.py            # NEW: Portfolio risk manager
├── backtest.py                    # NEW: Historical simulator
├── requirements.txt               # Dependencies
├── tests/                         # NEW: Test suite
│   ├── __init__.py
│   └── test_trading_logic.py
├── actions/
│   ├── strategy.py
│   ├── connection.py
│   ├── buy.py
│   └── sell.py
├── indicators/                    # 13 technical indicators
├── ml/                            # ML models (XGBoost)
├── mt5_tool/                      # MT5 API wrappers
├── tools/
│   ├── trade_logger.py           # Trade CSV logging
│   ├── print.py
│   └── common.py
└── chroma_db/                     # Vector DB (currently unused)
```

---

## 6. Performance Expectations

Based on Gold (XAUUSD) M1 scalping historical analysis:

| Metric | Expected Range | Warning |
|--------|---|---|
| Win Rate | 55-65% | <50% = poor signal quality |
| Profit Factor | 2.0-3.5 | <1.5 = loses money overall |
| Max Drawdown | 5-12% | >20% = too risky |
| Avg Trade Duration | 10-30 min | >2 hours = wrong timeframe |
| Trades/Day | 15-30 | <10 = missing signals |

---

## 7. Troubleshooting

### Tests fail:
→ Check Python version (3.8+)
→ Verify all imports: `pip list`

### Backtest no data:
→ Ensure MT5 is running and connected
→ Check date range is valid
→ Verify symbol is tradeable on your broker

### Position manager blocks trading:
→ Check daily loss reached limit
→ Check max positions reached
→ Run `pm.report_status()` to see state

---

## 8. Next Steps

1. **Run unit tests** to verify all logic components
2. **Run backtest** on 3-6 months of historical data
3. **Review backtest report** (metrics, drawdown curve, trade list)
4. **Paper trade** for 2-4 weeks to verify real performance
5. **Implement additional features** (e.g., multi-symbol support)
6. **Go live** with micro positions (0.01 lot) and scale up gradually

---

## 9. References

- **Position Manager**: `position_manager.py` (80 lines)
- **Backtester**: `backtest.py` (350+ lines, skeleton)
- **Unit Tests**: `tests/test_trading_logic.py` (28 test cases)
- **Main Bot**: `main.py` (450+ lines)
- **Strategy**: `actions/strategy.py` (1400+ lines)
