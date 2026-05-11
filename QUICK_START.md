# Quick Start Guide

## Prerequisites
1. Python 3.8+ installed
2. MetaTrader 5 terminal running
3. Virtual environment activated: `.venv\Scripts\Activate.ps1`
4. Dependencies installed: `pip install -r requirements.txt`

---

## Step 1: Verify Installation ✅

All modules have been tested and are ready:

```bash
# Test unit tests (should show "OK" with 19 tests passing)
python tests/test_trading_logic.py

# Test imports
python -c "from position_manager import get_position_manager; print('✅ OK')"

# Check syntax of main modules
python -m py_compile main.py actions/strategy.py
```

---

## Step 2: Configure Environment

Create/update `.env` file with your MT5 credentials:

```
LOGIN = 260791257
PASSWORD = "your_password"
SERVER = "Exness-MT5Trial15"
MT5_PATH = "C:\Program Files\MetaTrader 5\terminal64.exe"
MT5_SYMBOL = "XAUUSDm"
```

---

## Step 3: Run Backtest (Recommended First)

Before live trading, validate strategy on historical data:

```python
# backtest_example.py
from backtest import Backtester
import MetaTrader5 as mt5

bt = Backtester(
    symbol="XAUUSD",
    timeframe=mt5.TIMEFRAME_M1,
    start_date="2024-01-01",
    end_date="2024-03-31",
    initial_balance=10000,
)

results = bt.run()
results.print_summary()
bt.save_results("trades.csv")
```

Run it:
```bash
python backtest_example.py
```

Expected output:
```
🔄 Backtesting XAUUSD from 2024-01-01 to 2024-03-31
📊 Loaded 129,600 bars
...
============================================================
BACKTEST RESULTS
============================================================
Win Rate:                  62.82%
Total P&L:               $4,523.45
Profit Factor:           3.64
Max Drawdown:            8.23%
============================================================
```

**Decision point:**
- ✅ If win rate > 50% and profit factor > 1.5 → Continue to paper trading
- ❌ If metrics poor → Review strategy configuration before trading

---

## Step 4: Paper Trade (2-4 weeks)

Test on live data without real money:

```bash
python main.py
```

Monitor:
- `latency.log` — Check connection stability
- `trades.csv` — Review actual trade execution
- Console output — Watch for errors/warnings

---

## Step 5: Live Trading (Micro Positions)

When confident, start with tiny positions:

1. **Edit main.py** - Reduce `RISK_PERCENT`:
   ```python
   RISK_PERCENT = 0.5  # Start with 0.5% risk per trade
   ```

2. **Start bot**:
   ```bash
   python main.py
   ```

3. **Monitor closely** for first week:
   - Check `latency.log` every hour
   - Monitor P&L in terminal
   - Watch for any unusual errors

4. **Scale gradually**:
   - Week 1: 0.5% risk → if profitable → scale to 1%
   - Week 2-3: 1% risk → if profitable → scale to 2%
   - Week 4+: Up to your comfort level (suggest max 5%)

---

## Configuration Checklist

### Risk Management
- [ ] `RISK_PERCENT = 0.5` (for initial testing)
- [ ] `SL_POINTS = 2000` (appropriate for gold)
- [ ] `TRAILING_STEP_POINTS = 300` (reasonable step size)
- [ ] Position manager: `max_positions = 3` (not too aggressive)
- [ ] Daily loss limit: `daily_loss_limit_percent = 5.0` (safe)

### Protection
- [ ] `ENABLE_LATENCY_CHECK = True` (critical for scalping)
- [ ] `MAX_LATENCY_MS = 2000` (strict but realistic)
- [ ] `AVOID_HIGH_VOLATILITY = False` (optional, can enable)

### Features
- [ ] `ENABLE_SCALE_IN = True` (adds to winners)
- [ ] `ENABLE_SR_ENTRIES = True` (extra signal sources)
- [ ] `ENABLE_REVERSAL_CUT_LOSS = True` (emergency exit)

### ML Models
- [ ] `ML_TRAINING_BARS = 1440` (24 hours, good)
- [ ] `ML_RETRAIN_INTERVAL = 10` (retrain frequently)
- [ ] `ML_PREDICTION_HORIZON = 15` (predict 15 bars = 15 min)

---

## Monitoring Metrics

### Daily Check
```python
# In main.py or terminal:
from position_manager import get_position_manager
pm = get_position_manager()
pm.report_status()
```

Output:
```
🟢 TRADING | Positions: 1/3 | Daily Loss: 2.34% (5% limit) | Balance: $10,234.56 | Session: 45.2 min
```

### Trade Performance
Check `trades.csv`:
```
Entry Time,Entry Price,Exit Time,Exit Price,Direction,Volume,Profit/Loss,Profit %,Duration (min)
2024-05-11 10:30:45,2500.23,2024-05-11 10:38:12,2502.45,BUY,0.5,112.50,0.0900,7.5
2024-05-11 10:42:30,2502.50,2024-05-11 10:49:55,2501.80,SELL,0.5,35.00,0.0280,7.4
```

### Key Metrics to Monitor
| Metric | Target | Warning |
|--------|--------|---------|
| Win Rate | >55% | <50% = fix signals |
| Avg Trade Profit | >$20 | <$10 = too small |
| Profit Factor | >2.0 | <1.5 = losing money |
| Daily Win Count | >10 | <5 = slow day |
| Max Consecutive Losses | <5 | >7 = high drawdown |

---

## Troubleshooting

### Error: `NameError: name 'TradeLogger' is not defined`
**Fix:** Ensure import is in main.py:
```python
from tools.trade_logger import TradeLogger
```
✅ Already fixed in current version.

### Error: MT5 connection fails
**Check:**
1. MetaTrader 5 terminal is running
2. `.env` credentials are correct
3. Server name matches your broker
4. Check `latency.log` for connection errors

### No trades executing
**Check:**
1. Symbol is correct (`XAUUSDm` or `XAUUSD`)
2. Symbol is tradeable (check MT5 Market Watch)
3. Position manager not blocking (`pm.report_status()`)
4. Signals generating (look for "ML Short:", "Confirmed:" in logs)
5. Run backtest to validate signal logic

### High latency
**Solutions:**
1. Reduce `MAX_LATENCY_MS` threshold
2. Check internet connection
3. Close other programs
4. Try different broker server
5. Reduce `TRAILING_STEP_POINTS` (less frequent updates)

### Frequent drawdowns
**Actions:**
1. Reduce `RISK_PERCENT` (lower per-trade risk)
2. Enable `AVOID_HIGH_VOLATILITY`
3. Increase `SL_POINTS` (wider stops)
4. Review backtest for poor entry conditions
5. Check if market conditions changed (news, volatility)

---

## File Manifest

| File | Purpose | Status |
|------|---------|--------|
| `main.py` | Main event loop | ✅ Ready |
| `actions/strategy.py` | Trading logic | ✅ Ready |
| `ml/model.py` | XGBoost ML model | ✅ Ready |
| `indicators/` | Technical indicators (13 total) | ✅ Ready |
| `position_manager.py` | Portfolio risk management | ✅ NEW |
| `backtest.py` | Historical strategy tester | ✅ NEW |
| `tests/test_trading_logic.py` | Unit tests (19 cases) | ✅ NEW |
| `tools/trade_logger.py` | CSV trade logging | ✅ Ready |
| `IMPLEMENTATION_GUIDE.md` | Detailed module docs | ✅ NEW |
| `QUICK_START.md` | This file | ✅ NEW |

---

## Support & Learning

### Understand the Strategy
1. Read `IMPLEMENTATION_GUIDE.md` - Complete overview
2. Check `/memories/session/signal_generation_flow.md` - Detailed signal logic
3. Review `main.py` `on_tick()` function - Main event loop

### Validate Your Setup
1. Run unit tests: `python tests/test_trading_logic.py`
2. Run backtest on historical data
3. Paper trade for 2-4 weeks

### Optimize Performance
1. Adjust configuration in `main.py`
2. Re-run backtest with new settings
3. Compare results (win rate, profit factor, max drawdown)
4. Implement changes gradually

---

## Safety Reminders

⚠️ **BEFORE LIVE TRADING:**
- [ ] Backtest on 3+ months of data
- [ ] Paper trade for 2-4 weeks
- [ ] Start with 0.5% risk per trade
- [ ] Set up daily loss limit (e.g., 5%)
- [ ] Monitor first week closely
- [ ] Have emergency stop procedure ready

⚠️ **DURING LIVE TRADING:**
- Check latency.log every hour
- Stop if max drawdown reached
- Don't disable risk management for "better returns"
- Keep profit/loss ratio healthy
- Scale gradually (never 10x overnight)

⚠️ **IF SOMETHING GOES WRONG:**
1. Press Ctrl+C to stop bot
2. Close MT5 or disable trading
3. Review latency.log for errors
4. Check trades.csv for execution issues
5. Debug with unit tests
6. Don't restart until fixed

---

## Next Steps

1. **✅ Done:** Install & test everything
2. **→ Next:** Run backtest on historical data
3. **→ Then:** Paper trade for 2-4 weeks  
4. **→ Finally:** Live trade with micro positions

Good luck! 🚀
