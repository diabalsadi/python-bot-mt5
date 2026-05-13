# python-bot-mt5 — Gold Scalping Bot

Algorithmic trading bot for XAUUSDm (gold) on MetaTrader 5.
3-tier ML stack: **XGBoost** direction prediction → **DQN Reinforcement Learning** entry gating → **SHAP** explainability.

---

## Stack

| Layer | What it does |
|---|---|
| XGBoost | Predicts price direction from momentum / volatility / trend / RSI / S&R / liquidity |
| DQN RL agent | Gates every entry — learns from `trades.csv` mistakes, adapts each session |
| SHAP | Explains why XGBoost predicted what it did; `shap_conflict` flag fed to RL state |
| Hard filters | ATR trend gate, consecutive-loss pause, anti-hedge guard, max positions cap |

---

## Requirements

- Python 3.10+
- MetaTrader 5 terminal running and logged in (Windows only)

---

## Installation

```bash
# 1. Clone
git clone https://github.com/diabalsadi/python-bot-mt5.git
cd python-bot-mt5

# 2. Virtual environment
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Windows CMD
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Dependencies
pip install -r requirements.txt
```

---

## Configuration

Create `.env` in the project root:

```
LOGIN    = 260791257
PASSWORD = "your_password"
SERVER   = "Exness-MT5Trial15"
MT5_PATH = "C:\Program Files\MetaTrader 5\terminal64.exe"
```

---

## Usage

### Run tests (no MT5 needed)

```bash
python -m pytest tests/ -v
```

### Backtest on historical data

```bash
# Last 30 days (default)
python backtest.py

# Custom range and symbol
python backtest.py --symbol XAUUSDm --days 90

# Full options
python backtest.py --symbol XAUUSDm --days 90 --balance 10000 --risk 10 --sl 150
```

Output:
```
🔄 Backtesting XAUUSDm  2024-11-01 → 2025-01-30
   Risk: 10.0%/trade  SL: 150pt  TP: 300pt
📊 43,200 bars loaded  (30.0 trading days)
  … 10,000/43,200 bars  trades=47  balance=$11,234.56

============================================================
BACKTEST RESULTS
============================================================
Total Trades:              312
Win Rate:                  58.33%
Total P&L:               $4,821.00
Profit Factor:           2.14
Max Drawdown:            12.40%
Sharpe Ratio:            1.87
============================================================
```

Results are saved to `backtest_results.csv`.

### Run the live bot

```bash
python main.py
```

### Check bot status manually

```python
from position_manager import get_position_manager
get_position_manager().report_status()
```

### View RL agent stats

```python
from ml.rl_agent import RLAgent
agent = RLAgent(weights_path="rl_weights.npz")
agent.report()
```

---

## Key configuration (main.py)

```python
SYMBOL               = "XAUUSDm"
RISK_PERCENT         = 10.0      # % of balance risked per trade
SL_POINTS            = 2000      # stop-loss distance in points
TRAILING_STEP_POINTS = 300       # trailing step
ML_TRAINING_BARS     = 1440      # bars used for XGBoost training (24h on M1)
ML_RETRAIN_INTERVAL  = 10        # retrain every N bars
ENABLE_SCALE_IN      = True      # add to winning positions
MAX_CONSECUTIVE_LOSSES = 3       # pause direction after N losses
LOSS_PAUSE_SECONDS   = 300       # pause duration (5 min)
```

---

## Output files

| File | Contents |
|---|---|
| `trades.csv` | Every trade: entry/exit, P&L, SHAP top feature, conflict flag, duration |
| `rl_weights.npz` | RL agent weights — persists learning across sessions |
| `backtest_results.csv` | Backtest trade list |
| `latency.log` | MT5 connection latency per tick |

---

## Project structure

```
python-bot-mt5/
├── main.py                      # Event loop — on_tick()
├── backtest.py                  # Historical simulation (CLI)
├── position_manager.py          # Max concurrent positions cap
├── actions/
│   ├── connection.py            # MT5 connect + reconnection watchdog
│   └── strategy.py              # All trading logic (entries, SL/TP, scale-in)
├── ml/
│   ├── model.py                 # XGBoost prediction model
│   ├── features.py              # Feature engineering (6 features)
│   ├── rl_agent.py              # DQN reinforcement learning agent
│   └── shap_explainer.py        # SHAP feature attribution
├── indicators/                  # ATR, RSI, S&R, Fibonacci, liquidity zones
├── tools/
│   ├── trade_logger.py          # CSV trade logging with SHAP columns
│   └── print.py                 # Pretty-print helpers
├── tests/
│   └── test_trading_logic.py    # 35 unit tests (pytest)
├── requirements.txt
└── .env                         # Credentials (not committed)
```

---

## How the ML stack works

```
on_tick()
  │
  ├─ XGBoost (M1)  ──┐
  ├─ XGBoost (M15) ──┼─ confluence filter (must agree, mid can't dominate opposite)
  ├─ XGBoost (H1)  ──┘
  │
  ├─ SHAP explain → conflict flag (top feature contradicts prediction?)
  │
  ├─ Hard filters:
  │   ├─ ATR trend gate      (blocks counter-trend during surge)
  │   ├─ Consecutive loss    (pauses direction after 3 losses)
  │   └─ Anti-hedge guard    (no BUY if SELL open, vice versa)
  │
  └─ DQN RL agent (10-dim state incl. SHAP dims)
      ├─ HOLD → skip entry
      └─ CONFIRM → execute_buy/sell_market()
```

---

## Troubleshooting

**`MT5 init failed`** — Open MT5 terminal and log in before running the bot.

**`No historical data`** — Check symbol name: try `XAUUSDm` (micro) or `XAUUSD`. Must be in MT5 Market Watch.

**`PositionManager: BUY blocked — Max positions reached`** — 3 positions already open. Adjust `max_positions` in `position_manager.py`.

**Bot not trading** — Check console for `🚫 Trend gate` or `⏸ SELL paused` — these are the hard filters working. Also check `ML Short: ... Confirmed: +0.00000` — if confirmed is zero, M1/M15 models disagree.

**RL agent starting cold** — Normal on first run. Train it faster by running `python -c "from ml.rl_agent import RLAgent; a = RLAgent(); a.train_from_csv('trades.csv')"` after accumulating a session's worth of trades.

---

## Dependencies

```
MetaTrader5     broker connection (Windows only)
python-dotenv   .env credential loading
numpy           maths, RL backprop, OLS fallback
xgboost         gradient-boosted tree prediction model
shap            SHAP feature attribution (TreeExplainer)
pytest          test runner
```

Install: `pip install -r requirements.txt`
