"""
Deep Q-Network (DQN) Reinforcement Learning Agent
--------------------------------------------------
Learns from real trade history (trades.csv) to override or confirm
the XGBoost signal — essentially a meta-policy on top of the ML model.

Why RL on top of XGBoost:
  - XGBoost predicts direction well in stationary conditions.
  - RL learns *when* to act: it penalises entering during streaks,
    trending against momentum, or after multiple consecutive losses.
  - Trained offline from trades.csv, then updated online after each close.

State (8 features):
  [momentum, volatility, trend, rsi_norm,
   consecutive_losses, time_sin, time_cos, recent_avg_pnl]

Actions:
  0 = HOLD  (skip the ML signal — don't trade)
  1 = BUY
  2 = SELL

Reward:
  actual trade P&L in USD (clipped to [-5, 5] to stabilise training)

Architecture:
  Input(8) → Dense(64, ReLU) → Dense(32, ReLU) → Output(3)
  Trained with experience replay and a frozen target network.
  Pure numpy — no PyTorch / TensorFlow required.

Usage:
    from ml.rl_agent import RLAgent
    agent = RLAgent()
    agent.train_from_csv("trades.csv")   # offline warm-start
    action = agent.act(state)            # 0=HOLD,1=BUY,2=SELL
    agent.store(state, action, reward, next_state, done)
    agent.learn()                        # online update
    agent.save("rl_weights.npz")
    agent.load("rl_weights.npz")
"""

from __future__ import annotations

import csv
import math
import os
import time
from collections import deque
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
STATE_DIM    = 8
ACTION_DIM   = 3          # HOLD=0, BUY=1, SELL=2
HIDDEN1      = 64
HIDDEN2      = 32
GAMMA        = 0.95       # discount factor
LR           = 1e-3       # learning rate
EPSILON_START = 1.0
EPSILON_MIN   = 0.05
EPSILON_DECAY = 0.995
BATCH_SIZE    = 32
REPLAY_CAP    = 2000
TARGET_UPDATE = 50        # sync target net every N learn steps
REWARD_CLIP   = 5.0       # clip P&L reward to [-5, 5]

ACTIONS = {0: "HOLD", 1: "BUY", 2: "SELL"}
CSV_DATETIME_FMT = "%Y-%m-%d %H:%M:%S"


# ── Tiny MLP (numpy) ─────────────────────────────────────────────────────────

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float64)


class _MLP:
    """2-hidden-layer MLP with He initialisation, trained via SGD."""

    def __init__(self, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)

        def he(fan_in: int, fan_out: int) -> np.ndarray:
            return rng.standard_normal((fan_in, fan_out)) * math.sqrt(2 / fan_in)

        # Weights & biases
        self.W1 = he(STATE_DIM, HIDDEN1);  self.b1 = np.zeros((1, HIDDEN1))
        self.W2 = he(HIDDEN1,   HIDDEN2);  self.b2 = np.zeros((1, HIDDEN2))
        self.W3 = he(HIDDEN2,   ACTION_DIM); self.b3 = np.zeros((1, ACTION_DIM))

        # Adam moment estimates
        self._m  = [np.zeros_like(p) for p in self._params()]
        self._v  = [np.zeros_like(p) for p in self._params()]
        self._t  = 0

    def _params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, STATE_DIM) → (batch, ACTION_DIM)"""
        self._x0 = x
        self._z1 = x @ self.W1 + self.b1;   self._a1 = _relu(self._z1)
        self._z2 = self._a1 @ self.W2 + self.b2; self._a2 = _relu(self._z2)
        self._z3 = self._a2 @ self.W3 + self.b3
        return self._z3  # Q-values (no activation on output)

    def backward(self, dLoss_dOut: np.ndarray) -> None:
        """MSE gradient backprop + Adam update."""
        batch = dLoss_dOut.shape[0]

        # Layer 3
        dW3 = (self._a2.T @ dLoss_dOut) / batch
        db3 = dLoss_dOut.mean(axis=0, keepdims=True)
        dA2 = dLoss_dOut @ self.W3.T

        # Layer 2
        dZ2 = dA2 * _relu_grad(self._z2)
        dW2 = (self._a1.T @ dZ2) / batch
        db2 = dZ2.mean(axis=0, keepdims=True)
        dA1 = dZ2 @ self.W2.T

        # Layer 1
        dZ1 = dA1 * _relu_grad(self._z1)
        dW1 = (self._x0.T @ dZ1) / batch
        db1 = dZ1.mean(axis=0, keepdims=True)

        grads = [dW1, db1, dW2, db2, dW3, db3]
        self._adam_update(grads)

    def _adam_update(self, grads: list, beta1=0.9, beta2=0.999, eps=1e-8) -> None:
        self._t += 1
        for i, (g, p) in enumerate(zip(grads, self._params())):
            self._m[i] = beta1 * self._m[i] + (1 - beta1) * g
            self._v[i] = beta2 * self._v[i] + (1 - beta2) * g ** 2
            m_hat = self._m[i] / (1 - beta1 ** self._t)
            v_hat = self._v[i] / (1 - beta2 ** self._t)
            p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    def copy_weights_from(self, other: "_MLP") -> None:
        self.W1[:] = other.W1; self.b1[:] = other.b1
        self.W2[:] = other.W2; self.b2[:] = other.b2
        self.W3[:] = other.W3; self.b3[:] = other.b3

    def get_weights(self) -> dict:
        return {k: getattr(self, k).copy()
                for k in ("W1","b1","W2","b2","W3","b3")}

    def set_weights(self, d: dict) -> None:
        for k, v in d.items():
            setattr(self, k, v.copy())


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int = REPLAY_CAP) -> None:
        self._buf: deque[Tuple] = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self._buf.append((
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, n: int):
        idx = np.random.choice(len(self._buf), n, replace=False)
        batch = [self._buf[i] for i in idx]
        s  = np.stack([b[0] for b in batch])
        a  = np.array([b[1] for b in batch], dtype=np.int32)
        r  = np.array([b[2] for b in batch], dtype=np.float32)
        ns = np.stack([b[3] for b in batch])
        d  = np.array([b[4] for b in batch], dtype=np.float32)
        return s, a, r, ns, d

    def __len__(self) -> int:
        return len(self._buf)


# ── State builder ─────────────────────────────────────────────────────────────

def build_state(
    momentum:            float,
    volatility:          float,
    trend:               float,
    rsi:                 float,
    consecutive_losses:  int,
    open_time:           datetime,
    recent_pnl:          float,
) -> np.ndarray:
    """
    Build the 8-dim normalised state vector.

    Args:
        momentum:           (price_now - price_N_ago) / price_N_ago * 1000
        volatility:         std of last 14 closes (raw)
        trend:              (price_now - price_20_ago) / price_20_ago * 1000
        rsi:                RSI(14) value [0–100]
        consecutive_losses: count of same-direction losses in a row
        open_time:          datetime of the bar
        recent_pnl:         mean P&L of last 5 closed trades
    """
    tod  = (open_time.hour * 60 + open_time.minute) / 1440.0
    tod_sin = math.sin(2 * math.pi * tod)
    tod_cos = math.cos(2 * math.pi * tod)

    return np.array([
        np.clip(momentum / 5.0,   -3, 3),          # normalise ~[-3,3]
        np.clip(volatility / 2.0, 0, 5),            # normalise
        np.clip(trend / 5.0,      -3, 3),
        (rsi - 50.0) / 50.0,                        # [-1, 1]
        min(consecutive_losses / 5.0, 2.0),         # cap at 2
        tod_sin,
        tod_cos,
        np.clip(recent_pnl / 5.0, -3, 3),           # normalise
    ], dtype=np.float32)


# ── DQN Agent ─────────────────────────────────────────────────────────────────

class RLAgent:
    """
    DQN agent that learns to gate the XGBoost signal.

    Call flow:
        1.  agent.train_from_csv("trades.csv")   — offline warm-start
        2.  action = agent.act(state)             — live gating
        3.  agent.store(s, a, r, s2, done)        — after each close
        4.  agent.learn()                         — one gradient step
        5.  agent.save("rl_weights.npz")          — persist

    The agent returns one of three actions:
        HOLD (0) — skip this ML signal
        BUY  (1) — confirm BUY signal
        SELL (2) — confirm SELL signal
    """

    def __init__(self, weights_path: Optional[str] = None) -> None:
        self.q_net     = _MLP(seed=0)
        self.target_net = _MLP(seed=0)
        self.target_net.copy_weights_from(self.q_net)

        self.replay   = ReplayBuffer()
        self.epsilon  = EPSILON_START
        self._steps   = 0

        # Running stats for reporting
        self.episode_rewards: List[float] = []
        self._recent_pnl: deque[float]    = deque(maxlen=5)
        self._consec: dict[str, int]      = {"BUY": 0, "SELL": 0}

        if weights_path and os.path.exists(weights_path):
            self.load(weights_path)
            print(f"✅ RL agent loaded from {weights_path}")
        else:
            print("🤖 RL agent initialised (untrained — run train_from_csv first)")

    # ── Offline training from CSV ─────────────────────────────────────────────

    def train_from_csv(self, csv_path: str, epochs: int = 20) -> None:
        """
        Build experience replay from trades.csv and train for `epochs` passes.

        This warm-starts the agent so it already penalises the patterns
        observed in real trade history (streaks, momentum mismatches, etc.)
        before it sees a single live tick.
        """
        experiences = self._load_csv_experiences(csv_path)
        if not experiences:
            print(f"⚠️  RL train_from_csv: no usable trades in {csv_path}")
            return

        for exp in experiences:
            self.replay.push(*exp)

        print(f"📥 Loaded {len(experiences)} trade experiences from {csv_path}")

        if len(self.replay) < BATCH_SIZE:
            print(f"⚠️  Only {len(self.replay)} experiences — need {BATCH_SIZE} for training")
            return

        total_loss = 0.0
        steps = 0
        for epoch in range(epochs):
            for _ in range(max(1, len(self.replay) // BATCH_SIZE)):
                loss = self._learn_step()
                total_loss += loss
                steps += 1

        avg_loss = total_loss / max(steps, 1)
        # After offline training, lower epsilon since we have real priors
        self.epsilon = max(0.2, EPSILON_MIN)
        print(f"✅ RL offline training complete | {steps} steps | avg loss={avg_loss:.4f} | ε={self.epsilon:.2f}")

    def _load_csv_experiences(self, csv_path: str) -> List[Tuple]:
        """
        Convert rows in trades.csv into (state, action, reward, next_state, done) tuples.

        Mistakes the agent learns to avoid:
          - Entering SELL when consecutive_losses >= 3 in same direction
          - Entering against a strong momentum (e.g. positive momentum + SELL)
          - Quick SL hits (<1 min) → state had bad momentum
        """
        if not os.path.exists(csv_path):
            print(f"⚠️  {csv_path} not found — skipping offline training")
            return []

        rows = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if not r.get("open_time"):
                    continue
                rows.append(r)

        if not rows:
            return []

        experiences = []
        consec: dict[str, int] = {"BUY": 0, "SELL": 0}
        prev_price: dict[str, float] = {}
        recent_pnls: deque[float]    = deque(maxlen=5)

        for i, row in enumerate(rows):
            try:
                direction    = row["direction"].strip().upper()
                entry_price  = float(row["entry_price"])
                profit_usd   = float(row["profit_usd"])
                duration_min = float(row["duration_mins"])
                open_dt      = datetime.strptime(row["open_time"], CSV_DATETIME_FMT)
            except (ValueError, KeyError):
                continue

            if direction not in ("BUY", "SELL"):
                continue

            # ── Build state ───────────────────────────────────────────
            pp  = prev_price.get(direction, entry_price)
            mom = (entry_price - pp) / max(pp, 1) * 1000

            # Approximate volatility from duration and price move
            exit_price = float(row.get("exit_price") or entry_price)
            vol = abs(exit_price - entry_price) / max(duration_min, 0.1)

            # Approximate trend from session momentum (entry vs session start)
            session_start = float(rows[0]["entry_price"]) if rows else entry_price
            trend = (entry_price - session_start) / max(session_start, 1) * 1000

            # Approximate RSI (oversold/overbought proxy)
            rsi = 30.0 if direction == "BUY" and profit_usd > 0 else \
                  70.0 if direction == "SELL" and profit_usd > 0 else 50.0

            state = build_state(
                momentum=mom,
                volatility=vol,
                trend=trend,
                rsi=rsi,
                consecutive_losses=consec[direction],
                open_time=open_dt,
                recent_pnl=float(np.mean(list(recent_pnls))) if recent_pnls else 0.0,
            )

            # ── Action ───────────────────────────────────────────────
            action = 1 if direction == "BUY" else 2

            # ── Reward ───────────────────────────────────────────────
            # Penalise extra for quick SL hits (bad entry timing)
            reward = profit_usd
            if duration_min < 1.0 and profit_usd < 0:
                reward *= 1.5   # 50% harsher penalty for rushed entries
            reward = float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP))

            # ── Next state ───────────────────────────────────────────
            # Use next row's entry as the "next state" approximation
            if i + 1 < len(rows):
                nr = rows[i + 1]
                try:
                    next_dir = nr["direction"].strip().upper()
                    next_ep  = float(nr["entry_price"])
                    next_dt  = datetime.strptime(nr["open_time"], CSV_DATETIME_FMT)
                    next_consec = consec.get(next_dir, 0)
                    next_mom    = (next_ep - entry_price) / max(entry_price, 1) * 1000
                    next_state  = build_state(
                        momentum=next_mom, volatility=vol, trend=trend,
                        rsi=50.0, consecutive_losses=next_consec,
                        open_time=next_dt,
                        recent_pnl=float(np.mean([*recent_pnls, profit_usd][-5:])),
                    )
                except (ValueError, KeyError):
                    next_state = state.copy()
            else:
                next_state = state.copy()

            done = (i == len(rows) - 1)
            experiences.append((state, action, reward, next_state, done))

            # ── Update trackers ───────────────────────────────────────
            consec[direction] = (consec[direction] + 1) if profit_usd < 0 else 0
            prev_price[direction] = entry_price
            recent_pnls.append(profit_usd)

        return experiences

    # ── Live inference ────────────────────────────────────────────────────────

    def act(self, state: np.ndarray, ml_action: Optional[int] = None) -> int:
        """
        Choose an action given the current state.

        Args:
            state:     8-dim state vector from build_state()
            ml_action: XGBoost suggestion (1=BUY, 2=SELL) or None.
                       If provided and RL says HOLD, the ML signal is blocked.

        Returns:
            0 = HOLD, 1 = BUY, 2 = SELL
        """
        if np.random.random() < self.epsilon:
            # During exploration: either follow ML suggestion or HOLD randomly
            if ml_action and np.random.random() < 0.7:
                return ml_action   # mostly follow ML during exploration
            return np.random.randint(0, ACTION_DIM)

        x = state.reshape(1, -1).astype(np.float64)
        q = self.q_net.forward(x)[0]   # shape (3,)

        # If ML gave a suggestion, only choose between HOLD and that suggestion
        if ml_action is not None:
            best = ml_action if q[ml_action] > q[0] else 0
        else:
            best = int(np.argmax(q))

        return best

    def act_verbose(self, state: np.ndarray, ml_action: Optional[int] = None) -> Tuple[int, str]:
        """Same as act() but returns (action, explanation string) for logging."""
        x = state.reshape(1, -1).astype(np.float64)
        q = self.q_net.forward(x)[0]

        if ml_action is not None:
            action = ml_action if q[ml_action] > q[0] else 0
        else:
            action = int(np.argmax(q))

        explanation = (
            f"Q[HOLD]={q[0]:+.3f}  Q[BUY]={q[1]:+.3f}  Q[SELL]={q[2]:+.3f}"
            f"  → {ACTIONS[action]}"
            f"  (ε={self.epsilon:.2f})"
        )
        return action, explanation

    # ── Online learning ───────────────────────────────────────────────────────

    def store(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool = False,
    ) -> None:
        """Store a real trade outcome in the replay buffer."""
        reward = float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP))
        self.replay.push(state, action, reward, next_state, done)
        self.episode_rewards.append(reward)
        self._recent_pnl.append(reward)

        direction = ACTIONS.get(action, "HOLD")
        if direction in ("BUY", "SELL"):
            self._consec[direction] = (self._consec[direction] + 1) if reward < 0 else 0

    def learn(self) -> float:
        """One gradient update step. Returns the loss."""
        if len(self.replay) < BATCH_SIZE:
            return 0.0

        loss = self._learn_step()

        # Decay exploration
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        self._steps += 1

        # Sync target network
        if self._steps % TARGET_UPDATE == 0:
            self.target_net.copy_weights_from(self.q_net)

        return loss

    def _learn_step(self) -> float:
        """Core DQN Bellman update."""
        s, a, r, ns, done = self.replay.sample(BATCH_SIZE)

        s  = s.astype(np.float64)
        ns = ns.astype(np.float64)

        # Current Q-values
        q_pred = self.q_net.forward(s)           # (batch, 3)

        # Target Q-values (Bellman equation)
        q_next  = self.target_net.forward(ns)    # (batch, 3)
        q_max   = q_next.max(axis=1)             # (batch,)
        q_target = r + GAMMA * q_max * (1 - done)

        # Only update Q for the taken action
        q_labels = q_pred.copy()
        for i in range(BATCH_SIZE):
            q_labels[i, a[i]] = q_target[i]

        # MSE loss gradient
        dLoss = 2 * (q_pred - q_labels) / BATCH_SIZE
        self.q_net.backward(dLoss)

        loss = float(np.mean((q_pred - q_labels) ** 2))
        return loss

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = "rl_weights.npz") -> None:
        w = self.q_net.get_weights()
        np.savez(path, epsilon=np.array([self.epsilon]),
                 steps=np.array([self._steps]), **w)
        print(f"💾 RL agent saved → {path} (ε={self.epsilon:.3f}, steps={self._steps})")

    def load(self, path: str = "rl_weights.npz") -> None:
        d = np.load(path)
        self.q_net.set_weights({k: d[k] for k in ("W1","b1","W2","b2","W3","b3")})
        self.target_net.copy_weights_from(self.q_net)
        self.epsilon = float(d["epsilon"][0])
        self._steps  = int(d["steps"][0])

    # ── Reporting ─────────────────────────────────────────────────────────────

    def report(self) -> None:
        """Print agent stats — call periodically."""
        recent = list(self._recent_pnl)
        avg_r  = sum(recent) / len(recent) if recent else 0.0
        wins   = sum(1 for r in self.episode_rewards if r > 0)
        total  = len(self.episode_rewards)
        wr     = wins / total * 100 if total else 0

        print(
            f"🤖 RL Agent | ε={self.epsilon:.3f} | "
            f"steps={self._steps} | win%={wr:.1f} | "
            f"recent avg P&L=${avg_r:+.2f} | "
            f"replay={len(self.replay)} | "
            f"SELL streak={self._consec.get('SELL',0)} "
            f"BUY streak={self._consec.get('BUY',0)}"
        )
