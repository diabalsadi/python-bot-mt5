"""
Policy-gradient reinforcement learning agent.

The agent learns a trading policy directly instead of estimating Q-values.
State is expanded to 18 features so the policy can see market context,
multi-timeframe agreement, regime, order-flow stress, and SHAP conflict.
"""

from __future__ import annotations

import csv
import math
import os
from collections import deque
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np

STATE_DIM = 18
ACTION_DIM = 3
HIDDEN1 = 64
HIDDEN2 = 32
LR = 7e-4
EPSILON_START = 0.35
EPSILON_MIN = 0.03
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
REPLAY_CAP = 3000
REWARD_CLIP = 5.0
ENTROPY_BETA = 0.01

ACTIONS = {0: "HOLD", 1: "BUY", 2: "SELL"}
CSV_DATETIME_FMT = "%Y-%m-%d %H:%M:%S"


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(np.float64)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


class _PolicyMLP:
    """Small numpy MLP that outputs action logits."""

    def __init__(self, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)

        def he(fan_in: int, fan_out: int) -> np.ndarray:
            return rng.standard_normal((fan_in, fan_out)) * math.sqrt(2.0 / fan_in)

        self.W1 = he(STATE_DIM, HIDDEN1)
        self.b1 = np.zeros((1, HIDDEN1))
        self.W2 = he(HIDDEN1, HIDDEN2)
        self.b2 = np.zeros((1, HIDDEN2))
        self.W3 = he(HIDDEN2, ACTION_DIM)
        self.b3 = np.zeros((1, ACTION_DIM))
        self._m = [np.zeros_like(p) for p in self._params()]
        self._v = [np.zeros_like(p) for p in self._params()]
        self._t = 0

    def _params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return action logits. Kept as `forward` for compatibility."""
        self._x0 = x
        self._z1 = x @ self.W1 + self.b1
        self._a1 = _relu(self._z1)
        self._z2 = self._a1 @ self.W2 + self.b2
        self._a2 = _relu(self._z2)
        return self._a2 @ self.W3 + self.b3

    def backward_logits(self, dloss_dlogits: np.ndarray) -> None:
        batch = max(1, dloss_dlogits.shape[0])
        dW3 = (self._a2.T @ dloss_dlogits) / batch
        db3 = dloss_dlogits.mean(axis=0, keepdims=True)
        dA2 = dloss_dlogits @ self.W3.T

        dZ2 = dA2 * _relu_grad(self._z2)
        dW2 = (self._a1.T @ dZ2) / batch
        db2 = dZ2.mean(axis=0, keepdims=True)
        dA1 = dZ2 @ self.W2.T

        dZ1 = dA1 * _relu_grad(self._z1)
        dW1 = (self._x0.T @ dZ1) / batch
        db1 = dZ1.mean(axis=0, keepdims=True)
        self._adam_update([dW1, db1, dW2, db2, dW3, db3])

    def _adam_update(self, grads: list, beta1=0.9, beta2=0.999, eps=1e-8) -> None:
        self._t += 1
        for i, (g, p) in enumerate(zip(grads, self._params())):
            self._m[i] = beta1 * self._m[i] + (1.0 - beta1) * g
            self._v[i] = beta2 * self._v[i] + (1.0 - beta2) * (g ** 2)
            m_hat = self._m[i] / (1.0 - beta1 ** self._t)
            v_hat = self._v[i] / (1.0 - beta2 ** self._t)
            p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    def get_weights(self) -> dict:
        return {k: getattr(self, k).copy() for k in ("W1", "b1", "W2", "b2", "W3", "b3")}

    def set_weights(self, d: dict) -> None:
        for k, v in d.items():
            setattr(self, k, v.copy())


class ReplayBuffer:
    def __init__(self, capacity: int = REPLAY_CAP) -> None:
        self._buf: deque[Tuple] = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state=None, done=False) -> None:
        self._buf.append((
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state if next_state is not None else state, dtype=np.float32),
            float(done),
        ))

    def sample(self, n: int):
        idx = np.random.choice(len(self._buf), n, replace=False)
        batch = [self._buf[i] for i in idx]
        s = np.stack([b[0] for b in batch])
        a = np.array([b[1] for b in batch], dtype=np.int32)
        r = np.array([b[2] for b in batch], dtype=np.float64)
        ns = np.stack([b[3] for b in batch])
        d = np.array([b[4] for b in batch], dtype=np.float32)
        return s, a, r, ns, d

    def __len__(self) -> int:
        return len(self._buf)


def build_state(
    momentum: float,
    volatility: float,
    trend: float,
    rsi: float,
    consecutive_losses: int,
    open_time: datetime,
    recent_pnl: float,
    shap_top_norm: float = 0.0,
    shap_conflict: float = 0.0,
    ml_prediction: float = 0.0,
    mid_prediction: float = 0.0,
    long_prediction: float = 0.0,
    mtf_score: float = 0.0,
    regime_confidence: float = 0.0,
    spread_norm: float = 0.0,
    bar_range_ratio: float = 0.0,
    liquidity_score: float = 0.0,
    position_bias: float = 0.0,
) -> np.ndarray:
    """Build the 18-dim normalized policy state vector."""
    tod = (open_time.hour * 60 + open_time.minute) / 1440.0
    tod_sin = math.sin(2.0 * math.pi * tod)
    tod_cos = math.cos(2.0 * math.pi * tod)

    return np.array([
        np.clip(momentum / 5.0, -3.0, 3.0),
        np.clip(volatility / 2.0, 0.0, 5.0),
        np.clip(trend / 5.0, -3.0, 3.0),
        np.clip((rsi - 50.0) / 50.0, -1.5, 1.5),
        min(consecutive_losses / 5.0, 2.0),
        tod_sin,
        tod_cos,
        np.clip(recent_pnl / 5.0, -3.0, 3.0),
        float(np.clip(shap_top_norm, -1.0, 1.0)),
        float(np.clip(shap_conflict, 0.0, 1.0)),
        np.clip(ml_prediction / 5.0, -3.0, 3.0),
        np.clip(mid_prediction / 5.0, -3.0, 3.0),
        np.clip(long_prediction / 5.0, -3.0, 3.0),
        np.clip(mtf_score, -1.0, 1.0),
        np.clip(regime_confidence, 0.0, 1.0),
        np.clip(spread_norm / 2.0, 0.0, 3.0),
        np.clip(bar_range_ratio / 3.0, 0.0, 3.0),
        np.clip(liquidity_score + position_bias, -2.0, 2.0),
    ], dtype=np.float32)


class RLAgent:
    """Policy-gradient agent that gates ML BUY/SELL signals."""

    def __init__(self, weights_path: Optional[str] = None) -> None:
        self.policy = _PolicyMLP(seed=0)
        self.q_net = self.policy  # compatibility for existing diagnostics/tests
        self.replay = ReplayBuffer()
        self.epsilon = EPSILON_START
        self._steps = 0
        self.episode_rewards: List[float] = []
        self._recent_pnl: deque[float] = deque(maxlen=5)
        self._consec: dict[str, int] = {"BUY": 0, "SELL": 0}

        if weights_path and os.path.exists(weights_path):
            self.load(weights_path)
            print(f"RL policy loaded from {weights_path}")
        else:
            print("RL policy initialised (policy gradient)")

    def train_from_csv(self, csv_path: str, epochs: int = 20) -> None:
        experiences = self._load_csv_experiences(csv_path)
        if not experiences:
            print(f"RL train_from_csv: no usable trades in {csv_path}")
            return

        for exp in experiences:
            self.replay.push(*exp)

        if len(self.replay) < BATCH_SIZE:
            print(f"Only {len(self.replay)} RL experiences; need {BATCH_SIZE}")
            return

        total_loss = 0.0
        steps = 0
        for _ in range(epochs):
            for _ in range(max(1, len(self.replay) // BATCH_SIZE)):
                total_loss += self._learn_step()
                steps += 1

        self.epsilon = max(0.08, EPSILON_MIN)
        avg_loss = total_loss / max(steps, 1)
        print(f"RL policy training complete | {steps} steps | loss={avg_loss:.4f}")

    def _load_csv_experiences(self, csv_path: str) -> List[Tuple]:
        if not os.path.exists(csv_path):
            return []

        with open(csv_path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("open_time")]

        experiences = []
        consec = {"BUY": 0, "SELL": 0}
        prev_entry = None
        recent_pnls: deque[float] = deque(maxlen=5)

        for i, row in enumerate(rows):
            try:
                direction = row["direction"].strip().upper()
                entry_price = float(row["entry_price"])
                exit_price = float(row.get("exit_price") or entry_price)
                profit_usd = float(row["profit_usd"])
                duration_min = float(row.get("duration_mins") or 1.0)
                open_dt = datetime.strptime(row["open_time"], CSV_DATETIME_FMT)
            except (ValueError, KeyError):
                continue

            if direction not in ("BUY", "SELL"):
                continue

            ref_price = prev_entry if prev_entry is not None else entry_price
            momentum = (entry_price - ref_price) / max(ref_price, 1.0) * 1000.0
            volatility = abs(exit_price - entry_price) / max(duration_min, 0.1)
            trend = momentum
            rsi = 35.0 if direction == "BUY" else 65.0
            recent = float(np.mean(list(recent_pnls))) if recent_pnls else 0.0
            action = 1 if direction == "BUY" else 2
            reward = float(np.clip(profit_usd, -REWARD_CLIP, REWARD_CLIP))
            if duration_min < 1.0 and profit_usd < 0:
                reward = float(np.clip(reward * 1.5, -REWARD_CLIP, REWARD_CLIP))

            state = build_state(
                momentum=momentum,
                volatility=volatility,
                trend=trend,
                rsi=rsi,
                consecutive_losses=consec[direction],
                open_time=open_dt,
                recent_pnl=recent,
                ml_prediction=momentum,
                mid_prediction=trend,
            )
            next_state = state.copy()
            experiences.append((state, action, reward, next_state, i == len(rows) - 1))

            consec[direction] = consec[direction] + 1 if profit_usd < 0 else 0
            recent_pnls.append(profit_usd)
            prev_entry = entry_price

        return experiences

    def _action_probs(self, state: np.ndarray) -> np.ndarray:
        logits = self.policy.forward(state.reshape(1, -1).astype(np.float64))
        return _softmax(logits)[0]

    def act(self, state: np.ndarray, ml_action: Optional[int] = None) -> int:
        probs = self._action_probs(state)
        allowed = [0, ml_action] if ml_action is not None else [0, 1, 2]
        allowed = [a for a in allowed if a is not None]

        if np.random.random() < self.epsilon:
            weights = np.array([0.35 if a == 0 else 0.65 for a in allowed], dtype=np.float64)
            weights = weights / weights.sum()
            return int(np.random.choice(allowed, p=weights))

        best = max(allowed, key=lambda a: probs[a])
        return int(best)

    def act_verbose(self, state: np.ndarray, ml_action: Optional[int] = None) -> Tuple[int, str]:
        probs = self._action_probs(state)
        action = self.act(state, ml_action=ml_action)
        explanation = (
            f"P[HOLD]={probs[0]:.3f}  P[BUY]={probs[1]:.3f}  P[SELL]={probs[2]:.3f}"
            f" -> {ACTIONS[action]} (epsilon={self.epsilon:.2f})"
        )
        return action, explanation

    def store(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool = False) -> None:
        reward = float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP))
        self.replay.push(state, action, reward, next_state, done)
        self.episode_rewards.append(reward)
        self._recent_pnl.append(reward)

        direction = ACTIONS.get(action, "HOLD")
        if direction in ("BUY", "SELL"):
            self._consec[direction] = self._consec[direction] + 1 if reward < 0 else 0

    def learn(self) -> float:
        if len(self.replay) < BATCH_SIZE:
            return 0.0
        loss = self._learn_step()
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        self._steps += 1
        return loss

    def _learn_step(self) -> float:
        states, actions, rewards, _next_states, _done = self.replay.sample(BATCH_SIZE)
        states = states.astype(np.float64)
        logits = self.policy.forward(states)
        probs = _softmax(logits)

        reward_std = float(np.std(rewards))
        if reward_std > 1e-9:
            advantages = (rewards - float(np.mean(rewards))) / (reward_std + 1e-9)
        else:
            advantages = rewards.copy()

        grad = probs.copy()
        grad[np.arange(BATCH_SIZE), actions] -= 1.0
        grad *= advantages[:, None]

        entropy_grad = probs * (np.log(np.clip(probs, 1e-9, 1.0)) + 1.0)
        grad += ENTROPY_BETA * entropy_grad
        self.policy.backward_logits(grad)

        chosen = np.clip(probs[np.arange(BATCH_SIZE), actions], 1e-9, 1.0)
        loss = -float(np.mean(advantages * np.log(chosen)))
        return abs(loss)

    def save(self, path: str = "rl_weights.npz") -> None:
        np.savez(path, epsilon=np.array([self.epsilon]), steps=np.array([self._steps]), **self.policy.get_weights())
        print(f"RL policy saved -> {path} (epsilon={self.epsilon:.3f}, steps={self._steps})")

    def load(self, path: str = "rl_weights.npz") -> None:
        d = np.load(path)
        if d["W1"].shape[0] != STATE_DIM:
            print(f"RL weights shape {d['W1'].shape} does not match STATE_DIM={STATE_DIM}; reinitialising")
            return
        self.policy.set_weights({k: d[k] for k in ("W1", "b1", "W2", "b2", "W3", "b3")})
        self.epsilon = float(d["epsilon"][0])
        self._steps = int(d["steps"][0])

    def report(self) -> None:
        recent = list(self._recent_pnl)
        avg_r = sum(recent) / len(recent) if recent else 0.0
        wins = sum(1 for r in self.episode_rewards if r > 0)
        total = len(self.episode_rewards)
        wr = wins / total * 100.0 if total else 0.0
        print(
            f"RL Policy | epsilon={self.epsilon:.3f} | steps={self._steps} | "
            f"win%={wr:.1f} | recent avg P&L=${avg_r:+.2f} | replay={len(self.replay)}"
        )
