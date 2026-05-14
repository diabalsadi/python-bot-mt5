import time
from dataclasses import dataclass


@dataclass
class TradeDecision:
    allowed: bool
    reason: str


class TradeFilter:
    def __init__(self):
        self.last_trade_time = 0.0
        self.consecutive_losses = 0
        self.pause_until = 0.0

        self.MIN_SECONDS_BETWEEN_TRADES = 60
        self.MAX_CONSECUTIVE_LOSSES = 3
        self.LOSS_PAUSE_SECONDS = 1800

        self.MAX_SPREAD_NORM = 1.5
        self.MAX_BAR_RANGE = 2.5
        self.MIN_LIQUIDITY = -0.30
        self.MAX_VOLATILITY = 3.0

    def register_loss(self):
        self.consecutive_losses += 1

        if self.consecutive_losses >= self.MAX_CONSECUTIVE_LOSSES:
            self.pause_until = time.time() + self.LOSS_PAUSE_SECONDS

    def register_win(self):
        self.consecutive_losses = 0

    def validate(
        self,
        spread_norm,
        bar_range_ratio,
        liquidity_score,
        volatility,
    ):
        now = time.time()

        if now < self.pause_until:
            return TradeDecision(False, "loss cooldown")

        if now - self.last_trade_time < self.MIN_SECONDS_BETWEEN_TRADES:
            return TradeDecision(False, "trade cooldown")

        if spread_norm > self.MAX_SPREAD_NORM:
            return TradeDecision(False, "spread too high")

        if bar_range_ratio > self.MAX_BAR_RANGE:
            return TradeDecision(False, "market unstable")

        if liquidity_score < self.MIN_LIQUIDITY:
            return TradeDecision(False, "poor liquidity")

        if abs(volatility) > self.MAX_VOLATILITY:
            return TradeDecision(False, "volatility spike")

        self.last_trade_time = now

        return TradeDecision(True, "approved")