"""Multi-timeframe signal fusion helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FusionResult:
    confirmed_prediction: float
    score: float
    direction: int
    confidence: float
    reason: str


def _signed_strength(value: float, scale: float) -> float:
    if scale <= 0:
        scale = max(abs(value), 1e-9)
    return math.tanh(value / scale)


def fuse_timeframes(
    short_prediction: float,
    mid_prediction: float,
    long_prediction: float,
    long_volatility: float,
    regime_confidence: float = 0.0,
    min_confidence: float = 0.22,
) -> FusionResult:
    """
    Weighted, confidence-aware timeframe fusion.

    Short timeframe keeps execution timing, mid timeframe gets veto power when
    it is strongly opposed, and long timeframe shapes position bias instead of
    acting as a blunt soft filter.
    """
    scale = max(abs(short_prediction), abs(mid_prediction), abs(long_prediction), long_volatility, 1e-9)
    s = _signed_strength(short_prediction, scale)
    m = _signed_strength(mid_prediction, scale)
    l = _signed_strength(long_prediction, max(scale, long_volatility * 0.75))

    mid_opposes_hard = s * m < 0 and abs(m) > abs(s) * 1.4
    long_opposes_hard = s * l < 0 and abs(long_prediction) > max(long_volatility * 0.65, abs(short_prediction) * 1.2)
    if mid_opposes_hard:
        return FusionResult(0.0, 0.0, 0, 0.0, "mid timeframe hard opposition")
    if long_opposes_hard and regime_confidence > 0.45:
        return FusionResult(0.0, 0.0, 0, 0.0, "long timeframe regime opposition")

    regime_weight = min(max(regime_confidence, 0.0), 1.0)
    w_short = 0.50 - 0.10 * regime_weight
    w_mid = 0.30
    w_long = 0.20 + 0.10 * regime_weight
    score = s * w_short + m * w_mid + l * w_long
    confidence = abs(score)

    if confidence < min_confidence:
        return FusionResult(0.0, score, 0, confidence, "weak fused confidence")

    direction = 1 if score > 0 else -1
    if short_prediction * direction <= 0:
        return FusionResult(0.0, score, 0, confidence, "short timing disagrees with fused direction")

    confirmed = abs(short_prediction) * direction * (0.75 + min(confidence, 1.0) * 0.5)
    return FusionResult(confirmed, score, direction, confidence, "fused")
