
FEATURE_STATS = {
    "momentum": {"mean": 0.0, "std": 100.0},
    "volatility": {"mean": 2.0, "std": 3.0},
    "trend": {"mean": 0.0, "std": 100.0},
    "rsi": {"mean": 0.0, "std": 10.0},
    "sr_distance": {"mean": 0.0, "std": 100.0},
    "liquidity": {"mean": 0.5, "std": 0.3},
    "volume_delta": {"mean": 0.0, "std": 100.0},
    "spread_norm": {"mean": 1.0, "std": 1.5},
    "bar_range_ratio": {"mean": 1.0, "std": 1.0},
    "macd": {"mean": 0.0, "std": 5.0},
    "stochastic": {"mean": 0.0, "std": 0.5},
    "adx": {"mean": 0.0, "std": 0.5},
    "bollinger_prox": {"mean": 0.0, "std": 0.5},
    "smc": {"mean": 0.0, "std": 0.5},
    "ut_bot": {"mean": 0.0, "std": 0.5},
}
def get_feature_stats(feature_name: str) -> dict:
    if feature_name not in FEATURE_STATS:
        raise ValueError(f"Unknown feature: {feature_name}")
    return FEATURE_STATS[feature_name]

