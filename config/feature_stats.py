"""
Feature normalization statistics.

Computed from historical training data.
Used to normalize live trading features before inference.

These values should be computed once during model training
and reused for all live predictions.
"""

FEATURE_STATS = {
    "momentum": {
        "mean": 0.0,
        "std": 100.0,  # typical range: -100 to +100
    },
    "volatility": {
        "mean": 2.0,
        "std": 3.0,  # typical range: 0.5 to 8
    },
    "trend": {
        "mean": 0.0,
        "std": 100.0,  # typical range: -200 to +200
    },
    "rsi": {
        "mean": 0.0,
        "std": 10.0,  # normalized to (raw_rsi - 50) / 10
    },
    "sr_distance": {
        "mean": 0.0,
        "std": 100.0,  # typical range: -200 to +200
    },
    "liquidity": {
        "mean": 0.5,
        "std": 0.3,  # typical range: 0 to 1
    },
    "volume_delta": {
        "mean": 0.0,
        "std": 100.0,  # typical range: -200 to +200
    },
    "spread_norm": {
        "mean": 1.0,
        "std": 1.5,  # ATR multiple: typically 0.2 to 3.0
    },
    "bar_range_ratio": {
        "mean": 1.0,
        "std": 1.0,  # typical range: 0.5 to 3.0
    },
}


def get_feature_stats(feature_name: str) -> dict:
    """
    Retrieve mean and std for a specific feature.
    
    Args:
        feature_name: One of the keys in FEATURE_STATS
        
    Returns:
        Dict with 'mean' and 'std' keys
    """
    if feature_name not in FEATURE_STATS:
        raise ValueError(f"Unknown feature: {feature_name}")
    return FEATURE_STATS[feature_name]
