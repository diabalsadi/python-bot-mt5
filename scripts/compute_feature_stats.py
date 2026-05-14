"""
Compute feature statistics from historical data.

Usage:
    python scripts/compute_feature_stats.py

This script:
1. Pulls historical OHLCV data
2. Computes all 9 features
3. Calculates mean/std for each feature
4. Updates config/feature_stats.py

Run this ONCE after collecting enough historical data to ensure
stable statistics for live trading.
"""

import numpy as np
import MetaTrader5 as mt5
from ml.features import get_features


def compute_feature_stats(symbol: str, timeframe: int, n_bars: int = 5000):
    """
    Compute mean and std for all 9 features.
    
    Args:
        symbol: MT5 symbol (e.g., 'XAUUSDm')
        timeframe: MT5 timeframe (e.g., mt5.TIMEFRAME_M5)
        n_bars: number of historical bars to use
    
    Returns:
        Dict with feature stats
    """
    
    if not mt5.initialize():
        print("❌ MT5 not initialized")
        return None
    
    print(f"🔍 Computing stats for {symbol} | {n_bars} bars")
    
    # Get feature_window and rsi_period from config
    FEATURE_WINDOW = 12
    RSI_PERIOD = 14
    
    feature_names = [
        "momentum", "volatility", "trend", "rsi",
        "sr_distance", "liquidity", "volume_delta",
        "spread_norm", "bar_range_ratio"
    ]
    
    # Collect all feature values
    feature_data = {name: [] for name in feature_names}
    
    for shift in range(n_bars):
        try:
            feats = get_features(
                symbol,
                timeframe,
                shift,
                FEATURE_WINDOW,
                RSI_PERIOD,
            )
            
            for i, name in enumerate(feature_names):
                feature_data[name].append(feats[i])
                
        except Exception as e:
            print(f"⚠️  Error at shift {shift}: {e}")
            continue
    
    # Compute stats
    stats = {}
    for name in feature_names:
        values = np.array(feature_data[name])
        mean = float(np.mean(values))
        std = float(np.std(values))
        
        stats[name] = {
            "mean": round(mean, 4),
            "std": round(max(std, 0.01), 4),  # Ensure std >= 0.01
        }
        
        print(f"  {name:15} | mean={mean:8.4f} | std={std:8.4f}")
    
    return stats


def update_config(stats: dict):
    """Update config/feature_stats.py with computed stats."""
    
    config_content = '''"""
Feature normalization statistics (auto-generated).

Computed from historical data.
DO NOT edit manually — use scripts/compute_feature_stats.py to update.
"""

FEATURE_STATS = {
'''
    
    for name, values in stats.items():
        config_content += f'''    "{name}": {{
        "mean": {values["mean"]},
        "std": {values["std"]},
    }},
'''
    
    config_content += '''}


def get_feature_stats(feature_name: str) -> dict:
    """Retrieve mean and std for a specific feature."""
    if feature_name not in FEATURE_STATS:
        raise ValueError(f"Unknown feature: {feature_name}")
    return FEATURE_STATS[feature_name]
'''
    
    with open("config/feature_stats.py", "w") as f:
        f.write(config_content)
    
    print(f"\n✅ Updated config/feature_stats.py")


if __name__ == "__main__":
    # Configure these
    SYMBOL = "XAUUSDm"
    TIMEFRAME = 5  # M5
    N_BARS = 5000  # 1+ month of history
    
    stats = compute_feature_stats(SYMBOL, TIMEFRAME, N_BARS)
    
    if stats:
        update_config(stats)
        print(f"\n✅ Done! New stats are ready for live trading.")
    else:
        print("❌ Failed to compute stats")
