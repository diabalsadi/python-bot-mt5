# Feature Normalization Guide

## Problem

Without normalization, feature values explode:
```
rsi = 1767
spread_norm = -2105
trend_slope = 1315
```

This causes the ML model to see these as "HUGE SIGNALS" when they're actually scale issues.

## Solution

All features are normalized using **robust normalization**:

```
normalized_value = (raw_value - mean) / std
```

Then clipped to [-5, 5] to prevent outliers.

## Architecture

```
Market Data
    ↓
Feature Calculation (raw values can be huge)
    ↓
Normalization (using config/feature_stats.py)
    ↓
Clipping to [-5, 5]
    ↓
ML Model (receives stable [-5, 5] features)
```

## Files Involved

- **`ml/features.py`** — Computes raw features, then normalizes each one
- **`utils/normalization.py`** — `robust_normalize()` and `safe_clip()` functions
- **`config/feature_stats.py`** — Stores historical mean/std for each feature
- **`scripts/compute_feature_stats.py`** — Generates new stats from historical data

## Initialization

When you first start:
1. `config/feature_stats.py` has default mean/std values
2. Run the bot in **backtest mode** to accumulate data
3. Call `python scripts/compute_feature_stats.py`
4. This updates `feature_stats.py` with real statistics
5. Restart the bot with accurate stats

## Critical: Consistency

**MUST normalize identically during:**
- Training: `ml/model.py` calls `get_features()` → normalizes → trains
- Inference: `ml/model.py` calls `get_features()` → normalizes → predicts

If you train on normalized data but infer on raw data, **the model breaks**.

## Feature Stats Format

```python
FEATURE_STATS = {
    "momentum": {
        "mean": 0.0,
        "std": 100.0,
    },
    "rsi": {
        "mean": 0.0,
        "std": 10.0,
    },
    # ... 7 more features
}
```

## Updating Stats

After trading for 2-4 weeks with real data:

```bash
python scripts/compute_feature_stats.py
```

This:
1. Pulls 5000+ bars of historical data
2. Computes mean/std for each of 9 features
3. Updates `config/feature_stats.py` automatically
4. Prints the new stats to console

## Troubleshooting

**Q: Features still seem large (e.g., rsi=5.0 after normalization)**
- A: This is expected! Values in [-5, 5] are clipped outliers
- The model trained on similar ranges

**Q: Predictions suddenly change after updating stats**
- A: Normal! Stats are now more accurate to current market conditions
- Old trades used biased statistics

**Q: Should I update stats frequently?**
- A: Every 2-4 weeks, or when market regime changes (e.g., new volatility environment)
- Too frequent = overfitting to noise
- Too infrequent = stale statistics
