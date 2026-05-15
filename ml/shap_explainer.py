"""
SHAP Explainer
--------------
Wraps shap.TreeExplainer around the XGBoost model to produce
per-prediction feature attributions.

Two outputs are used downstream:

1.  Logging  — top 2 SHAP features + values written to trades.csv for
    post-session audit ("why did the model predict SELL here?")

2.  RL state — two new dims added to the 10-dim state vector:
        dim 9:  shap_top_value   — magnitude of the most influential
                                   feature, normalised to [-1, 1]
        dim 10: shap_conflict    — 1.0 if the top SHAP feature CONTRADICTS
                                   the prediction direction, else 0.0
                                   (e.g. model predicts SELL but momentum
                                   SHAP is strongly positive → conflict)

"Conflict" is the key signal: both losing sessions showed the model
confidently wrong because one feature (momentum) was pushing the wrong
way while others agreed.  The RL agent learns to HOLD when conflict=1.

Requires: pip install shap
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

try:
    import shap as _shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    print("⚠️  shap not installed — run: pip install shap")

FEATURE_NAMES = [
    "momentum",
    "volatility",
    "trend_slope",
    "rsi",
    "sr_distance",
    "liquidity",
    "volume_delta",
    "spread_norm",
    "bar_range_ratio",
    "macd",
    "stochastic",
    "adx",
    "bollinger_prox",
    "smc",
    "ut_bot",
]

_N_FEATURES = len(FEATURE_NAMES)

# Which features are directional (positive = bullish signal)
_BULLISH_POSITIVE = {
    "momentum":       True,
    "volatility":     False,
    "trend_slope":    True,
    "rsi":            True,
    "sr_distance":    False,
    "liquidity":      False,
    "volume_delta":   True,    # positive delta = more buy volume
    "spread_norm":    False,   # non-directional
    "bar_range_ratio":False,   # non-directional
    "macd":           True,    # positive MACD = bullish
    "stochastic":     True,    # positive Stoch = bullish
    "adx":            False,   # trend strength, not direction
    "bollinger_prox": False,   # >0 could be overbought, non-directional direct
    "smc":            True,    # positive = bullish institutional bias
    "ut_bot":         True,    # positive = bullish ATR trailing crossover
}


class SHAPExplainer:
    """
    Thin wrapper around shap.TreeExplainer.

    Lifecycle:
        explainer = SHAPExplainer()
        explainer.fit(xgb_model)           # after each model.train()
        result = explainer.explain(X_row)  # shape (1, 6) numpy array
        # result.top_feature, result.top_shap_value, result.conflict
    """

    def __init__(self) -> None:
        self._explainer = None
        self.is_fitted  = False

    def fit(self, xgb_model) -> None:
        """
        Fit the TreeExplainer to a trained XGBRegressor.
        Call this immediately after model.train() completes.

        Args:
            xgb_model: trained xgboost.XGBRegressor instance
        """
        if not _SHAP_AVAILABLE:
            return
        try:
            self._explainer = _shap.TreeExplainer(xgb_model)
            self.is_fitted  = True
        except Exception as e:
            print(f"⚠️  SHAPExplainer.fit failed: {e}")
            self.is_fitted = False

    def explain(
        self,
        X_row: np.ndarray,
        prediction: float,
    ) -> "_SHAPResult":
        """
        Compute SHAP values for a single feature row.

        Args:
            X_row:      shape (1, 6) or (6,) — the raw (unscaled) feature values
            prediction: the model's predicted price change (positive=bullish)

        Returns:
            _SHAPResult with fields:
                .shap_values        np.ndarray shape (6,)
                .top_feature        str  — name of most influential feature
                .top_shap_value     float — SHAP value of top feature
                .top_raw_value      float — raw input value of top feature
                .conflict           bool — True if top feature contradicts prediction
                .summary            str  — human-readable one-liner
                .top2_str           str  — "momentum=+1.23, trend=-0.45" for CSV
        """
        null = _SHAPResult.null()

        if not self.is_fitted or not _SHAP_AVAILABLE:
            return null

        try:
            X = np.array(X_row, dtype=np.float64).reshape(1, -1)
            sv = self._explainer.shap_values(X)  # shape (1, N) or (N,)
            sv = np.array(sv).flatten()[:_N_FEATURES]  # always (N_FEATURES,)

            # Top feature by absolute SHAP value
            top_idx  = int(np.argmax(np.abs(sv)))
            top_name = FEATURE_NAMES[top_idx]
            top_val  = float(sv[top_idx])
            top_raw  = float(X[0, top_idx])

            # Conflict detection:
            # prediction > 0 = model is bullish
            # if top feature is directional AND its SHAP attribution pushes the
            # OPPOSITE way → conflict
            is_directional = _BULLISH_POSITIVE.get(top_name, False)
            conflict = False
            if is_directional:
                model_bullish = prediction > 0
                # SHAP value positive means "this feature pushed prediction UP"
                shap_bullish  = top_val > 0
                conflict = (model_bullish != shap_bullish)

            # Top-2 string for CSV
            top2_idx = np.argsort(np.abs(sv))[::-1][:2]
            top2_str = ", ".join(
                f"{FEATURE_NAMES[i]}={sv[i]:+.4f}" for i in top2_idx
            )

            pred_dir  = "BULL" if prediction > 0 else "BEAR"
            conf_flag = " ⚠️ CONFLICT" if conflict else ""
            summary   = (
                f"[SHAP] pred={pred_dir} | top={top_name}({top_val:+.4f})"
                f"{conf_flag} | {top2_str}"
            )

            return _SHAPResult(
                shap_values=sv,
                top_feature=top_name,
                top_shap_value=top_val,
                top_raw_value=top_raw,
                conflict=conflict,
                summary=summary,
                top2_str=top2_str,
            )

        except Exception as e:
            print(f"⚠️  SHAPExplainer.explain error: {e}")
            return null

    def shap_rl_dims(
        self,
        X_row: np.ndarray,
        prediction: float,
    ) -> Tuple[float, float]:
        """
        Return the two SHAP-derived dims to append to the RL state vector.

        Returns:
            (shap_top_norm, shap_conflict)
            shap_top_norm:  top SHAP value clipped and normalised to [-1, 1]
            shap_conflict:  1.0 if conflict, 0.0 otherwise
        """
        result = self.explain(X_row, prediction)
        shap_top_norm = float(np.clip(result.top_shap_value / 5.0, -1.0, 1.0))
        shap_conflict = 1.0 if result.conflict else 0.0
        return shap_top_norm, shap_conflict


class _SHAPResult:
    __slots__ = (
        "shap_values", "top_feature", "top_shap_value",
        "top_raw_value", "conflict", "summary", "top2_str",
    )

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def null(cls) -> "_SHAPResult":
        return cls(
            shap_values=np.zeros(_N_FEATURES),
            top_feature="unknown",
            top_shap_value=0.0,
            top_raw_value=0.0,
            conflict=False,
            summary="[SHAP] unavailable",
            top2_str="",
        )
