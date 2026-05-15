"""
XGBoost Prediction Model
------------------------
Replaces the previous hand-rolled linear regression with a proper
gradient-boosted tree model (XGBoost) and feature standardisation.

Why XGBoost over linear regression for gold scalping:
  - Captures non-linear relationships between features and price movement
  - Handles correlated features without inflating coefficients
  - Built-in feature importance for transparency
  - Regularisation (L1/L2) avoids overfitting on noisy tick data

XGBoost is scale-invariant (tree splits on thresholds, not distances),
so scikit-learn's StandardScaler is not needed.  We store mean/std
manually for the OLS fallback path and for normalising SHAP inputs.

Requires:
    pip install xgboost
"""

from __future__ import annotations

import numpy as np
import MetaTrader5 as mt5

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    print("⚠️  xgboost/scikit-learn not installed — falling back to OLS regression.")
    print("    Run: pip install xgboost scikit-learn")

from ml.features import get_features
from ml.shap_explainer import SHAPExplainer


class LinearRegressionModel:
    """
    XGBoost-backed prediction model with StandardScaler normalisation.

    Falls back to OLS (via np.linalg.lstsq) if xgboost is not installed.
    Maintains the same public interface as the old LinearRegressionModel
    so all call-sites in main.py / strategy.py work unchanged.

    Public attributes kept for backward compatibility:
        is_trained, beta0..beta6 (mapped to feature importances when using XGB),
        prediction_horizon
    """

    def __init__(self) -> None:
        # Backward-compat beta attributes (repurposed as feature importances)
        self.beta0: float = 0.0
        self.beta1: float = 0.0
        self.beta2: float = 0.0
        self.beta3: float = 0.0
        self.beta4: float = 0.0
        self.beta5: float = 0.0
        self.beta6: float = 0.0
        self.beta7: float = 0.0   # volume_delta
        self.beta8: float = 0.0   # spread_norm
        self.beta9: float = 0.0   # bar_range_ratio
        self.beta10: float = 0.0  # macd
        self.beta11: float = 0.0  # stochastic
        self.beta12: float = 0.0  # adx
        self.beta13: float = 0.0  # bollinger_prox

        self.is_trained: bool = False
        self.prediction_horizon: int = 15

        # XGBoost internals
        self._xgb_model = None
        self._mlp_model = None  # Deep Neural Network component
        # No scaler needed for XGBoost, but MLP requires standardisation

        # OLS fallback internals (also stores mean/std for SHAP normalisation)
        self._ols_betas = None
        self._feature_mean = None
        self._feature_std = None

        self._use_xgb: bool = _XGB_AVAILABLE

        # SHAP explainer — fitted after each train() call
        self.shap = SHAPExplainer()

    # ── Training ──────────────────────────────────────────────────────

    def train(
        self,
        symbol: str,
        timeframe: int,
        training_bars: int,
        prediction_horizon: int,
        feature_window: int,
        rsi_period: int,
    ) -> None:
        """
        Fit the model on the most recent `training_bars` candles.

        Target label y[i] = (close[i - horizon] - close[i]) / point
        i.e. actual price movement `prediction_horizon` bars ahead.
        """
        self.prediction_horizon = prediction_horizon

        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            print("❌ ML Train: symbol info unavailable")
            return

        point = sym_info.point
        n     = training_bars
        count = n + prediction_horizon + feature_window + rsi_period + 10
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)

        if rates is None or len(rates) < count:
            got = 0 if rates is None else len(rates)
            print(f"❌ ML Train: need {count} bars, got {got}")
            return

        closes = rates["close"][::-1]   # index 0 = most recent

        # ── Build feature matrix X and target y ───────────────────────
        rows, ys = [], []
        for i in range(prediction_horizon, n + prediction_horizon):
            feats = get_features(
                symbol, timeframe, i, feature_window, rsi_period
            )
            current_price = closes[i]
            future_price  = closes[i - prediction_horizon]
            y = (future_price - current_price) / point
            rows.append(feats)
            ys.append(y)

        X = np.array(rows, dtype=np.float64)
        y = np.array(ys,   dtype=np.float64)

        if self._use_xgb:
            self._train_xgb(X, y, point)
        else:
            self._train_ols(X, y)

        self.is_trained = True

    def _train_xgb(self, X, y, point) -> None:
        """Fit an Ensemble of XGBRegressor and MLPRegressor (Deep Neural Network)."""
        mean = X.mean(axis=0)
        std  = X.std(axis=0);  std[std == 0] = 1.0
        self._feature_mean = mean
        self._feature_std  = std

        # 1. XGBoost (Scale-invariant)
        model_xgb = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="reg:squarederror",
            verbosity=0,
            n_jobs=1,
        )
        model_xgb.fit(X, y)
        self._xgb_model = model_xgb

        # 2. Deep Neural Network (MLP requires scaled inputs)
        try:
            from sklearn.neural_network import MLPRegressor
            X_norm = (X - mean) / std
            model_mlp = MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                solver='adam',
                max_iter=200,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            )
            model_mlp.fit(X_norm, y)
            self._mlp_model = model_mlp
        except Exception as e:
            print(f"⚠️  MLP training failed: {e}. Falling back to pure XGBoost.")
            self._mlp_model = None

        # Fit SHAP explainer on the XGBoost model (TreeExplainer is much faster)
        self.shap.fit(model_xgb)

        # Map feature importances to beta slots for the existing logging line
        fi = model_xgb.feature_importances_
        self.beta0 = 0.0
        self.beta1, self.beta2, self.beta3 = float(fi[0]), float(fi[1]), float(fi[2])
        self.beta4, self.beta5, self.beta6 = float(fi[3]), float(fi[4]), float(fi[5])
        self.beta7 = float(fi[6]) if len(fi) > 6 else 0.0
        self.beta8 = float(fi[7]) if len(fi) > 7 else 0.0
        self.beta9 = float(fi[8]) if len(fi) > 8 else 0.0
        self.beta10 = float(fi[9]) if len(fi) > 9 else 0.0
        self.beta11 = float(fi[10]) if len(fi) > 10 else 0.0
        self.beta12 = float(fi[11]) if len(fi) > 11 else 0.0
        self.beta13 = float(fi[12]) if len(fi) > 12 else 0.0

    def _train_ols(self, X, y) -> None:
        """Fallback: joint OLS via numpy lstsq with manual standardisation."""
        mean = X.mean(axis=0)
        std  = X.std(axis=0)
        std[std == 0] = 1.0
        X_norm = (X - mean) / std

        X_aug = np.column_stack([np.ones(len(X_norm)), X_norm])
        betas, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)

        self._ols_betas    = betas
        self._feature_mean = mean
        self._feature_std  = std

        self.beta0 = float(betas[0])
        self.beta1, self.beta2, self.beta3 = float(betas[1]), float(betas[2]), float(betas[3])
        self.beta4, self.beta5, self.beta6 = float(betas[4]), float(betas[5]), float(betas[6])

    # ── Prediction ────────────────────────────────────────────────────

    def predict(
        self,
        symbol: str,
        timeframe: int,
        feature_window: int,
        rsi_period: int,
    ) -> float:
        """
        Predict expected price change for the current bar (in price units).
        Uses current bar features — no look-ahead bias.
        """
        if not self.is_trained:
            return 0.0

        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            return 0.0

        point = sym_info.point
        feats = get_features(symbol, timeframe, 0, feature_window, rsi_period)
        X = np.array(feats, dtype=np.float64).reshape(1, -1)

        if self._use_xgb and self._xgb_model is not None:
            pred_xgb = float(self._xgb_model.predict(X)[0])
            
            if self._mlp_model is not None:
                X_norm = (X - self._feature_mean) / self._feature_std
                pred_mlp = float(self._mlp_model.predict(X_norm)[0])
                # Ensemble: 60% XGBoost, 40% Neural Network
                pred_pts = (pred_xgb * 0.6) + (pred_mlp * 0.4)
            else:
                pred_pts = pred_xgb

        else:
            X_norm = (X - self._feature_mean) / self._feature_std
            X_aug  = np.column_stack([np.ones(1), X_norm])
            pred_pts = float(X_aug @ self._ols_betas)

        return pred_pts * point

    def predict_ahead(
        self,
        symbol: str,
        timeframe: int,
        feature_window: int,
        rsi_period: int,
        horizon: int = 15,
    ) -> float:
        """
        Extrapolate prediction `horizon` bars ahead using current features only.
        No future OHLCV data is read — pure model extrapolation.
        """
        if not self.is_trained:
            return 0.0

        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            return 0.0

        point = sym_info.point
        feats = get_features(symbol, timeframe, 0, feature_window, rsi_period)
        X = np.array(feats, dtype=np.float64).reshape(1, -1)
        h_ratio = horizon / max(1, self.prediction_horizon)

        if self._use_xgb and self._xgb_model is not None:
            pred_xgb = float(self._xgb_model.predict(X)[0])
            
            if self._mlp_model is not None:
                X_norm = (X - self._feature_mean) / self._feature_std
                pred_mlp = float(self._mlp_model.predict(X_norm)[0])
                pred_pts = ((pred_xgb * 0.6) + (pred_mlp * 0.4)) * h_ratio
            else:
                pred_pts = pred_xgb * h_ratio
        else:
            X_norm = (X - self._feature_mean) / self._feature_std
            X_aug  = np.column_stack([np.ones(1), X_norm])
            pred_pts = float(X_aug @ self._ols_betas) * h_ratio

        return pred_pts * point

    # ── Repr ──────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        backend = "XGBoost+MLP Ensemble" if (self._use_xgb and self._mlp_model) else ("XGBoost" if self._xgb_model else "OLS")
        status  = "trained" if self.is_trained else "untrained"
        return f"PredictionModel({backend}, {status})"
