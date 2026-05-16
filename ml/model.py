"""
Gradient boosting prediction model.

Uses XGBoost and/or LightGBM when installed, plus an optional MLP
component. The public class name is kept for compatibility with the rest of
the bot.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import MetaTrader5 as mt5
import numpy as np

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False

from ml.features import get_features
from ml.shap_explainer import SHAPExplainer

USE_XGB = os.getenv("USE_XGB", "1").lower() in ("1", "true", "yes")
USE_LGBM = os.getenv("USE_LGBM", "1").lower() in ("1", "true", "yes")
USE_MLP = os.getenv("USE_MLP_ENSEMBLE", "1").lower() in ("1", "true", "yes")
ENSEMBLE_WEIGHTS = os.getenv("ENSEMBLE_WEIGHTS", "xgb:0.5,lgbm:0.3,mlp:0.2")


class LinearRegressionModel:
    """
    Backward-compatible name for the boosted prediction ensemble.

    Public attributes beta0..beta15 are retained for existing logging and are
    populated from feature importances when tree models are available.
    """

    def __init__(self) -> None:
        self.beta0 = 0.0
        for i in range(1, 16):
            setattr(self, f"beta{i}", 0.0)

        self.is_trained = False
        self.prediction_horizon = 15

        self._xgb_model = None
        self._lgbm_model = None
        self._mlp_model = None
        self._ols_betas = None
        self._feature_mean = None
        self._feature_std = None
        self._weights = self._parse_weights(ENSEMBLE_WEIGHTS)
        self.shap = SHAPExplainer()

        available = []
        if _XGB_AVAILABLE and USE_XGB:
            available.append("XGBoost")
        if _LGBM_AVAILABLE and USE_LGBM:
            available.append("LightGBM")
        if not available:
            print("No XGBoost/LightGBM backend available; using OLS fallback.")
        else:
            print(f"ML backends enabled: {', '.join(available)}")

    @staticmethod
    def _parse_weights(raw: str) -> Dict[str, float]:
        weights = {"xgb": 0.5, "lgbm": 0.3, "mlp": 0.2}
        try:
            if ":" in raw:
                parsed = {}
                for part in raw.split(","):
                    name, value = part.split(":", 1)
                    parsed[name.strip().lower()] = max(0.0, float(value))
                weights.update(parsed)
            else:
                vals = [max(0.0, float(x.strip())) for x in raw.split(",")]
                for key, value in zip(("xgb", "lgbm", "mlp"), vals):
                    weights[key] = value
        except (ValueError, TypeError):
            pass
        return weights

    def train(
        self,
        symbol: str,
        timeframe: int,
        training_bars: int,
        prediction_horizon: int,
        feature_window: int,
        rsi_period: int,
    ) -> None:
        """Fit the model on recent candles without look-ahead leakage."""
        self.prediction_horizon = prediction_horizon

        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            print("ML train: symbol info unavailable")
            return

        point = sym_info.point
        count = training_bars + prediction_horizon + feature_window + rsi_period + 10
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None or len(rates) < count:
            got = 0 if rates is None else len(rates)
            print(f"ML train: need {count} bars, got {got}")
            return

        closes = rates["close"][::-1]
        rows, targets = [], []
        for i in range(prediction_horizon, training_bars + prediction_horizon):
            rows.append(get_features(symbol, timeframe, i, feature_window, rsi_period))
            current_price = closes[i]
            future_price = closes[i - prediction_horizon]
            targets.append((future_price - current_price) / point)

        X = np.array(rows, dtype=np.float64)
        y = np.array(targets, dtype=np.float64)

        self._reset_models()
        if (_XGB_AVAILABLE and USE_XGB) or (_LGBM_AVAILABLE and USE_LGBM):
            self._train_ensemble(X, y)
        else:
            self._train_ols(X, y)

        self.is_trained = True

    def _reset_models(self) -> None:
        self._xgb_model = None
        self._lgbm_model = None
        self._mlp_model = None
        self._ols_betas = None

    def _train_ensemble(self, X: np.ndarray, y: np.ndarray) -> None:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        self._feature_mean = mean
        self._feature_std = std

        if _XGB_AVAILABLE and USE_XGB:
            try:
                model = xgb.XGBRegressor(
                    n_estimators=250,
                    max_depth=4,
                    learning_rate=0.04,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.1,
                    reg_lambda=1.2,
                    objective="reg:squarederror",
                    verbosity=0,
                    n_jobs=1,
                    random_state=42,
                )
                model.fit(X, y)
                self._xgb_model = model
            except Exception as exc:
                print(f"XGBoost training failed: {exc}")

        if _LGBM_AVAILABLE and USE_LGBM:
            try:
                self._lgbm_model = lgb.LGBMRegressor(
                    n_estimators=250,
                    max_depth=-1,
                    num_leaves=31,
                    learning_rate=0.04,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=1.0,
                    objective="regression",
                    random_state=42,
                    verbose=-1,
                )
                self._lgbm_model.fit(X, y)
            except Exception as exc:
                print(f"LightGBM training failed: {exc}")

        if USE_MLP:
            try:
                from sklearn.neural_network import MLPRegressor

                X_norm = (X - mean) / std
                self._mlp_model = MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    max_iter=250,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=42,
                )
                self._mlp_model.fit(X_norm, y)
            except Exception as exc:
                print(f"MLP training skipped: {exc}")

        if self._xgb_model is not None:
            self.shap.fit(self._xgb_model)
        elif self._lgbm_model is not None:
            self.shap.fit(self._lgbm_model)

        if not self._collect_predictions(np.zeros((1, X.shape[1]))):
            self._train_ols(X, y)
            return

        self._map_feature_importance(X.shape[1])

    def _train_ols(self, X: np.ndarray, y: np.ndarray) -> None:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        X_norm = (X - mean) / std
        X_aug = np.column_stack([np.ones(len(X_norm)), X_norm])
        betas, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)

        self._ols_betas = betas
        self._feature_mean = mean
        self._feature_std = std
        self.beta0 = float(betas[0])
        for i in range(1, min(16, len(betas))):
            setattr(self, f"beta{i}", float(betas[i]))

    def _map_feature_importance(self, feature_count: int) -> None:
        fi = np.zeros(feature_count)
        if self._xgb_model is not None:
            fi = np.asarray(self._xgb_model.feature_importances_, dtype=np.float64)
        elif self._lgbm_model is not None:
            fi = np.asarray(self._lgbm_model.feature_importances_, dtype=np.float64)

        total = float(np.sum(np.abs(fi)))
        if total > 0:
            fi = fi / total

        self.beta0 = 0.0
        for i in range(1, 16):
            setattr(self, f"beta{i}", float(fi[i - 1]) if i - 1 < len(fi) else 0.0)

    def _collect_predictions(self, X: np.ndarray) -> List[Tuple[str, float]]:
        preds: List[Tuple[str, float]] = []
        if self._xgb_model is not None:
            preds.append(("xgb", float(self._xgb_model.predict(X)[0])))
        if self._lgbm_model is not None:
            preds.append(("lgbm", float(self._lgbm_model.predict(X)[0])))
        if self._mlp_model is not None and self._feature_mean is not None:
            X_norm = (X - self._feature_mean) / self._feature_std
            preds.append(("mlp", float(self._mlp_model.predict(X_norm)[0])))
        return preds

    def _blend_predictions(self, preds: List[Tuple[str, float]]) -> float:
        if not preds:
            return 0.0
        weights = [self._weights.get(name, 1.0) for name, _ in preds]
        total = sum(weights)
        if total <= 0:
            return float(np.mean([value for _, value in preds]))
        return float(sum(value * weight for (_, value), weight in zip(preds, weights)) / total)

    def _predict_points(self, X: np.ndarray) -> float:
        preds = self._collect_predictions(X)
        if preds:
            return self._blend_predictions(preds)
        if self._ols_betas is not None:
            X_norm = (X - self._feature_mean) / self._feature_std
            X_aug = np.column_stack([np.ones(1), X_norm])
            return float(X_aug @ self._ols_betas)
        return 0.0

    def predict(self, symbol: str, timeframe: int, feature_window: int, rsi_period: int) -> float:
        """Predict expected price change for the current bar in price units."""
        if not self.is_trained:
            return 0.0

        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            return 0.0

        feats = get_features(symbol, timeframe, 0, feature_window, rsi_period)
        X = np.array(feats, dtype=np.float64).reshape(1, -1)
        return self._predict_points(X) * sym_info.point

    def predict_ahead(
        self,
        symbol: str,
        timeframe: int,
        feature_window: int,
        rsi_period: int,
        horizon: int = 15,
    ) -> float:
        """Extrapolate the current feature prediction to another horizon."""
        if not self.is_trained:
            return 0.0

        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            return 0.0

        feats = get_features(symbol, timeframe, 0, feature_window, rsi_period)
        X = np.array(feats, dtype=np.float64).reshape(1, -1)
        ratio = horizon / max(1, self.prediction_horizon)
        return self._predict_points(X) * ratio * sym_info.point

    def __repr__(self) -> str:
        models = []
        if self._xgb_model is not None:
            models.append("XGB")
        if self._lgbm_model is not None:
            models.append("LGBM")
        if self._mlp_model is not None:
            models.append("MLP")
        if self._ols_betas is not None:
            models.append("OLS")
        backend = "+".join(models) if models else "unfitted"
        status = "trained" if self.is_trained else "untrained"
        return f"PredictionModel({backend}, {status})"
