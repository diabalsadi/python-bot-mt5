"""
Linear Regression Model
-----------------------
A lightweight single-pass linear regression model that predicts future
price movement from four technical features.  The algorithm is a direct
port of MQL5's TrainLinearModel() / PredictPriceChange() — it uses the
univariate OLS formula applied independently to each feature, then combines
them with an intercept (beta0), intentionally matching the original EA's
simplified (not full multivariate) regression approach.

MQL5 equivalents:
  TrainLinearModel()    → LinearRegressionModel.train()
  PredictPriceChange()  → LinearRegressionModel.predict()
"""

from __future__ import annotations

import MetaTrader5 as mt5
import numpy as np

from ml.features import get_features


class LinearRegressionModel:
    """
    Simplified multivariate linear regression model:

        ŷ = β0 + β1·x1 + β2·x2 + β3·x3 + β4·x4

    where x1..x4 are momentum, volatility, trend slope, and RSI respectively,
    and ŷ is the predicted price change (in points) over the next
    `prediction_horizon` bars.
    """

    def __init__(self) -> None:
        self.beta0: float = 0.0
        self.beta1: float = 0.0   # momentum coefficient
        self.beta2: float = 0.0   # volatility coefficient
        self.beta3: float = 0.0   # trend coefficient
        self.beta4: float = 0.0   # RSI coefficient
        self.is_trained: bool = False

    # ──────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────

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
        Fit the model coefficients on the most recent `training_bars` candles.

        The target label y for each training bar i is:
            y = (close[i - horizon] - close[i]) / point
        i.e. the actual price movement `prediction_horizon` bars into the future
        (looking back in the already-recorded history).

        Args:
            symbol:             Trading symbol
            timeframe:          MT5 timeframe constant
            training_bars:      Number of historical bars to train on
            prediction_horizon: Forward-look in bars (MQL5 MLPredictionHorizon)
            feature_window:     Feature look-back window (MQL5 MLFeatureWindow)
            rsi_period:         RSI period for the RSI feature
        """
        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            print("❌ ML Train: symbol info unavailable")
            return

        point = sym_info.point
        n     = training_bars

        # Fetch enough bars once so individual feature calls can slice locally
        count = n + prediction_horizon + feature_window + rsi_period + 10
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)

        if rates is None or len(rates) < count:
            print(f"❌ ML Train: need {count} bars, got {0 if rates is None else len(rates)}")
            return

        closes = rates["close"][::-1]  # index 0 = current bar

        # ── Accumulators ──────────────────────────────────────────────
        sum_x1 = sum_x2 = sum_x3 = sum_x4 = 0.0
        sum_y  = 0.0
        sum_x1y = sum_x2y = sum_x3y = sum_x4y = 0.0
        sum_x1_2 = sum_x2_2 = sum_x3_2 = sum_x4_2 = 0.0

        for i in range(prediction_horizon, n + prediction_horizon):
            x1, x2, x3, x4 = get_features(symbol, timeframe, i, feature_window, rsi_period)

            current_price = closes[i]
            future_price  = closes[i - prediction_horizon]   # closer to "now"
            y = (future_price - current_price) / point

            sum_x1 += x1;   sum_x2 += x2;   sum_x3 += x3;   sum_x4 += x4
            sum_y  += y

            sum_x1y  += x1 * y;  sum_x2y  += x2 * y
            sum_x3y  += x3 * y;  sum_x4y  += x4 * y

            sum_x1_2 += x1 * x1; sum_x2_2 += x2 * x2
            sum_x3_2 += x3 * x3; sum_x4_2 += x4 * x4

        # ── OLS (univariate, per feature) ─────────────────────────────
        avg_x1 = sum_x1 / n;  avg_x2 = sum_x2 / n
        avg_x3 = sum_x3 / n;  avg_x4 = sum_x4 / n
        avg_y  = sum_y  / n

        var_x1 = (sum_x1_2 / n) - avg_x1 ** 2
        var_x2 = (sum_x2_2 / n) - avg_x2 ** 2
        var_x3 = (sum_x3_2 / n) - avg_x3 ** 2
        var_x4 = (sum_x4_2 / n) - avg_x4 ** 2

        self.beta1 = ((sum_x1y / n) - avg_x1 * avg_y) / var_x1 if var_x1 else 0.0
        self.beta2 = ((sum_x2y / n) - avg_x2 * avg_y) / var_x2 if var_x2 else 0.0
        self.beta3 = ((sum_x3y / n) - avg_x3 * avg_y) / var_x3 if var_x3 else 0.0
        self.beta4 = ((sum_x4y / n) - avg_x4 * avg_y) / var_x4 if var_x4 else 0.0

        self.beta0 = avg_y - (
            self.beta1 * avg_x1 + self.beta2 * avg_x2 +
            self.beta3 * avg_x3 + self.beta4 * avg_x4
        )

        self.is_trained = True

        print(
            f"🤖 ML Model Trained | "
            f"β0={self.beta0:.4f}  β1={self.beta1:.4f}  "
            f"β2={self.beta2:.4f}  β3={self.beta3:.4f}  β4={self.beta4:.4f}"
        )

    # ──────────────────────────────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────────────────────────────

    def predict(
        self,
        symbol: str,
        timeframe: int,
        feature_window: int,
        rsi_period: int,
    ) -> float:
        """
        Predict the expected price change for the current bar in price units.

        Uses shift=1 (last fully closed bar) to avoid look-ahead bias.

        Args:
            symbol:         Trading symbol
            timeframe:      MT5 timeframe constant
            feature_window: Feature window (must match what was used in train)
            rsi_period:     RSI period (must match train)

        Returns:
            Predicted price change in price units (not points).
            Returns 0.0 if the model has not been trained yet.
        """
        if not self.is_trained:
            return 0.0

        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            return 0.0

        point = sym_info.point
        x1, x2, x3, x4 = get_features(symbol, timeframe, 1, feature_window, rsi_period)

        prediction_points = (
            self.beta0
            + self.beta1 * x1
            + self.beta2 * x2
            + self.beta3 * x3
            + self.beta4 * x4
        )

        return prediction_points * point   # convert points → price units

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:  # noqa: D105
        status = "trained" if self.is_trained else "untrained"
        return (
            f"LinearRegressionModel({status}) "
            f"β=({self.beta0:.4f}, {self.beta1:.4f}, "
            f"{self.beta2:.4f}, {self.beta3:.4f}, {self.beta4:.4f})"
        )
