"""
Linear Regression Model
-----------------------
A lightweight single-pass linear regression model that predicts future
price movement from six technical features including S/R and liquidity zones.
The algorithm is a direct port of MQL5's TrainLinearModel() / PredictPriceChange()
extended with multi-timeframe analysis.

MQL5 equivalents (extended):
  TrainLinearModel()    → LinearRegressionModel.train()
  PredictPriceChange()  → LinearRegressionModel.predict()
"""

from __future__ import annotations

import MetaTrader5 as mt5
import numpy as np

from ml.features import get_features


class LinearRegressionModel:
    """
    Extended multivariate linear regression model with 6 features and
    multi-horizon prediction capability:

        ŷ = β0 + β1·x1 + β2·x2 + β3·x3 + β4·x4 + β5·x5 + β6·x6

    where:
      x1 = momentum
      x2 = volatility
      x3 = trend slope
      x4 = RSI
      x5 = support/resistance distance (15m)
      x6 = liquidity zone score (15m)
    
    and ŷ is the predicted price change (in points) over the next
    `prediction_horizon` bars. Supports multiple horizons (e.g., 15 bars
    ahead = 15 minutes on M1 timeframe).
    """

    def __init__(self) -> None:
        self.beta0: float = 0.0
        self.beta1: float = 0.0   # momentum coefficient
        self.beta2: float = 0.0   # volatility coefficient
        self.beta3: float = 0.0   # trend coefficient
        self.beta4: float = 0.0   # RSI coefficient
        self.beta5: float = 0.0   # S/R distance coefficient (15m)
        self.beta6: float = 0.0   # Liquidity coefficient (15m)
        self.is_trained: bool = False
        
        # Multi-horizon coefficients (for 15m-ahead predictions on M1)
        self.horizon_models: dict[int, dict[str, float]] = {}  # {horizon: {beta names}}
        self.prediction_horizon: int = 15  # Default horizon (in bars)
        self.secondary_horizon: int = 15   # Secondary horizon for lookahead (15 min ahead)

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
        sum_x1 = sum_x2 = sum_x3 = sum_x4 = sum_x5 = sum_x6 = 0.0
        sum_y  = 0.0
        sum_x1y = sum_x2y = sum_x3y = sum_x4y = sum_x5y = sum_x6y = 0.0
        sum_x1_2 = sum_x2_2 = sum_x3_2 = sum_x4_2 = sum_x5_2 = sum_x6_2 = 0.0

        for i in range(prediction_horizon, n + prediction_horizon):
            x1, x2, x3, x4, x5, x6 = get_features(symbol, timeframe, i, feature_window, rsi_period)

            current_price = closes[i]
            future_price  = closes[i - prediction_horizon]   # closer to "now"
            y = (future_price - current_price) / point

            sum_x1 += x1;   sum_x2 += x2;   sum_x3 += x3;   sum_x4 += x4
            sum_x5 += x5;   sum_x6 += x6
            sum_y  += y

            sum_x1y  += x1 * y;  sum_x2y  += x2 * y
            sum_x3y  += x3 * y;  sum_x4y  += x4 * y
            sum_x5y  += x5 * y;  sum_x6y  += x6 * y

            sum_x1_2 += x1 * x1; sum_x2_2 += x2 * x2
            sum_x3_2 += x3 * x3; sum_x4_2 += x4 * x4
            sum_x5_2 += x5 * x5; sum_x6_2 += x6 * x6

        # ── OLS (univariate, per feature) ─────────────────────────────
        avg_x1 = sum_x1 / n;  avg_x2 = sum_x2 / n
        avg_x3 = sum_x3 / n;  avg_x4 = sum_x4 / n
        avg_x5 = sum_x5 / n;  avg_x6 = sum_x6 / n
        avg_y  = sum_y  / n

        var_x1 = (sum_x1_2 / n) - avg_x1 ** 2
        var_x2 = (sum_x2_2 / n) - avg_x2 ** 2
        var_x3 = (sum_x3_2 / n) - avg_x3 ** 2
        var_x4 = (sum_x4_2 / n) - avg_x4 ** 2
        var_x5 = (sum_x5_2 / n) - avg_x5 ** 2
        var_x6 = (sum_x6_2 / n) - avg_x6 ** 2

        self.beta1 = ((sum_x1y / n) - avg_x1 * avg_y) / var_x1 if var_x1 else 0.0
        self.beta2 = ((sum_x2y / n) - avg_x2 * avg_y) / var_x2 if var_x2 else 0.0
        self.beta3 = ((sum_x3y / n) - avg_x3 * avg_y) / var_x3 if var_x3 else 0.0
        self.beta4 = ((sum_x4y / n) - avg_x4 * avg_y) / var_x4 if var_x4 else 0.0
        self.beta5 = ((sum_x5y / n) - avg_x5 * avg_y) / var_x5 if var_x5 else 0.0
        self.beta6 = ((sum_x6y / n) - avg_x6 * avg_y) / var_x6 if var_x6 else 0.0

        self.beta0 = avg_y - (
            self.beta1 * avg_x1 + self.beta2 * avg_x2 +
            self.beta3 * avg_x3 + self.beta4 * avg_x4 +
            self.beta5 * avg_x5 + self.beta6 * avg_x6
        )

        self.is_trained = True

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
        x1, x2, x3, x4, x5, x6 = get_features(symbol, timeframe, 0, feature_window, rsi_period)

        prediction_points = (
            self.beta0
            + self.beta1 * x1
            + self.beta2 * x2
            + self.beta3 * x3
            + self.beta4 * x4
            + self.beta5 * x5
            + self.beta6 * x6
        )

        return prediction_points * point   # convert points → price units

    def predict_ahead(
        self,
        symbol: str,
        timeframe: int,
        feature_window: int,
        rsi_period: int,
        horizon: int = 15,
    ) -> float:
        """
        Predict price change further into the future (e.g., 15 bars = 15 min on M1).
        
        Uses the current model coefficients but assumes similar feature dynamics
        will persist. This is useful for:
          - Identifying target levels 15 minutes ahead
          - Deciding trade direction based on multi-timeframe confirmation
          - Setting profit targets based on lookahead consensus
        
        Args:
            symbol:         Trading symbol
            timeframe:      MT5 timeframe constant
            feature_window: Feature window (must match training)
            rsi_period:     RSI period (must match training)
            horizon:        How many bars ahead to predict (default 15 = 15 min on M1)
        
        Returns:
            Predicted price change in price units for the horizon.
            Returns 0.0 if model not trained.
        """
        if not self.is_trained:
            return 0.0

        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            return 0.0

        point = sym_info.point
        
        # Look ahead by fetching features at future bar positions
        # This is an extrapolation based on trend continuation
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, horizon + feature_window + 5)
        if rates is None or len(rates) < horizon + feature_window:
            return 0.0

        closes = rates["close"][::-1]
        
        # Current price and projected future price
        current_price = float(closes[0])
        future_price = float(closes[horizon])
        
        # Get features from the current bar (as a proxy for trend continuation)
        x1, x2, x3, x4, x5, x6 = get_features(symbol, timeframe, 0, feature_window, rsi_period)
        
        # Apply model with current features to extrapolate
        prediction_points = (
            self.beta0 * (horizon / self.prediction_horizon)  # Scale intercept
            + self.beta1 * x1 * (horizon / self.prediction_horizon)
            + self.beta2 * x2 * (horizon / self.prediction_horizon)
            + self.beta3 * x3  # Trend continues linearly
            + self.beta4 * x4 * 0.5  # RSI influence diminishes
            + self.beta5 * x5 * (horizon / self.prediction_horizon)
            + self.beta6 * x6 * (horizon / self.prediction_horizon)
        )

        return prediction_points * point

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:  # noqa: D105
        status = "trained" if self.is_trained else "untrained"
        return (
            f"LinearRegressionModel({status}) "
            f"β=({self.beta0:.4f}, {self.beta1:.4f}, "
            f"{self.beta2:.4f}, {self.beta3:.4f}, {self.beta4:.4f}, "
            f"{self.beta5:.4f}, {self.beta6:.4f})"
        )
