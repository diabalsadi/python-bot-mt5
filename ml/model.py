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

        # ── Build feature matrix X and target vector y ────────────────
        rows = []
        ys = []
        for i in range(prediction_horizon, n + prediction_horizon):
            x1, x2, x3, x4, x5, x6 = get_features(symbol, timeframe, i, feature_window, rsi_period)
            current_price = closes[i]
            future_price  = closes[i - prediction_horizon]
            y = (future_price - current_price) / point
            rows.append([1.0, x1, x2, x3, x4, x5, x6])
            ys.append(y)

        X = np.array(rows, dtype=np.float64)
        y = np.array(ys,  dtype=np.float64)

        # ── Joint OLS via least-squares (handles correlated features) ──
        betas, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

        self.beta0 = float(betas[0])
        self.beta1 = float(betas[1])
        self.beta2 = float(betas[2])
        self.beta3 = float(betas[3])
        self.beta4 = float(betas[4])
        self.beta5 = float(betas[5])
        self.beta6 = float(betas[6])

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
        Extrapolate predicted price change `horizon` bars ahead.

        Uses only the current bar's features and the trained coefficients —
        no future OHLCV data is read (which would introduce look-ahead bias).
        Momentum and trend continue linearly; RSI influence decays with time;
        volatility and structural features (S/R, liquidity) scale with horizon.

        Args:
            symbol:         Trading symbol
            timeframe:      MT5 timeframe constant
            feature_window: Feature window (must match training)
            rsi_period:     RSI period (must match training)
            horizon:        How many bars ahead to predict (default 15)

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

        x1, x2, x3, x4, x5, x6 = get_features(symbol, timeframe, 0, feature_window, rsi_period)

        h_ratio = horizon / max(1, self.prediction_horizon)

        # Trend (x3) scales linearly with horizon; RSI (x4) influence halves;
        # momentum, vol, S/R, liquidity all scale with h_ratio.
        prediction_points = (
            self.beta0 * h_ratio
            + self.beta1 * x1 * h_ratio        # momentum
            + self.beta2 * x2 * h_ratio        # volatility
            + self.beta3 * x3 * h_ratio        # trend (linear continuation)
            + self.beta4 * x4 * 0.5            # RSI decays
            + self.beta5 * x5 * h_ratio        # S/R distance
            + self.beta6 * x6 * h_ratio        # liquidity
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
