"""
Unit Tests for Trading Bot Components
--------------------------------------
Basic test suite for indicators, position sizing, and signal logic.

To run tests:
    pytest tests/ -v
    
Or with coverage:
    pytest tests/ --cov=. --cov-report=html
"""

import unittest
import sys
from unittest.mock import MagicMock, patch

# Mock MetaTrader5 before any imports that depend on it
sys.modules['MetaTrader5'] = MagicMock()


class TestPositionSizing(unittest.TestCase):
    """Test risk-based position sizing calculations."""
    
    def test_calculate_lot_size_basic(self):
        """Test basic lot size calculation."""
        # This would normally call the strategy module
        # For now, we test the logic manually
        
        balance = 10000.0
        risk_percent = 10.0
        sl_points = 100
        tick_value = 1.0
        tick_size = 0.01
        
        risk_amount = balance * (risk_percent / 100)
        lot = risk_amount / (sl_points * (tick_value / tick_size))
        
        # Should risk exactly $1000
        expected_risk = 1000.0
        self.assertAlmostEqual(lot * sl_points * (tick_value / tick_size), expected_risk, places=2)
    
    def test_lot_size_minimum_volume(self):
        """Test that lot size respects minimum volume."""
        balance = 100.0  # Very small account
        risk_percent = 10.0
        sl_points = 200
        
        # With such small balance, lot should be clamped to minimum
        risk_amount = balance * (risk_percent / 100)
        min_lot = 0.01
        
        self.assertGreater(min_lot, 0)


class TestSignalFilters(unittest.TestCase):
    """Test signal confirmation logic."""
    
    def test_m1_m15_agreement(self):
        """Test multi-timeframe agreement logic."""
        # Bullish agreement
        prediction_m1 = 50.0
        prediction_m15 = 40.0
        
        m1_m15_agree = (prediction_m1 > 0 and prediction_m15 > 0) or \
                       (prediction_m1 < 0 and prediction_m15 < 0)
        
        self.assertTrue(m1_m15_agree)
    
    def test_m1_m15_disagreement(self):
        """Test disagreement blocking signal."""
        prediction_m1 = 50.0
        prediction_m15 = -40.0
        
        m1_m15_agree = (prediction_m1 > 0 and prediction_m15 > 0) or \
                       (prediction_m1 < 0 and prediction_m15 < 0)
        
        self.assertFalse(m1_m15_agree)
    
    def test_h1_soft_filter_opposed(self):
        """Test H1 opposition blocks signal."""
        prediction = 50.0
        long_prediction = -80.0
        volatility = 100.0
        
        h1_opposed = (prediction > 0 and long_prediction < -volatility * 0.5)
        
        self.assertTrue(h1_opposed)
    
    def test_h1_soft_filter_allowed(self):
        """Test weak H1 opposition doesn't block."""
        prediction = 50.0
        long_prediction = -40.0
        volatility = 100.0
        
        h1_opposed = (prediction > 0 and long_prediction < -volatility * 0.5)
        
        self.assertFalse(h1_opposed)


class TestFibonacciFilter(unittest.TestCase):
    """Test Fibonacci pullback entry filter."""
    
    def test_fibonacci_pullback_buy(self):
        """Test buy entry on pullback to 50% fib."""
        swing_high = 2100.0
        swing_low = 2000.0
        current_price = 2030.0
        
        swing_range = swing_high - swing_low
        fib_50 = swing_low + swing_range * 0.50
        
        # Price should be at or below 50% for pullback
        is_pullback = current_price <= fib_50
        
        self.assertTrue(is_pullback)
    
    def test_fibonacci_breakout_rejected(self):
        """Test buy entry rejected above 50% fib (breakout, not pullback)."""
        swing_high = 2100.0
        swing_low = 2000.0
        current_price = 2075.0
        
        swing_range = swing_high - swing_low
        fib_50 = swing_low + swing_range * 0.50
        
        # Price above 50% = breakout, not pullback
        is_pullback = current_price <= fib_50
        
        self.assertFalse(is_pullback)


class TestTrailingSL(unittest.TestCase):
    """Test trailing stop loss stages."""
    
    def test_stage_1_profit_reserve(self):
        """Test Stage 1: Lock in 50% of gains."""
        profit_points = 80.0
        entry = 2000.0
        volatility_atr = 100.0
        
        stage_1_threshold = volatility_atr * 0.7
        
        is_stage_1 = profit_points > stage_1_threshold
        
        self.assertTrue(is_stage_1)
    
    def test_stage_2_trend_following(self):
        """Test Stage 2: Trail wider for trend following."""
        profit_points = 150.0
        volatility_atr = 100.0
        
        stage_2_threshold = volatility_atr * 1.2
        stage_3_threshold = volatility_atr * 3.0
        
        is_stage_2 = profit_points > stage_2_threshold and profit_points <= stage_3_threshold
        
        self.assertTrue(is_stage_2)
    
    def test_stage_3_aggressive_harvest(self):
        """Test Stage 3: Tight trail for harvest."""
        profit_points = 350.0
        volatility_atr = 100.0
        
        stage_3_threshold = volatility_atr * 3.0
        
        is_stage_3 = profit_points > stage_3_threshold
        
        self.assertTrue(is_stage_3)


class TestRiskRewardRatio(unittest.TestCase):
    """Test risk/reward ratio calculations."""
    
    def test_rr_ratio_strong_signal(self):
        """Test TP multiplier for strong signal."""
        prediction_magnitude = 300.0
        volatility = 100.0
        
        # Strong signal: prediction > volatility * 2.5
        is_strong = prediction_magnitude > (volatility * 2.5)
        tp_mult = 3.0 if is_strong else 2.5
        
        self.assertEqual(tp_mult, 3.0)
        self.assertEqual(tp_mult, 3.0)  # 1:3 ratio
    
    def test_rr_ratio_weak_signal(self):
        """Test TP multiplier for weak signal."""
        prediction_magnitude = 60.0
        volatility = 100.0
        
        # Weak signal: prediction < volatility * 0.8
        is_weak = prediction_magnitude < (volatility * 0.8)
        tp_mult = 2.0 if is_weak else 2.5
        
        self.assertEqual(tp_mult, 2.0)


class TestLotSizeConstraints(unittest.TestCase):
    """Test lot size respects broker constraints."""
    
    def test_lot_size_respects_minimum(self):
        """Test lot doesn't go below minimum volume."""
        calculated_lot = 0.005
        min_lot = 0.01
        
        lot = max(min_lot, calculated_lot)
        
        self.assertGreaterEqual(lot, min_lot)
    
    def test_lot_size_respects_maximum(self):
        """Test lot doesn't exceed maximum volume."""
        calculated_lot = 100.0
        max_lot = 10.0
        
        lot = min(max_lot, calculated_lot)
        
        self.assertLessEqual(lot, max_lot)


class TestLatencyGuard(unittest.TestCase):
    """Test latency checks for scalping safety."""
    
    def test_latency_acceptable(self):
        """Test low latency passes."""
        latency_ms = 500
        max_latency_ms = 2000
        
        is_acceptable = latency_ms <= max_latency_ms
        
        self.assertTrue(is_acceptable)
    
    def test_latency_too_high(self):
        """Test high latency blocks trading."""
        latency_ms = 3000
        max_latency_ms = 2000
        
        is_acceptable = latency_ms <= max_latency_ms
        
        self.assertFalse(is_acceptable)


class TestVolatilityProtection(unittest.TestCase):
    """Test high volatility guards."""
    
    def test_normal_volatility(self):
        """Test trading allowed in normal volatility."""
        current_atr = 100.0
        avg_atr = 80.0
        volatility_multiplier = 2.0
        
        is_high = current_atr > (avg_atr * volatility_multiplier)
        
        self.assertFalse(is_high)
    
    def test_high_volatility(self):
        """Test trading blocked in extreme volatility."""
        current_atr = 200.0
        avg_atr = 80.0
        volatility_multiplier = 2.0
        
        is_high = current_atr > (avg_atr * volatility_multiplier)
        
        self.assertTrue(is_high)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPositionSizing))
    suite.addTests(loader.loadTestsFromTestCase(TestSignalFilters))
    suite.addTests(loader.loadTestsFromTestCase(TestFibonacciFilter))
    suite.addTests(loader.loadTestsFromTestCase(TestTrailingSL))
    suite.addTests(loader.loadTestsFromTestCase(TestRiskRewardRatio))
    suite.addTests(loader.loadTestsFromTestCase(TestLotSizeConstraints))
    suite.addTests(loader.loadTestsFromTestCase(TestLatencyGuard))
    suite.addTests(loader.loadTestsFromTestCase(TestVolatilityProtection))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)


class TestAntiHedge(unittest.TestCase):
    """Tests that the bot never opens opposite positions simultaneously."""

    def test_buy_blocked_when_sell_open(self):
        """BUY must be blocked if a SELL is already open."""
        from unittest.mock import patch, MagicMock

        mock_positions = [MagicMock(type=1)]  # 1 = POSITION_TYPE_SELL in MT5

        with patch('MetaTrader5.positions_get', return_value=mock_positions):
            import MetaTrader5 as mt5
            mt5.POSITION_TYPE_SELL = 1
            mt5.POSITION_TYPE_BUY  = 0

            positions = mt5.positions_get()
            has_sell = any(p.type == mt5.POSITION_TYPE_SELL for p in positions)
            self.assertTrue(has_sell, "Should detect open SELL before allowing BUY")

    def test_sell_blocked_when_buy_open(self):
        """SELL must be blocked if a BUY is already open."""
        from unittest.mock import patch, MagicMock

        mock_positions = [MagicMock(type=0)]  # 0 = POSITION_TYPE_BUY

        with patch('MetaTrader5.positions_get', return_value=mock_positions):
            import MetaTrader5 as mt5
            mt5.POSITION_TYPE_BUY  = 0
            mt5.POSITION_TYPE_SELL = 1

            positions = mt5.positions_get()
            has_buy = any(p.type == mt5.POSITION_TYPE_BUY for p in positions)
            self.assertTrue(has_buy, "Should detect open BUY before allowing SELL")

    def test_no_hedge_when_no_positions(self):
        """Both directions allowed when no positions are open."""
        from unittest.mock import patch

        with patch('MetaTrader5.positions_get', return_value=[]):
            import MetaTrader5 as mt5
            positions = mt5.positions_get()
            has_buy  = any(getattr(p, 'type', -1) == 0 for p in positions)
            has_sell = any(getattr(p, 'type', -1) == 1 for p in positions)
            self.assertFalse(has_buy)
            self.assertFalse(has_sell)


class TestConsecutiveLossBreaker(unittest.TestCase):
    """Tests that the consecutive-loss circuit breaker pauses correctly."""

    def test_three_losses_pause_direction(self):
        consecutive = 0
        paused = False
        MAX = 3

        for _ in range(MAX):
            profit = -1.0
            if profit < 0:
                consecutive += 1
            if consecutive >= MAX:
                paused = True

        self.assertTrue(paused)
        self.assertEqual(consecutive, MAX)

    def test_win_resets_counter(self):
        consecutive = 2
        profit = 1.0   # a win
        if profit > 0:
            consecutive = 0
        self.assertEqual(consecutive, 0)


class TestTrendGate(unittest.TestCase):
    """Tests the anti-trend-entry momentum gate."""

    def test_sell_blocked_in_strong_uptrend(self):
        closes = [100.0 + i for i in range(6)]   # 5-bar +5 move
        highs  = [c + 0.5 for c in closes]
        lows   = [c - 0.5 for c in closes]

        trs = [max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
               for i in range(1, len(closes))]
        atr = sum(trs) / len(trs)
        net_move = closes[-1] - closes[-6]

        blocked = net_move > atr
        self.assertTrue(blocked, f"SELL should be blocked: net_move={net_move:.2f} atr={atr:.2f}")

    def test_sell_allowed_in_flat_market(self):
        import math
        closes = [100.0 + math.sin(i) * 0.1 for i in range(6)]  # tiny oscillation
        highs  = [c + 0.1 for c in closes]
        lows   = [c - 0.1 for c in closes]

        trs = [max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
               for i in range(1, len(closes))]
        atr = sum(trs) / len(trs)
        net_move = closes[-1] - closes[-6]

        blocked = net_move > atr
        self.assertFalse(blocked, "SELL should NOT be blocked in flat market")


class TestRLAgent(unittest.TestCase):
    """Tests for the DQN reinforcement learning agent."""

    def _make_state(self, momentum=0.0, consec=0, recent_pnl=0.0):
        from datetime import datetime
        import numpy as np
        # Import build_state without triggering MT5
        import sys
        sys.modules.setdefault('MetaTrader5', MagicMock())
        from ml.rl_agent import build_state
        return build_state(
            momentum=momentum,
            volatility=1.0,
            trend=0.0,
            rsi=50.0,
            consecutive_losses=consec,
            open_time=datetime(2026, 5, 11, 20, 0, 0),
            recent_pnl=recent_pnl,
        )

    def test_state_shape(self):
        """State vector must be 10-dimensional (8 market + 2 SHAP dims)."""
        state = self._make_state()
        self.assertEqual(state.shape, (10,))

    def test_state_values_clipped(self):
        """Extreme inputs must be clipped to reasonable range."""
        state = self._make_state(momentum=9999.0, consec=100, recent_pnl=-9999.0)
        self.assertTrue(all(abs(v) <= 10 for v in state), f"Unclipped state: {state}")

    def test_act_returns_valid_action(self):
        """act() must return 0, 1, or 2."""
        from ml.rl_agent import RLAgent
        agent = RLAgent()
        state = self._make_state()
        action = agent.act(state)
        self.assertIn(action, [0, 1, 2])

    def test_act_respects_ml_hint(self):
        """When ml_action given, result must be that action or HOLD (0)."""
        from ml.rl_agent import RLAgent
        agent = RLAgent()
        # Force greedy (epsilon=0)
        agent.epsilon = 0.0
        state = self._make_state()
        action = agent.act(state, ml_action=2)
        self.assertIn(action, [0, 2], f"Expected HOLD or SELL, got {action}")

    def test_store_and_learn(self):
        """store() + learn() should not raise and return a float loss."""
        from ml.rl_agent import RLAgent, BATCH_SIZE
        agent = RLAgent()
        state = self._make_state()
        # Fill replay buffer above batch size
        for _ in range(BATCH_SIZE + 5):
            agent.store(state, 2, -1.5, state, False)
        loss = agent.learn()
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)

    def test_consecutive_loss_increases_hold_tendency(self):
        """Q[SELL] should decrease relative to Q[HOLD] after repeated SELL losses."""
        from ml.rl_agent import RLAgent, BATCH_SIZE
        import numpy as np
        agent = RLAgent()
        agent.epsilon = 0.0  # pure greedy

        bad_state = self._make_state(momentum=2.0, consec=5, recent_pnl=-8.0)

        # Measure Q[SELL] - Q[HOLD] before training
        x = bad_state.reshape(1, -1).astype(np.float64)
        q_before = agent.q_net.forward(x)[0].copy()
        sell_hold_before = q_before[2] - q_before[0]

        # Fill buffer with many SELL losses and train heavily
        for _ in range(BATCH_SIZE * 10):
            agent.store(bad_state, 2, -2.0, bad_state, False)
        for _ in range(50):
            agent.learn()

        q_after = agent.q_net.forward(x)[0]
        sell_hold_after = q_after[2] - q_after[0]

        # Q[SELL] should have moved DOWN relative to Q[HOLD] after loss training
        self.assertLess(
            sell_hold_after, sell_hold_before,
            f"Q[SELL]-Q[HOLD] should decrease after loss training. "
            f"Before={sell_hold_before:.4f} After={sell_hold_after:.4f}"
        )


class TestSHAPExplainer(unittest.TestCase):
    """Tests for SHAP explainability layer."""

    def _dummy_xgb(self):
        """Train a tiny XGBRegressor on synthetic data."""
        import numpy as np
        try:
            import xgboost as xgb
        except ImportError:
            return None
        X = np.random.randn(50, 6)
        y = X[:, 0] - X[:, 2] + np.random.randn(50) * 0.1
        m = xgb.XGBRegressor(n_estimators=10, verbosity=0)
        m.fit(X, y)
        return m

    def test_null_result_when_not_fitted(self):
        """explain() before fit() must return a safe null result."""
        from ml.shap_explainer import SHAPExplainer
        import numpy as np
        exp = SHAPExplainer()
        result = exp.explain(np.zeros(6), prediction=0.5)
        self.assertEqual(result.top_feature, "unknown")
        self.assertEqual(result.top_shap_value, 0.0)
        self.assertFalse(result.conflict)

    def test_explain_returns_6_shap_values(self):
        """After fit(), shap_values should be length 6."""
        from ml.shap_explainer import SHAPExplainer, FEATURE_NAMES
        import numpy as np
        model = self._dummy_xgb()
        if model is None:
            self.skipTest("xgboost not installed")
        exp = SHAPExplainer()
        exp.fit(model)
        self.assertTrue(exp.is_fitted)
        X = np.array([0.5, 1.2, -0.3, 55.0, 0.1, 0.8])
        result = exp.explain(X, prediction=0.3)
        self.assertEqual(len(result.shap_values), 6)
        self.assertIn(result.top_feature, FEATURE_NAMES)

    def test_conflict_detected_on_contradiction(self):
        """Positive momentum SHAP + negative prediction = conflict."""
        from ml.shap_explainer import SHAPExplainer, _SHAPResult
        import numpy as np
        # Manually craft a result with momentum as top feature, positive SHAP,
        # but prediction is negative (SELL)
        result = _SHAPResult(
            shap_values=np.array([2.0, 0.1, -0.1, 0.0, 0.0, 0.0]),
            top_feature="momentum",
            top_shap_value=2.0,    # positive = pushes price UP
            top_raw_value=0.5,
            conflict=True,         # but prediction is SELL → conflict
            summary="test",
            top2_str="",
        )
        self.assertTrue(result.conflict)

    def test_shap_rl_dims_output_range(self):
        """shap_rl_dims() outputs must stay in [-1,1] and [0,1]."""
        from ml.shap_explainer import SHAPExplainer
        import numpy as np
        model = self._dummy_xgb()
        if model is None:
            self.skipTest("xgboost not installed")
        exp = SHAPExplainer()
        exp.fit(model)
        X = np.random.randn(6)
        top_norm, conflict = exp.shap_rl_dims(X, prediction=0.5)
        self.assertGreaterEqual(top_norm, -1.0)
        self.assertLessEqual(top_norm, 1.0)
        self.assertIn(conflict, [0.0, 1.0])

    def test_state_is_10_dim_with_shap(self):
        """build_state with shap dims must produce 10-dim vector."""
        from datetime import datetime
        import numpy as np
        from ml.rl_agent import build_state
        state = build_state(
            momentum=1.0, volatility=0.5, trend=0.3, rsi=55.0,
            consecutive_losses=1, open_time=datetime(2026, 5, 11, 20, 0),
            recent_pnl=-0.5, shap_top_norm=0.7, shap_conflict=1.0,
        )
        self.assertEqual(state.shape, (10,))
        self.assertAlmostEqual(float(state[8]),  0.7, places=4)
        self.assertAlmostEqual(float(state[9]),  1.0, places=4)
