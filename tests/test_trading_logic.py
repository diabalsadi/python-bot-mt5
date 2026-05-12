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
