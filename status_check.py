#!/usr/bin/env python
"""
Bot Status Check - Verify all systems are ready
"""

import sys

def check_imports():
    """Check all critical imports."""
    print("=" * 60)
    print("DEPENDENCY CHECK")
    print("=" * 60)
    
    packages = [
        ("MetaTrader5", "mt5"),
        ("python-dotenv", "dotenv"),
        ("numpy", "numpy"),
        ("xgboost", "xgboost"),
        ("scikit-learn", "sklearn"),
    ]
    
    all_ok = True
    for pkg_name, import_name in packages:
        try:
            mod = __import__(import_name)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✅ {pkg_name:20} {version}")
        except ImportError as e:
            print(f"❌ {pkg_name:20} — {str(e)}")
            all_ok = False
    
    return all_ok


def check_modules():
    """Check project modules."""
    print("\n" + "=" * 60)
    print("PROJECT MODULES CHECK")
    print("=" * 60)
    
    modules = [
        ("main", "from main import _trade_logger, model"),
        ("position_manager", "from position_manager import get_position_manager"),
        ("backtest", "from backtest import Backtester"),
        ("test_suite", "from tests.test_trading_logic import run_tests"),
    ]
    
    all_ok = True
    for mod_name, import_stmt in modules:
        try:
            exec(import_stmt)
            print(f"✅ {mod_name:20} loaded successfully")
        except Exception as e:
            print(f"❌ {mod_name:20} — {str(e)[:50]}")
            all_ok = False
    
    return all_ok


def check_unit_tests():
    """Run unit tests."""
    print("\n" + "=" * 60)
    print("UNIT TESTS")
    print("=" * 60)
    
    try:
        from tests.test_trading_logic import run_tests
        success = run_tests()
        return success
    except Exception as e:
        print(f"❌ Tests failed: {e}")
        return False


def main():
    """Run all checks."""
    print("\n🤖 GOLD SCALPER BOT - STATUS CHECK\n")
    
    deps_ok = check_imports()
    modules_ok = check_modules()
    
    print("\n" + "=" * 60)
    print("FINAL STATUS")
    print("=" * 60)
    
    if deps_ok and modules_ok:
        print("✅ All systems operational")
        print("\n📋 Next steps:")
        print("   1. Run backtest: python backtest.py")
        print("   2. Paper trade: python main.py")
        print("   3. Monitor: tail -f latency.log")
        return 0
    else:
        print("❌ Some systems need attention")
        return 1


if __name__ == "__main__":
    sys.exit(main())
