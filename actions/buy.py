import MetaTrader5 as mt5
from mt5_tool.order import place_order

def order_buy(symbol, volume, price=None, deviation=20):
    """Place a buy order"""

    return place_order(
        symbol=symbol,
        volume=volume,
        order_type=mt5.ORDER_TYPE_BUY,
        price=price,
        deviation=deviation,
    )
    
def buy_market(symbol, volume, deviation=20):
    """Place a market buy order"""
    return order_buy(symbol, volume, price=None, deviation=deviation)

def buy_stop(symbol, volume, stop_price, deviation=20):
    """Place a BUY STOP pending order that triggers at stop_price
    
    Note: This may not work on all brokers/symbols.
    Alternative: Use buy_market() and manually set stop loss in MT5
    
    Args:
        symbol: Trading symbol
        volume: Order volume
        stop_price: Price where the order should trigger
        deviation: Maximum price deviation
    """
    return place_order(
        symbol=symbol,
        volume=volume,
        order_type=mt5.ORDER_TYPE_BUY_STOP,
        price=stop_price,
        deviation=deviation,
    )