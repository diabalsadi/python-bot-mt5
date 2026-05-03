import MetaTrader5 as mt5
from mt5_tool.order import place_order

def order_sell(symbol, volume, price=None, deviation=20):
    """Place a sell order"""

    return place_order(
        symbol=symbol,
        volume=volume,
        order_type=mt5.ORDER_TYPE_SELL,
        price=price,
        deviation=deviation,
    )
    
def sell_market(symbol, volume, deviation=20):
    """Place a market sell order"""
    return order_sell(symbol, volume, price=None, deviation=deviation)

def sell_stop(symbol, volume, stop_price, deviation=20):
    """Place a SELL STOP pending order that triggers at stop_price
    
    Note: This may not work on all brokers/symbols.
    Alternative: Use sell_market() and manually set stop loss in MT5
    
    Args:
        symbol: Trading symbol
        volume: Order volume
        stop_price: Price where the order should trigger
        deviation: Maximum price deviation
    """
    return place_order(
        symbol=symbol,
        volume=volume,
        order_type=mt5.ORDER_TYPE_SELL_STOP,
        price=stop_price,
        deviation=deviation,
    )
