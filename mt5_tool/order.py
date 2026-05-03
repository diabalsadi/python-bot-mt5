import MetaTrader5 as mt5


def place_order(symbol, volume, order_type, price=None, deviation=20):
    """Place an order on MT5"""
    
    # Get symbol info
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"❌ Symbol {symbol} not found")
        return None
    
    print(f"📋 Symbol Info: bid={symbol_info.bid}, ask={symbol_info.ask}, digits={symbol_info.digits}")
    print(f"   Trade mode: {symbol_info.trade_mode}")
    
    # Check if this is a pending order (stop/limit)
    is_pending_order = order_type in [mt5.ORDER_TYPE_BUY_STOP, mt5.ORDER_TYPE_SELL_STOP, 
                                       mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT]
    
    # For market orders, use current bid/ask price
    if price is None and not is_pending_order:
        if order_type == mt5.ORDER_TYPE_BUY:
            price = symbol_info.ask
        else:
            price = symbol_info.bid
    
    # Round price to symbol's decimal precision
    if price is not None:
        price = round(price, symbol_info.digits)
    
    # Build request based on order type
    if is_pending_order:
        # Pending orders (stop/limit)
        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "type_time": mt5.ORDER_TIME_GTC,
            "comment": f"Pending order"
        }
    else:
        # Market orders
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price if price else symbol_info.ask,
            "deviation": deviation,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
            "comment": f"Market order"
        }
    
    order_type_names = ["BUY", "SELL", "BUY LIMIT", "SELL LIMIT", "BUY STOP", "SELL STOP"]
    order_type_name = order_type_names[order_type] if order_type < len(order_type_names) else f"TYPE_{order_type}"
    print(f"📤 Sending {order_type_name} order at {price if price else 'market'}: {request}")
    result = mt5.order_send(request)
    
    if result is None:
        print(f"❌ Order send failed: No response from MT5")
        print(f"MT5 Error: {mt5.last_error()}")
        return None
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Order failed: {result.comment}")
        print(f"Return code: {result.retcode}")
        print(f"Full result: {result}")
        return None
    
    print(f"✅ Order placed: {result.order}")
    return result
