import MetaTrader5 as mt5
from actions.buy import buy_market
from actions.sell import sell_market
from actions.connection import initialize_connection
from mt5_tool.symbol import get_symbol, stream_ticks
from tools.print import pretty_print

if __name__ == "__main__":
    initialize_connection()

    # Pretty print account info
    pretty_print(mt5.account_info(), "Account Info")

    # Get correct symbol name
    symbol = get_symbol("BTCUSD")
    
    # Note: BUY STOP pending orders may not work on all brokers/symbols
    # If needed, manually set stop loss in MT5 through the platform UI
    # buy_stop(symbol, volume=0.01, stop_price=80000)
    # sell_stop(symbol, volume=0.01, stop_price=70000)
    
    if symbol:
        stream_ticks(symbol, interval=1)

    mt5.shutdown()