import MetaTrader5 as mt5
import time
from tools.common import format_time
from tools.print import print_tick_pretty

def get_symbol(symbol_base):
    print(f"Searching for symbol like '{symbol_base}'...")

    symbols = mt5.symbols_get()
    for s in symbols:
        if symbol_base in s.name:
            print(f"✅ Found symbol: {s.name}")
            return s.name

    print("❌ No matching symbol found")
    return None


def stream_ticks(symbol, interval=1):
    """
    Stream and log ticks continuously
    """
    if not mt5.symbol_select(symbol, True):
        print(f"❌ Failed to select symbol: {symbol}")
        return

    print(f"\n🚀 Streaming ticks for {symbol} (Ctrl+C to stop)\n")

    last_time = None

    try:
        while True:
            tick = mt5.symbol_info_tick(symbol)

            if tick is None:
                print("❌ No tick data")
            else:
                tick_dict = tick._asdict()

                # Avoid duplicate ticks
                if tick_dict["time_msc"] != last_time:
                    last_time = tick_dict["time_msc"]

                    # ✅ Format time
                    tick_dict["time_readable"] = format_time(tick_dict["time_msc"])

                    print_tick_pretty(tick_dict)

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n🛑 Stopped streaming")