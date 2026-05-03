
from pprint import pprint


def pretty_print(obj, title=None):
    """Convert MT5 objects to dict and pretty print"""
    if title:
        print(f"\n📌 {title}")

    if obj is None:
        print("❌ None")
        return

    try:
        pprint(obj._asdict())
    except AttributeError:
        pprint(obj)

def print_tick_pretty(tick_dict):
    print(
        f"""
🟢 TICK UPDATE
━━━━━━━━━━━━━━━━━━━━━━━━━━
⏰ Time   : {tick_dict['time_readable']}
💰 Bid    : {tick_dict['bid']}
💰 Ask    : {tick_dict['ask']}
📊 Spread : {round(tick_dict['ask'] - tick_dict['bid'], 2)}
📦 Volume : {tick_dict['volume_real']}
━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
    )