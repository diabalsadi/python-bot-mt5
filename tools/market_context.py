import MetaTrader5 as mt5
from datetime import datetime, timedelta
import json
import requests
import logging

def get_recent_performance(symbol, days=3):
    """
    Fetch recent trade history for the symbol to provide context on current performance.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Fetch history
    history = mt5.history_deals_get(start_date, end_date, group=f"*{symbol}*")
    if history is None or len(history) == 0:
        return {"summary": "No recent trades found.", "trades": []}
    
    trades = []
    total_profit = 0
    wins = 0
    losses = 0
    
    # Sort by time descending
    sorted_history = sorted(history, key=lambda x: x.time, reverse=True)
    
    for deal in sorted_history[:10]: # Look at last 10 deals
        # 1 = DEAL_ENTRY_OUT (Closing a position)
        if deal.entry == 1:
            profit = deal.profit + deal.commission + deal.swap
            total_profit += profit
            if profit > 0: wins += 1
            else: losses += 1
            
            trades.append({
                "time": datetime.fromtimestamp(deal.time).strftime("%H:%M:%S"),
                "type": "BUY" if deal.type == 1 else "SELL", # Note: 0 is Buy deal, 1 is Sell deal in MT5 history
                "profit": round(profit, 2),
                "reason": "Unknown" # MT5 doesn't always store 'reason' in deals
            })
            
    win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0
    
    return {
        "summary": f"Last {len(trades)} trades: {wins} Wins, {losses} Losses | Win Rate: {win_rate:.1f}% | Total Net: {total_profit:.2f}",
        "trades": trades
    }

def get_market_snapshot(symbol, timeframe, model, model_m5, long_model, config):
    """
    Generate a comprehensive snapshot of the current market state.
    """
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return None

    # 1. Basic Price Data
    snapshot = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
        "price": {
            "bid": tick.bid,
            "ask": tick.ask,
            "spread": (tick.ask - tick.bid) / mt5.symbol_info(symbol).point
        }
    }

    # 2. Correlation Data
    correlation_symbol = None
    corr_type = "inverse"
    if "XAUUSD" in symbol:
        correlation_symbol = "USDCHFm"
        corr_type = "inverse"
    elif "BTCUSD" in symbol:
        correlation_symbol = "USTEC_x100m"
        corr_type = "positive"
        
    if correlation_symbol:
        corr_tick = mt5.symbol_info_tick(correlation_symbol)
        if corr_tick:
            rates = mt5.copy_rates_from_pos(correlation_symbol, mt5.TIMEFRAME_D1, 0, 1)
            prev_close = rates[0]['close'] if rates is not None else corr_tick.bid
            daily_change_pct = ((corr_tick.bid - prev_close) / prev_close) * 100
            
            snapshot["correlation"] = {
                "symbol": correlation_symbol,
                "price": corr_tick.bid,
                "daily_change_pct": round(daily_change_pct, 4),
                "type": corr_type,
                "sentiment": "Strength" if daily_change_pct > 0 else "Weakness"
            }

    # 3. Technical Indicators
    from indicators.support_resistance import identify_sr_levels
    from indicators.liquidity_zones import get_liquidity_zones
    supports, resistances = identify_sr_levels(symbol, timeframe, config['SR_LOOKBACK_BARS'])
    zones = get_liquidity_zones(symbol, timeframe, config['SR_LOOKBACK_BARS'])
    
    snapshot["technical"] = {
        "supports": supports[:3],
        "resistances": resistances[:3],
        "liquidity_zones": [
            {"level": z["level"], "type": z["type"], "strength": round(z["strength"], 2)}
            for z in zones[:5]
        ]
    }

    # 4. ML Signals (Multi-Timeframe)
    pred_m1 = model.predict(symbol, timeframe, config['ML_FEATURE_WINDOW'], config['RSI_PERIOD'])
    pred_m5 = model_m5.predict(symbol, mt5.TIMEFRAME_M5, config['ML_FEATURE_WINDOW'], config['RSI_PERIOD'])
    pred_h1 = long_model.predict(symbol, config['LONG_TIMEFRAME'], config['ML_FEATURE_WINDOW'], config['RSI_PERIOD'])
    
    snapshot["ml_signals"] = {
        "m1_prediction": round(pred_m1, 5),
        "m5_prediction": round(pred_m5, 5),
        "h1_prediction": round(pred_h1, 5),
        "alignment": "full_alignment" if (pred_m1 * pred_m5 * pred_h1) > 0 else "mixed"
    }

    # 5. OpenBB News
    news_search_symbol = "XAU" if "XAU" in symbol else "BTC"
    snapshot["news"] = []
    try:
        news_resp = requests.get(f"http://localhost:6900/api/v1/news/company?symbol={news_search_symbol}&limit=3", timeout=2)
        if news_resp.status_code == 200:
            news_data = news_resp.json()
            if "results" in news_data:
                for item in news_data["results"]:
                    snapshot["news"].append({
                        "title": item.get("title", ""),
                        "summary": item.get("text", "")[:200],
                        "date": item.get("date", "")
                    })
    except Exception as e:
        logging.warning(f"Failed to fetch news from OpenBB for {news_search_symbol}: {e}")

    # 6. Performance History (LEARNING FROM LOGS)
    snapshot["performance"] = get_recent_performance(symbol)

    # 7. Account & Positions
    acc = mt5.account_info()
    positions = mt5.positions_get(symbol=symbol)
    snapshot["account"] = {
        "balance": acc.balance,
        "equity": acc.equity,
        "margin_free": acc.margin_free
    }
    
    pos_list = []
    if positions:
        for p in positions:
            pos_list.append({
                "ticket": p.ticket,
                "type": "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL",
                "profit": p.profit,
                "lot": p.volume,
                "sl": p.sl,
                "tp": p.tp
            })
    snapshot["open_positions"] = pos_list

    return snapshot

def format_snapshot_for_llm(snapshot):
    """Convert JSON snapshot to a clear text prompt segment."""
    if not snapshot:
        return "No market data available."
        
    text = f"--- MARKET SNAPSHOT | {snapshot['timestamp']} ---\n"
    text += f"Asset: {snapshot['symbol']} | Bid: {snapshot['price']['bid']} | Ask: {snapshot['price']['ask']}\n"
    
    if "performance" in snapshot:
        perf = snapshot["performance"]
        text += f"\n[RECENT LOGS & PERFORMANCE]\n"
        text += f"Summary: {perf['summary']}\n"
        if perf['trades']:
            text += "Recent Outcomes: " + ", ".join([f"{t['type']}({t['profit']})" for t in perf['trades'][:5]]) + "\n"

    if "correlation" in snapshot:
        c = snapshot["correlation"]
        rel = "inversely" if c['type'] == "inverse" else "positively"
        text += f"\n[Correlation Alert]\n{snapshot['symbol']} is {rel} correlated with {c['symbol']}.\n"
        text += f"{c['symbol']} is at {c['price']} ({c['daily_change_pct']}%). State: {c['sentiment']}\n"
    
    text += f"\n[Multi-Timeframe ML Signals]\n"
    text += f"M1 (Entry): {snapshot['ml_signals']['m1_prediction']}\n"
    text += f"M5 (Bridge): {snapshot['ml_signals']['m5_prediction']}\n"
    text += f"H1 (Macro): {snapshot['ml_signals']['h1_prediction']}\n"
    text += f"Alignment: {snapshot['ml_signals']['alignment']}\n"
    
    text += f"\n[Technical Landscape]\n"
    text += f"Supports: {', '.join([str(s) for s in snapshot['technical']['supports']])}\n"
    text += f"Resistances: {', '.join([str(r) for r in snapshot['technical']['resistances']])}\n"
    
    if snapshot["news"]:
        text += f"\n[Market News for {snapshot['symbol']}]\n"
        for n in snapshot["news"]:
            text += f"- {n['title']}\n"
    
    text += f"\n[Current Positions]\n"
    if not snapshot['open_positions']:
        text += "None\n"
    else:
        for p in snapshot['open_positions']:
            text += f"#{p['ticket']} {p['type']} | Profit: {p['profit']} | Lot: {p['lot']}\n"
            
    text += f"\n[Account State]\n"
    text += f"Equity: {snapshot['account']['equity']} | Free Margin: {snapshot['account']['margin_free']}\n"
    
    return text
