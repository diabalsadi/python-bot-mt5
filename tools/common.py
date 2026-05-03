from datetime import datetime

def format_time(ms):
    """Convert milliseconds timestamp to dd/mm/yyyy hh:mm:ss"""
    dt = datetime.fromtimestamp(ms / 1000.0)
    return dt.strftime("%d/%m/%Y %H:%M:%S")
