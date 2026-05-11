"""
MT5 Connection Manager
-----------------------
Handles initial connection and automatic reconnection if the terminal
drops mid-session. Call ensure_connected() at the top of on_tick()
instead of relying on a one-shot initialize().
"""

import time
import MetaTrader5 as mt5
import os
from dotenv import load_dotenv

load_dotenv()

MT5_PATH = os.getenv("MT5_PATH")
LOGIN    = int(os.getenv("LOGIN"))
PASSWORD = os.getenv("PASSWORD")
SERVER   = os.getenv("SERVER")

print(f"MT5 Path: {MT5_PATH}")
print(f"Login:    {LOGIN}")
print(f"Server:   {SERVER}")
# Password intentionally not printed

# ── Reconnection state ────────────────────────────────────────────────
_connected: bool = False
_last_reconnect_attempt: float = 0.0
_RECONNECT_COOLDOWN_SECS: float = 15.0   # don't hammer the terminal
_MAX_RETRIES: int = 5


def initialize_connection() -> None:
    """One-shot startup connection. Exits the process on failure."""
    global _connected
    print("Initializing connection...")
    if not _try_connect():
        print("❌ initialize() failed:", mt5.last_error())
        quit()
    _connected = True
    print("✅ Connected to MT5")


def ensure_connected() -> bool:
    """
    Call at the top of every on_tick().
    Returns True if connected (or successfully reconnected).
    Returns False if still disconnected — caller should skip the tick.
    """
    global _connected, _last_reconnect_attempt

    # Fast path — terminal reports it's fine
    if mt5.terminal_info() is not None and mt5.account_info() is not None:
        _connected = True
        return True

    # Cooldown to avoid hammering
    now = time.time()
    if now - _last_reconnect_attempt < _RECONNECT_COOLDOWN_SECS:
        return False

    _last_reconnect_attempt = now
    print("⚠️  MT5 connection lost — attempting reconnect…")

    for attempt in range(1, _MAX_RETRIES + 1):
        mt5.shutdown()
        time.sleep(2)
        if _try_connect():
            _connected = True
            print(f"✅ Reconnected to MT5 (attempt {attempt})")
            return True
        print(f"   Attempt {attempt}/{_MAX_RETRIES} failed: {mt5.last_error()}")

    print(f"❌ Could not reconnect after {_MAX_RETRIES} attempts — skipping tick")
    _connected = False
    return False


def _try_connect() -> bool:
    """Low-level connect attempt. Returns True on success."""
    return mt5.initialize(
        path=MT5_PATH,
        login=LOGIN,
        server=SERVER,
        password=PASSWORD,
    )
