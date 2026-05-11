import MetaTrader5 as mt5
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get credentials from environment variables
MT5_PATH = os.getenv("MT5_PATH")
LOGIN = int(os.getenv("LOGIN"))
PASSWORD = os.getenv("PASSWORD")
SERVER = os.getenv("SERVER")

print(f"MT5 Path: {MT5_PATH}")
print(f"Login:    {LOGIN}")
print(f"Server:   {SERVER}")
# Password intentionally not printed


def initialize_connection():
    print("Initializing connection...")

    if not mt5.initialize(
        path=MT5_PATH,
        login=LOGIN,
        server=SERVER,
        password=PASSWORD
    ):
        print("❌ initialize() failed:", mt5.last_error())
        quit()

    print("✅ Connected to MT5")
