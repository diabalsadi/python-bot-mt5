# Python MetaTrader5 Tool

## Setup

### Prerequisites
- Python 3.8 or higher
- MetaTrader 5 installed on your system

### Installation

1. Clone or download this repository

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:

**On Windows (PowerShell):**
```bash
.venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```bash
.venv\Scripts\activate
```

**On macOS/Linux:**
```bash
source .venv/bin/activate
```

4. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root directory with your MetaTrader5 credentials:
```
LOGIN = your_login_number
PASSWORD = "your_password"
SERVER = "your_server_name"
MT5_PATH = "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
```

## Running the Project

To run the application, execute:
```bash
python main.py
```

This will:
- Connect to your MetaTrader5 account
- Display your account information
- Fetch the BTC/USD symbol
- Stream live tick data for the symbol
- Disconnect when done

## Project Structure
- `main.py` - Main entry point
- `actions/connection.py` - MT5 connection initialization
- `mt5_tool/index.py` - Symbol and data streaming tools
- `tools/` - Utility functions for printing and common operations
- `.env` - Environment configuration (create this file with your credentials)
