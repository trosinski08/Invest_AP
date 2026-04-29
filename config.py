import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (API Keys)
load_dotenv()

# --- PROJECT PATHS ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
LOG_FILE = DATA_DIR / "trades.log"

# --- 1. API CONNECTIONS ---
# Remember: Keep API keys in .env file, never hardcode them directly!
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

# --- 2. LLM MODEL PARAMETERS ---
LLM_MODEL = "gpt-4o-mini"       # Model for analysis (gpt-4o, gpt-4o-mini)
LLM_TEMPERATURE = 0.2           # Low temperature = more deterministic decisions

# --- 3. MARKET PARAMETERS ---
# Ticker on which the bot will operate (e.g., BTC/USDT)
TRADING_PAIR = "BTC/USDT"
TIMEFRAME = "1h"                 # Analysis interval (1m, 5m, 1h, 1d)
CANDLE_LIMIT = 100               # Number of historical candles to fetch for analysis

# --- 4. HARD SAFETY LIMITS (GUARDRAILS) ---
# These are parameters that AI CANNOT change.
MAX_ORDER_VALUE_USD = 20.0       # Maximum order value in USD
MAX_DAILY_LOSS_USD = 40.0        # Bot shuts down after X USD loss per day
MAX_OPEN_POSITIONS = 1           # How many transactions can be open at once
MAX_TRADES_PER_24H = 5           # Transaction limit/24h to avoid burning fees
COOLDOWN_MINUTES = 60            # Minimum interval between transactions (minutes)
STOP_LOSS_PCT = 3.0              # Automatic stop-loss (%) on position
TAKE_PROFIT_PCT = 5.0            # Automatic take-profit (%) on position

# --- 5. ANALYSIS LOGIC ---
NEWS_COUNT = 5                   # Number of latest news items to send to LLM
SENTIMENT_THRESHOLD = 0.7        # Minimum AI confidence (raised for safety)

# --- 6. OPERATION MODE ---
IS_PAPER_TRADING = True          # Change to False ONLY when ready for Real Money
LOOP_INTERVAL_SECONDS = 3600     # How often agent runs analysis (default 1h)

# --- 7. LOGGING ---
LOG_LEVEL = logging.INFO

# --- VALIDATION ---
def validate_config():
    """Check if critical environment variables are set."""
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not IS_PAPER_TRADING:
        # In real-money mode, Binance keys are required
        if not BINANCE_API_KEY:
            missing.append("BINANCE_API_KEY")
        if not BINANCE_SECRET_KEY:
            missing.append("BINANCE_SECRET_KEY")
    if missing:
        raise EnvironmentError(
            f"Missing environment variables: {', '.join(missing)}. "
            f"Complete the .env file."
        )