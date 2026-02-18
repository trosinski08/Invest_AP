import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Załaduj zmienne środowiskowe z pliku .env (API Keys)
load_dotenv()

# --- ŚCIEŻKI PROJEKTU ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
LOG_FILE = DATA_DIR / "trades.log"

# --- 1. POŁĄCZENIA API ---
# Pamiętaj: Klucze trzymaj w pliku .env, nigdy bezpośrednio w kodzie!
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

# --- 2. PARAMETRY MODELU LLM ---
LLM_MODEL = "gpt-4o-mini"       # Model do analizy (gpt-4o, gpt-4o-mini)
LLM_TEMPERATURE = 0.2           # Niska temperatura = bardziej deterministyczne decyzje

# --- 3. PARAMETRY RYNKOWE ---
# Ticker, na którym bot będzie operował (np. BTC/USDT)
TRADING_PAIR = "BTC/USDT"
TIMEFRAME = "1h"                 # Interwał analizy (1m, 5m, 1h, 1d)
CANDLE_LIMIT = 100               # Ile świec historycznych pobierać do analizy

# --- 4. SZTYWNE LIMITY BEZPIECZEŃSTWA (GUARDRAILS) ---
# To są parametry, których AI NIE MOŻE zmienić.
MAX_ORDER_VALUE_USD = 20.0       # Maksymalna kwota jednego zlecenia w USD
MAX_DAILY_LOSS_USD = 40.0        # Bot wyłącza się po stracie X USD w ciągu dnia
MAX_OPEN_POSITIONS = 1           # Ile transakcji może być otwartych naraz
MAX_TRADES_PER_24H = 5           # Limit transakcji/24h, aby nie spalił prowizji
COOLDOWN_MINUTES = 60            # Minimalny odstęp między transakcjami (minuty)
STOP_LOSS_PCT = 3.0              # Automatyczny stop-loss (%) na pozycji
TAKE_PROFIT_PCT = 5.0            # Automatyczny take-profit (%) na pozycji

# --- 5. LOGIKA ANALIZY ---
NEWS_COUNT = 5                   # Ile najnowszych newsów wysyłać do LLM
SENTIMENT_THRESHOLD = 0.7        # Minimalna pewność AI (podniesiona dla bezpieczeństwa)

# --- 6. TRYB PRACY ---
IS_PAPER_TRADING = True          # Zmień na False TYLKO gdy będziesz gotowy na Real Money
LOOP_INTERVAL_SECONDS = 3600     # Co ile sekund agent uruchamia analizę (domyślnie 1h)

# --- 7. LOGGING ---
LOG_LEVEL = logging.INFO

# --- WALIDACJA ---
def validate_config():
    """Sprawdź czy krytyczne zmienne środowiskowe są ustawione."""
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not IS_PAPER_TRADING:
        # W trybie real-money wymagane są klucze Binance
        if not BINANCE_API_KEY:
            missing.append("BINANCE_API_KEY")
        if not BINANCE_SECRET_KEY:
            missing.append("BINANCE_SECRET_KEY")
    if missing:
        raise EnvironmentError(
            f"Brakujące zmienne środowiskowe: {', '.join(missing)}. "
            f"Uzupełnij plik .env."
        )