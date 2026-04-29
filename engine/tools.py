"""
engine/tools.py
Tools for fetching market data (OHLCV) and news.
"""
import logging
from datetime import datetime
from typing import Optional

import ccxt
import pandas as pd
import ta
from duckduckgo_search import DDGS

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. MARKET DATA (Market Data)
# ---------------------------------------------------------------------------

def get_exchange(sandbox: bool = True) -> ccxt.binance:
    """Returns configured Binance exchange instance."""
    exchange = ccxt.binance({
        "apiKey": config.BINANCE_API_KEY,
        "secret": config.BINANCE_SECRET_KEY,
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })
    if sandbox or config.IS_PAPER_TRADING:
        exchange.set_sandbox_mode(True)
    return exchange


def fetch_ohlcv(
    pair: str = config.TRADING_PAIR,
    timeframe: str = config.TIMEFRAME,
    limit: int = config.CANDLE_LIMIT,
) -> pd.DataFrame:
    """
    Fetches OHLCV candles from Binance and returns DataFrame.
    Columns: timestamp, open, high, low, close, volume
    """
    try:
        exchange = get_exchange()
        raw = exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        logger.info("Fetched %d candles for %s (%s)", len(df), pair, timeframe)
        return df
    except Exception as e:
        logger.error("Error fetching OHLCV: %s", e)
        raise


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates technical indicators and returns extended DataFrame.
    - SMA 20 / SMA 50
    - RSI (14)
    - MACD (12, 26, 9)
    - Bollinger Bands (20, 2)
    """
    df = df.copy()

    # Simple Moving Averages
    df["sma_20"] = ta.trend.sma_indicator(df["close"], window=20)
    df["sma_50"] = ta.trend.sma_indicator(df["close"], window=50)

    # RSI
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)

    # MACD
    macd = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()

    logger.info("Calculated technical indicators (SMA, RSI, MACD, BB)")
    return df


def summarize_market(df: pd.DataFrame) -> str:
    """
    Creates text summary of market from latest data,
    ready to send as context to LLM.
    """
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    price_change_pct = ((latest["close"] - prev["close"]) / prev["close"]) * 100

    summary = (
        f"=== TECHNICAL ANALYSIS ({config.TRADING_PAIR}, {config.TIMEFRAME}) ===\n"
        f"Current price: {latest['close']:.2f} USD (change: {price_change_pct:+.2f}%)\n"
        f"Open: {latest['open']:.2f} | High: {latest['high']:.2f} | "
        f"Low: {latest['low']:.2f} | Volume: {latest['volume']:.2f}\n"
        f"SMA(20): {latest['sma_20']:.2f} | SMA(50): {latest['sma_50']:.2f}\n"
        f"RSI(14): {latest['rsi']:.1f}\n"
        f"MACD: {latest['macd']:.2f} | Signal: {latest['macd_signal']:.2f} | "
        f"Histogram: {latest['macd_hist']:.2f}\n"
        f"Bollinger Bands: Lower={latest['bb_lower']:.2f}, "
        f"Mid={latest['bb_mid']:.2f}, Upper={latest['bb_upper']:.2f}\n"
    )
    return summary


# ---------------------------------------------------------------------------
# 2. NEWS / SENTIMENT (News Ingestion)
# ---------------------------------------------------------------------------

def fetch_news(
    query: Optional[str] = None,
    count: int = config.NEWS_COUNT,
) -> list[dict]:
    """
    Fetches latest news from DuckDuckGo Search.
    Returns list of dicts with keys: title, body, url, date.
    """
    if query is None:
        # Default query based on trading pair
        base_asset = config.TRADING_PAIR.split("/")[0]  # e.g. "BTC"
        query = f"{base_asset} cryptocurrency news"

    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.news(query, max_results=count):
                results.append({
                    "title": r.get("title", ""),
                    "body": r.get("body", ""),
                    "url": r.get("url", ""),
                    "date": r.get("date", ""),
                })
        logger.info("Fetched %d news for query: '%s'", len(results), query)
        return results
    except Exception as e:
        logger.warning("Error fetching news: %s — continuing without it.", e)
        return []


def format_news_for_llm(news: list[dict]) -> str:
    """Formats news into text ready to paste into LLM prompt."""
    if not news:
        return "No news available.\n"

    lines = ["=== LATEST NEWS ==="]
    for i, item in enumerate(news, 1):
        lines.append(
            f"{i}. [{item['date']}] {item['title']}\n"
            f"   {item['body'][:200]}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. PORTFOLIO (Portfolio / Balance)
# ---------------------------------------------------------------------------

def fetch_balance(exchange: Optional[ccxt.binance] = None) -> dict:
    """
    Fetches portfolio status from exchange.
    In paper-trading mode returns simulated portfolio.
    """
    if config.IS_PAPER_TRADING:
        return _get_paper_balance()

    if exchange is None:
        exchange = get_exchange(sandbox=False)

    try:
        balance = exchange.fetch_balance()
        usdt = balance.get("USDT", {})
        return {
            "total_usdt": usdt.get("total", 0.0),
            "free_usdt": usdt.get("free", 0.0),
            "used_usdt": usdt.get("used", 0.0),
        }
    except Exception as e:
        logger.error("Error fetching balance: %s", e)
        raise


def _get_paper_balance() -> dict:
    """Returns simulated paper-trading portfolio from state file."""
    import json
    state_file = config.DATA_DIR / "paper_state.json"
    if state_file.exists():
        with open(state_file, "r") as f:
            return json.load(f)
    # Default initial state
    default = {
        "total_usdt": 100.0,
        "free_usdt": 100.0,
        "used_usdt": 0.0,
        "positions": [],
    }
    _save_paper_state(default)
    return default


def _save_paper_state(state: dict) -> None:
    """Saves paper-trading state to file."""
    import json
    state_file = config.DATA_DIR / "paper_state.json"
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2, default=str)
