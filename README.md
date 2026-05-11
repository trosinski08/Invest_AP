# Autonomous AI Investment Agent

Autonomous trading agent powered by GPT-4o-mini with a real-time Streamlit monitoring dashboard. Connects to Binance for market data, uses LLM-based decision making, and enforces configurable safety guardrails before executing any trade.

## Architecture

```
main.py              ← entry point, runs the agent loop
app.py               ← Streamlit dashboard for real-time monitoring
config.py            ← all parameters (API keys via .env, trading pair, guardrails)
engine/
  agent.py           ← LLM-based trading logic (OpenAI GPT-4o-mini)
  tools.py           ← Binance API: fetch balance, candles, place orders
  guardrails.py      ← CircuitBreaker: max loss, max trades, cooldown enforcement
  vertex_wrapper.py  ← optional Vertex AI backend
```

## Features

- **LLM decision engine** — GPT-4o-mini analyzes OHLCV candles and outputs BUY / SELL / HOLD
- **Safety guardrails** — configurable max order size, daily loss cap, trade frequency limiter, cooldown
- **Paper trading mode** — dry-run by default, no real capital at risk unless `IS_PAPER_TRADING=false`
- **Streamlit dashboard** — live circuit breaker metrics, trade log, position overview

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in API keys
python main.py         # run agent
streamlit run app.py   # open dashboard
```

## Configuration (`.env`)

```env
OPENAI_API_KEY=...
BINANCE_API_KEY=...
BINANCE_SECRET_KEY=...
TRADING_PAIR=BTC/USDT
TIMEFRAME=1h
IS_PAPER_TRADING=true
MAX_ORDER_VALUE_USD=50
MAX_DAILY_LOSS_USD=100
MAX_TRADES_PER_24H=10
```

## Stack

Python · OpenAI API (GPT-4o-mini) · Binance API · Streamlit · python-dotenv

## Disclaimer

This is a research/learning project. Not financial advice. Use paper trading mode only unless you understand the risks of automated trading.
