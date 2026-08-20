# Autonomous AI Investment Agent

Autonomous trading agent powered by GPT-4o-mini with a real-time Streamlit monitoring dashboard. Connects to Binance for market data, uses LLM-based decision making, and enforces configurable safety guardrails before executing any trade.

## Architecture

```
main.py              ← entry point — runs the agent loop (EB Worker tier)
app.py               ← Streamlit dashboard (EB Web tier)
config.py            ← all parameters (API keys via .env, trading pair, guardrails)
engine/
  agent.py           ← LLM trading logic (OpenAI) + TradingDecision Pydantic model
  tools.py           ← Binance API: fetch balance, candles, place orders
  guardrails.py      ← CircuitBreaker: max loss, max trades, cooldown enforcement
  vertex_wrapper.py  ← optional Vertex AI backend
tests/
  test_agent.py      ← unit tests for TradingDecision validation and _parse_decision
```

## Features

- **LLM decision engine** — GPT-4o-mini analyzes OHLCV candles and outputs BUY / SELL / HOLD
- **Pydantic v2 decision model** — all LLM output validated, normalised, and guardrailed at model level
- **Safety guardrails** — configurable max order size, daily loss cap, trade frequency limiter, cooldown
- **Paper trading mode** — dry-run by default; no real capital at risk unless `IS_PAPER_TRADING=false`
- **Streamlit dashboard** — live circuit breaker metrics, trade log, position overview

## Quickstart (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # fill in API keys

# Terminal 1 — agent loop
python main.py

# Terminal 2 — dashboard
streamlit run app.py
```

## Configuration (`.env`)

```env
OPENAI_API_KEY=sk-...
BINANCE_API_KEY=...
BINANCE_SECRET_KEY=...

# Trading behaviour
TRADING_PAIR=BTC/USDT
TIMEFRAME=1h
IS_PAPER_TRADING=true

# Guardrails
MAX_ORDER_VALUE_USD=20
MAX_DAILY_LOSS_USD=40
MAX_TRADES_PER_24H=5
COOLDOWN_MINUTES=60
SENTIMENT_THRESHOLD=0.7
```

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

## AWS Elastic Beanstalk deployment

The project uses **two separate EB environments** (one per process type):

### Web tier — Streamlit dashboard

```bash
eb init -p python-3.11 investap-web --region eu-west-1
eb create investap-web-prod
eb setenv OPENAI_API_KEY=sk-... IS_PAPER_TRADING=true
eb deploy
```

`Procfile` (already correct for web):
```
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

### Worker tier — agent trading loop

```bash
# In the worker deployment package: cp Procfile.worker Procfile
eb init -p python-3.11 investap-worker --region eu-west-1
eb create investap-worker-prod
eb setenv OPENAI_API_KEY=sk-... IS_PAPER_TRADING=true DATA_DIR=/tmp/invest_ap
eb deploy
```

`Procfile.worker` (rename to `Procfile` in the worker EB environment):
```
worker: python main.py
```

### Environment variables (set via `eb setenv`, never commit to repo)

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | yes | OpenAI key |
| `BINANCE_API_KEY` | real-money only | Binance key |
| `BINANCE_SECRET_KEY` | real-money only | Binance secret |
| `IS_PAPER_TRADING` | yes | `true` / `false` |
| `DATA_DIR` | recommended on EB | Writable path for state/log files |

---

## MVP Persistent Storage on AWS

Currently the agent stores state in **local files** (`data/trades.log`, `data/paper_state.json`).
These are **lost on EB instance restart / redeploy**. Recommended migration path:

| Current (local file) | AWS replacement | Effort |
|---|---|---|
| `data/trades.log` (JSONL trade history) | **DynamoDB** table `InvestAP_Trades` — append-only, query by timestamp | Low — swap `_load_trades` / `record_trade` in `guardrails.py` |
| `data/paper_state.json` (portfolio state) | **DynamoDB** item (single row) or **S3** JSON object | Low — swap `_get_paper_balance` / `_save_paper_state` in `tools.py` |
| `data/agent.log` (text logs) | **CloudWatch Logs** — automatic if logging to stdout | Zero — EB ships stdout to CloudWatch by default |

### Minimal DynamoDB schema

```
Table: InvestAP_Trades
  PK: pair      (String)  — e.g. "BTC/USDT"
  SK: timestamp (String)  — ISO-8601 UTC
  Attributes: action, price, amount, value_usd, pnl_usd, confidence, paper
```

Until DynamoDB is wired you can mount an **EFS volume** at `DATA_DIR` to persist files across restarts with zero code changes.

---

## Stack

Python 3.11 · OpenAI API (GPT-4o-mini) · Binance API (ccxt) · Streamlit · Pydantic v2 · python-dotenv

## Disclaimer

Research/learning project only. Not financial advice. Use paper trading mode unless you fully understand the risks of automated trading.
