"""
main.py
Main loop of the autonomous investment agent.
Run: python main.py
"""
import json
import logging
import sys
import time
from datetime import datetime, timezone

import config
from engine.agent import TradingAgent
from engine.guardrails import CircuitBreaker
from engine.tools import (
    calculate_indicators,
    fetch_balance,
    fetch_news,
    fetch_ohlcv,
    format_news_for_llm,
    get_exchange,
    summarize_market,
    _save_paper_state,
)

# ---------------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.DATA_DIR / "agent.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# TRADE EXECUTION
# ---------------------------------------------------------------------------

def execute_trade(decision: dict, circuit: CircuitBreaker) -> dict:
    """
    Executes a trade based on agent's decision.
    In paper-trading mode, simulates the trade locally.
    """
    action = decision["action"]
    value_usd = decision["value_usd"]
    pair = config.TRADING_PAIR

    if action == "HOLD":
        trade_record = {
            "action": "HOLD",
            "pair": pair,
            "price": 0,
            "amount": 0,
            "value_usd": 0,
            "pnl_usd": 0,
            "reasoning": decision.get("reasoning", ""),
            "confidence": decision.get("confidence", 0),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "paper": config.IS_PAPER_TRADING,
        }
        circuit.record_trade(trade_record)
        return trade_record

    # Check guardrails
    allowed, reason = circuit.can_trade(value_usd)
    if not allowed:
        logger.warning("Trade blocked: %s", reason)
        return {
            "action": "BLOCKED",
            "reason": reason,
            "original_decision": decision,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    if config.IS_PAPER_TRADING:
        return _execute_paper_trade(decision, circuit)
    else:
        return _execute_real_trade(decision, circuit)


def _execute_paper_trade(decision: dict, circuit: CircuitBreaker) -> dict:
    """Simulates a trade in paper-trading mode."""
    action = decision["action"]
    value_usd = decision["value_usd"]
    pair = config.TRADING_PAIR

    # Fetch current price
    try:
        df = fetch_ohlcv(limit=1)
        current_price = float(df.iloc[-1]["close"])
    except Exception:
        logger.error("Cannot fetch price — trade cancelled")
        return {"action": "ERROR", "reason": "Cannot fetch price"}

    amount = value_usd / current_price
    balance = fetch_balance()

    if action == "BUY":
        if balance["free_usdt"] < value_usd:
            return {"action": "BLOCKED", "reason": "Insufficient funds"}
        balance["free_usdt"] -= value_usd
        balance["used_usdt"] += value_usd
        balance.setdefault("positions", []).append({
            "pair": pair,
            "entry_price": current_price,
            "amount": amount,
            "value_usd": value_usd,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        pnl = 0.0

    elif action == "SELL":
        # Find open position
        positions = balance.get("positions", [])
        matching = [p for p in positions if p.get("pair") == pair]
        if not matching:
            return {"action": "BLOCKED", "reason": "No open position to close"}
        pos = matching[0]
        pnl = (current_price - pos["entry_price"]) * pos["amount"]
        balance["free_usdt"] += pos["amount"] * current_price
        balance["used_usdt"] -= pos.get("value_usd", 0)
        balance["total_usdt"] = balance["free_usdt"] + balance["used_usdt"]
        positions.remove(pos)
        balance["positions"] = positions
    else:
        return {"action": "ERROR", "reason": f"Unknown action: {action}"}

    balance["total_usdt"] = balance["free_usdt"] + balance["used_usdt"]
    _save_paper_state(balance)

    trade_record = {
        "action": action,
        "pair": pair,
        "price": current_price,
        "amount": amount,
        "value_usd": value_usd,
        "pnl_usd": pnl,
        "reasoning": decision.get("reasoning", ""),
        "confidence": decision.get("confidence", 0),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "paper": True,
    }
    circuit.record_trade(trade_record)
    logger.info("📝 PAPER TRADE: %s %s — %.6f @ %.2f USD (PnL: %.2f)",
                action, pair, amount, current_price, pnl)
    return trade_record


def _execute_real_trade(decision: dict, circuit: CircuitBreaker) -> dict:
    """Executes a real trade through Binance API."""
    action = decision["action"]
    value_usd = decision["value_usd"]
    pair = config.TRADING_PAIR

    exchange = get_exchange(sandbox=False)
    symbol = pair.replace("/", "")

    try:
        # Fetch current price
        ticker = exchange.fetch_ticker(pair)
        current_price = ticker["last"]
        amount = value_usd / current_price

        if action == "BUY":
            order = exchange.create_market_buy_order(pair, amount)
        elif action == "SELL":
            order = exchange.create_market_sell_order(pair, amount)
        else:
            return {"action": "ERROR", "reason": f"Unknown action: {action}"}

        trade_record = {
            "action": action,
            "pair": pair,
            "price": current_price,
            "amount": amount,
            "value_usd": value_usd,
            "pnl_usd": 0.0,
            "order_id": order.get("id"),
            "reasoning": decision.get("reasoning", ""),
            "confidence": decision.get("confidence", 0),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "paper": False,
        }
        circuit.record_trade(trade_record)
        logger.info("💰 REAL TRADE: %s %s — %.6f @ %.2f USD",
                     action, pair, amount, current_price)
        return trade_record

    except Exception as e:
        logger.error("Order execution error: %s", e)
        return {"action": "ERROR", "reason": str(e)}


# ---------------------------------------------------------------------------
# STOP-LOSS / TAKE-PROFIT CHECK
# ---------------------------------------------------------------------------

def check_stop_loss_take_profit(circuit: CircuitBreaker) -> list[dict]:
    """
    Checks open positions for stop-loss and take-profit.
    Returns list of automatically generated SELL decisions.
    """
    open_positions = circuit.get_open_positions()
    if not open_positions:
        return []

    try:
        df = fetch_ohlcv(limit=1)
        current_price = float(df.iloc[-1]["close"])
    except Exception:
        logger.warning("Cannot check SL/TP — no price data available")
        return []

    auto_sells = []
    for pos in open_positions:
        entry = pos.get("price", pos.get("entry_price", 0))
        if entry <= 0:
            continue

        change_pct = ((current_price - entry) / entry) * 100

        if change_pct <= -config.STOP_LOSS_PCT:
            logger.warning("⛔ STOP-LOSS triggered! Change: %.2f%% (limit: -%.1f%%)",
                           change_pct, config.STOP_LOSS_PCT)
            auto_sells.append({
                "action": "SELL",
                "confidence": 1.0,
                "value_usd": pos.get("value_usd", config.MAX_ORDER_VALUE_USD),
                "reasoning": f"Automatic STOP-LOSS: price dropped by {change_pct:.2f}%",
            })

        elif change_pct >= config.TAKE_PROFIT_PCT:
            logger.info("✅ TAKE-PROFIT triggered! Change: +%.2f%% (limit: +%.1f%%)",
                        change_pct, config.TAKE_PROFIT_PCT)
            auto_sells.append({
                "action": "SELL",
                "confidence": 1.0,
                "value_usd": pos.get("value_usd", config.MAX_ORDER_VALUE_USD),
                "reasoning": f"Automatic TAKE-PROFIT: price rose by {change_pct:.2f}%",
            })

    return auto_sells


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------

def run_cycle(agent: TradingAgent, circuit: CircuitBreaker) -> dict:
    """Single analysis cycle with optional trading."""
    logger.info("=" * 60)
    logger.info("NEW ANALYSIS CYCLE — %s", datetime.now(timezone.utc).isoformat())
    logger.info("=" * 60)

    # 0) Check SL/TP on open positions
    auto_decisions = check_stop_loss_take_profit(circuit)
    for auto_dec in auto_decisions:
        execute_trade(auto_dec, circuit)

    # 1) Fetch market data
    logger.info("Fetching market data...")
    try:
        df = fetch_ohlcv()
        df = calculate_indicators(df)
        market_summary = summarize_market(df)
    except Exception as e:
        logger.error("Failed to fetch market data: %s", e)
        return {"action": "ERROR", "reason": str(e)}

    # 2) Fetch news
    logger.info("Fetching news...")
    news = fetch_news()
    news_summary = format_news_for_llm(news)

    # 3) Portfolio state
    portfolio = fetch_balance()
    open_positions = circuit.get_open_positions()

    # 4) LLM analysis
    logger.info("Analyzing with LLM...")
    decision = agent.analyze(market_summary, news_summary, portfolio, open_positions)

    # 5) Execute trade (guardrails checked inside)
    result = execute_trade(decision, circuit)

    # 6) Summary
    daily_pnl = circuit.get_daily_pnl()
    logger.info("Daily PnL: %.2f USD | Decision: %s | Confidence: %.2f",
                daily_pnl, decision["action"], decision["confidence"])

    return result


def main():
    """Entry point — runs agent loop."""
    logger.info("🚀 Starting Autonomous AI Investment Agent")
    logger.info("Pair: %s | Interval: %s | Paper: %s",
                config.TRADING_PAIR, config.TIMEFRAME, config.IS_PAPER_TRADING)

    # Validate configuration
    try:
        config.validate_config()
    except EnvironmentError as e:
        logger.error("Configuration error: %s", e)
        sys.exit(1)

    agent = TradingAgent()
    circuit = CircuitBreaker()

    logger.info("Agent ready. Starting loop (interval: %ds)...",
                config.LOOP_INTERVAL_SECONDS)

    cycle_count = 0
    while True:
        cycle_count += 1
        try:
            logger.info("--- Cycle #%d ---", cycle_count)
            result = run_cycle(agent, circuit)
            logger.info("Cycle result: %s", json.dumps(result, default=str, ensure_ascii=False))
        except KeyboardInterrupt:
            logger.info("Stopped by user (Ctrl+C)")
            break
        except Exception as e:
            logger.error("Unexpected error in cycle #%d: %s", cycle_count, e, exc_info=True)

        logger.info("Next cycle in %d seconds...", config.LOOP_INTERVAL_SECONDS)
        try:
            time.sleep(config.LOOP_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            logger.info("Stopped by user (Ctrl+C)")
            break

    logger.info("Agent finished.")


if __name__ == "__main__":
    main()
