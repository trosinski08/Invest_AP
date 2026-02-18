"""
main.py
Główna pętla autonomicznego agenta inwestycyjnego.
Uruchamianie: python main.py
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
# WYKONYWANIE TRANSAKCJI
# ---------------------------------------------------------------------------

def execute_trade(decision: dict, circuit: CircuitBreaker) -> dict:
    """
    Wykonuje transakcję na podstawie decyzji agenta.
    W trybie paper-trading symuluje transakcję lokalnie.
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

    # Sprawdź guardrails
    allowed, reason = circuit.can_trade(value_usd)
    if not allowed:
        logger.warning("Transakcja zablokowana: %s", reason)
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
    """Symuluje transakcję w trybie paper-trading."""
    action = decision["action"]
    value_usd = decision["value_usd"]
    pair = config.TRADING_PAIR

    # Pobierz aktualną cenę
    try:
        df = fetch_ohlcv(limit=1)
        current_price = float(df.iloc[-1]["close"])
    except Exception:
        logger.error("Nie można pobrać ceny — transakcja anulowana")
        return {"action": "ERROR", "reason": "Nie można pobrać ceny"}

    amount = value_usd / current_price
    balance = fetch_balance()

    if action == "BUY":
        if balance["free_usdt"] < value_usd:
            return {"action": "BLOCKED", "reason": "Brak wystarczających środków"}
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
        # Znajdź otwartą pozycję
        positions = balance.get("positions", [])
        matching = [p for p in positions if p.get("pair") == pair]
        if not matching:
            return {"action": "BLOCKED", "reason": "Brak otwartej pozycji do zamknięcia"}
        pos = matching[0]
        pnl = (current_price - pos["entry_price"]) * pos["amount"]
        balance["free_usdt"] += pos["amount"] * current_price
        balance["used_usdt"] -= pos.get("value_usd", 0)
        balance["total_usdt"] = balance["free_usdt"] + balance["used_usdt"]
        positions.remove(pos)
        balance["positions"] = positions
    else:
        return {"action": "ERROR", "reason": f"Nieznana akcja: {action}"}

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
    """Wykonuje rzeczywistą transakcję przez API Binance."""
    action = decision["action"]
    value_usd = decision["value_usd"]
    pair = config.TRADING_PAIR

    exchange = get_exchange(sandbox=False)
    symbol = pair.replace("/", "")

    try:
        # Pobierz aktualną cenę
        ticker = exchange.fetch_ticker(pair)
        current_price = ticker["last"]
        amount = value_usd / current_price

        if action == "BUY":
            order = exchange.create_market_buy_order(pair, amount)
        elif action == "SELL":
            order = exchange.create_market_sell_order(pair, amount)
        else:
            return {"action": "ERROR", "reason": f"Nieznana akcja: {action}"}

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
        logger.error("Błąd wykonania zlecenia: %s", e)
        return {"action": "ERROR", "reason": str(e)}


# ---------------------------------------------------------------------------
# SPRAWDZANIE STOP-LOSS / TAKE-PROFIT
# ---------------------------------------------------------------------------

def check_stop_loss_take_profit(circuit: CircuitBreaker) -> list[dict]:
    """
    Sprawdza otwarte pozycje pod kątem stop-loss i take-profit.
    Zwraca listę automatycznie wygenerowanych decyzji SELL.
    """
    open_positions = circuit.get_open_positions()
    if not open_positions:
        return []

    try:
        df = fetch_ohlcv(limit=1)
        current_price = float(df.iloc[-1]["close"])
    except Exception:
        logger.warning("Nie można sprawdzić SL/TP — brak danych cenowych")
        return []

    auto_sells = []
    for pos in open_positions:
        entry = pos.get("price", pos.get("entry_price", 0))
        if entry <= 0:
            continue

        change_pct = ((current_price - entry) / entry) * 100

        if change_pct <= -config.STOP_LOSS_PCT:
            logger.warning("⛔ STOP-LOSS aktywowany! Zmiana: %.2f%% (limit: -%.1f%%)",
                           change_pct, config.STOP_LOSS_PCT)
            auto_sells.append({
                "action": "SELL",
                "confidence": 1.0,
                "value_usd": pos.get("value_usd", config.MAX_ORDER_VALUE_USD),
                "reasoning": f"Automatyczny STOP-LOSS: cena spadła o {change_pct:.2f}%",
            })

        elif change_pct >= config.TAKE_PROFIT_PCT:
            logger.info("✅ TAKE-PROFIT aktywowany! Zmiana: +%.2f%% (limit: +%.1f%%)",
                        change_pct, config.TAKE_PROFIT_PCT)
            auto_sells.append({
                "action": "SELL",
                "confidence": 1.0,
                "value_usd": pos.get("value_usd", config.MAX_ORDER_VALUE_USD),
                "reasoning": f"Automatyczny TAKE-PROFIT: cena wzrosła o {change_pct:.2f}%",
            })

    return auto_sells


# ---------------------------------------------------------------------------
# GŁÓWNA PĘTLA
# ---------------------------------------------------------------------------

def run_cycle(agent: TradingAgent, circuit: CircuitBreaker) -> dict:
    """Pojedynczy cykl analizy i (opcjonalnego) handlu."""
    logger.info("=" * 60)
    logger.info("NOWY CYKL ANALIZY — %s", datetime.now(timezone.utc).isoformat())
    logger.info("=" * 60)

    # 0) Sprawdź SL/TP na otwartych pozycjach
    auto_decisions = check_stop_loss_take_profit(circuit)
    for auto_dec in auto_decisions:
        execute_trade(auto_dec, circuit)

    # 1) Pobierz dane rynkowe
    logger.info("Pobieram dane rynkowe...")
    try:
        df = fetch_ohlcv()
        df = calculate_indicators(df)
        market_summary = summarize_market(df)
    except Exception as e:
        logger.error("Nie udało się pobrać danych rynkowych: %s", e)
        return {"action": "ERROR", "reason": str(e)}

    # 2) Pobierz wiadomości
    logger.info("Pobieram wiadomości...")
    news = fetch_news()
    news_summary = format_news_for_llm(news)

    # 3) Stan portfela
    portfolio = fetch_balance()
    open_positions = circuit.get_open_positions()

    # 4) Analiza LLM
    logger.info("Analizuję dane z LLM...")
    decision = agent.analyze(market_summary, news_summary, portfolio, open_positions)

    # 5) Wykonaj transakcję (guardrails sprawdzane wewnątrz)
    result = execute_trade(decision, circuit)

    # 6) Podsumowanie
    daily_pnl = circuit.get_daily_pnl()
    logger.info("Dzienny PnL: %.2f USD | Decyzja: %s | Confidence: %.2f",
                daily_pnl, decision["action"], decision["confidence"])

    return result


def main():
    """Punkt wejścia — uruchamia pętlę agenta."""
    logger.info("🚀 Uruchamiam Autonomous AI Investment Agent")
    logger.info("Para: %s | Interwał: %s | Paper: %s",
                config.TRADING_PAIR, config.TIMEFRAME, config.IS_PAPER_TRADING)

    # Walidacja konfiguracji
    try:
        config.validate_config()
    except EnvironmentError as e:
        logger.error("Błąd konfiguracji: %s", e)
        sys.exit(1)

    agent = TradingAgent()
    circuit = CircuitBreaker()

    logger.info("Agent gotowy. Rozpoczynam pętlę (interwał: %ds)...",
                config.LOOP_INTERVAL_SECONDS)

    cycle_count = 0
    while True:
        cycle_count += 1
        try:
            logger.info("--- Cykl #%d ---", cycle_count)
            result = run_cycle(agent, circuit)
            logger.info("Wynik cyklu: %s", json.dumps(result, default=str, ensure_ascii=False))
        except KeyboardInterrupt:
            logger.info("Zatrzymano przez użytkownika (Ctrl+C)")
            break
        except Exception as e:
            logger.error("Nieoczekiwany błąd w cyklu #%d: %s", cycle_count, e, exc_info=True)

        logger.info("Następny cykl za %d sekund...", config.LOOP_INTERVAL_SECONDS)
        try:
            time.sleep(config.LOOP_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            logger.info("Zatrzymano przez użytkownika (Ctrl+C)")
            break

    logger.info("Agent zakończył pracę.")


if __name__ == "__main__":
    main()
