"""
engine/guardrails.py
Safety module (Circuit Breaker).
Independent of LLM — hard limits that AI cannot bypass.
"""
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """
    Checks hard safety conditions BEFORE executing trades.
    Any violation blocks trading and logs the reason.
    """

    def __init__(self, log_file: Optional[Path] = None):
        self.log_file = log_file or config.LOG_FILE
        self._ensure_log_file()

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def can_trade(self, order_value_usd: float) -> tuple[bool, str]:
        """
        Main validation method. Returns (True, "") if trading allowed,
        or (False, reason) if blocked.
        """
        checks = [
            self._check_paper_mode,
            lambda: self._check_order_size(order_value_usd),
            self._check_daily_loss,
            self._check_trade_count,
            self._check_open_positions,
            self._check_cooldown,
        ]
        for check in checks:
            allowed, reason = check()
            if not allowed:
                logger.warning("CIRCUIT BREAKER — blocked: %s", reason)
                return False, reason

        logger.info("CIRCUIT BREAKER — trading allowed (order=%.2f USD)", order_value_usd)
        return True, ""

    def record_trade(self, trade: dict) -> None:
        """
        Records transaction to log.
        trade should contain: action, pair, price, amount, value_usd, timestamp, reasoning
        """
        trade.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(trade, ensure_ascii=False) + "\n")
        logger.info("Recorded trade: %s %s @ %.2f USD",
                     trade.get("action"), trade.get("pair"), trade.get("value_usd", 0))

    def get_trade_history(self, hours: int = 24) -> list[dict]:
        """Returns trades from last N hours."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        trades = self._load_trades()
        recent = []
        for t in trades:
            ts = t.get("timestamp", "")
            try:
                trade_time = datetime.fromisoformat(ts)
                if trade_time >= cutoff:
                    recent.append(t)
            except (ValueError, TypeError):
                continue
        return recent

    def get_daily_pnl(self) -> float:
        """Calculates P&L from today's closed trades (in USD)."""
        trades = self.get_trade_history(hours=24)
        pnl = 0.0
        for t in trades:
            pnl += t.get("pnl_usd", 0.0)
        return pnl

    def get_open_positions(self) -> list[dict]:
        """Returns currently open positions from log."""
        trades = self._load_trades()
        positions = {}
        for t in trades:
            pair = t.get("pair", "")
            action = t.get("action", "").upper()
            if action == "BUY":
                positions[pair] = t
            elif action == "SELL" and pair in positions:
                del positions[pair]
        return list(positions.values())

    def get_all_trades(self) -> list[dict]:
        """Returns entire trade history."""
        return self._load_trades()

    # ------------------------------------------------------------------
    # PRIVATE CHECK METHODS
    # ------------------------------------------------------------------

    def _check_paper_mode(self) -> tuple[bool, str]:
        """Informally logs operation mode."""
        if config.IS_PAPER_TRADING:
            logger.debug("Mode: PAPER TRADING")
        return True, ""

    def _check_order_size(self, value_usd: float) -> tuple[bool, str]:
        """Checks if order does not exceed maximum value."""
        if value_usd > config.MAX_ORDER_VALUE_USD:
            return False, (
                f"Order value ({value_usd:.2f} USD) exceeds limit "
                f"({config.MAX_ORDER_VALUE_USD:.2f} USD)"
            )
        if value_usd <= 0:
            return False, "Order value must be > 0"
        return True, ""

    def _check_daily_loss(self) -> tuple[bool, str]:
        """Checks if daily loss did not exceed limit."""
        daily_pnl = self.get_daily_pnl()
        if daily_pnl < 0 and abs(daily_pnl) >= config.MAX_DAILY_LOSS_USD:
            return False, (
                f"Daily loss ({abs(daily_pnl):.2f} USD) reached limit "
                f"({config.MAX_DAILY_LOSS_USD:.2f} USD). Bot paused until tomorrow."
            )
        return True, ""

    def _check_trade_count(self) -> tuple[bool, str]:
        """Checks number of trades in last 24h."""
        recent = self.get_trade_history(hours=24)
        # Count only orders (BUY/SELL), not info entries
        executed = [t for t in recent if t.get("action", "").upper() in ("BUY", "SELL")]
        if len(executed) >= config.MAX_TRADES_PER_24H:
            return False, (
                f"Reached limit of {config.MAX_TRADES_PER_24H} trades/24h "
                f"(executed: {len(executed)})"
            )
        return True, ""

    def _check_open_positions(self) -> tuple[bool, str]:
        """Checks number of open positions."""
        open_pos = self.get_open_positions()
        if len(open_pos) >= config.MAX_OPEN_POSITIONS:
            return False, (
                f"Reached limit of {config.MAX_OPEN_POSITIONS} open positions "
                f"(currently: {len(open_pos)})"
            )
        return True, ""

    def _check_cooldown(self) -> tuple[bool, str]:
        """Checks minimum interval between trades."""
        recent = self.get_trade_history(hours=24)
        executed = [t for t in recent if t.get("action", "").upper() in ("BUY", "SELL")]
        if not executed:
            return True, ""

        last_trade = executed[-1]
        try:
            last_time = datetime.fromisoformat(last_trade["timestamp"])
            cooldown_end = last_time + timedelta(minutes=config.COOLDOWN_MINUTES)
            now = datetime.now(timezone.utc)
            if now < cooldown_end:
                remaining = (cooldown_end - now).total_seconds() / 60
                return False, (
                    f"Cooldown active — next trade possible in "
                    f"{remaining:.0f} min (interval: {config.COOLDOWN_MINUTES} min)"
                )
        except (ValueError, KeyError):
            pass
        return True, ""

    # ------------------------------------------------------------------
    # HELPER
    # ------------------------------------------------------------------

    def _ensure_log_file(self) -> None:
        """Creates log file if it doesn't exist."""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_file.exists():
            self.log_file.touch()

    def _load_trades(self) -> list[dict]:
        """Loads all trades from JSONL file."""
        trades = []
        if not self.log_file.exists():
            return trades
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        trades.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("Skipped corrupted log entry: %s", line[:80])
        return trades
