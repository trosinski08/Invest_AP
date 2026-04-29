"""
engine/agent.py
Decision-making logic based on LLM (OpenAI GPT).
Agent analyzes technical data + news and returns decision in JSON format.
"""
import json
import logging
from typing import Optional

from openai import OpenAI

import config

logger = logging.getLogger(__name__)

# System prompt enforcing structured JSON response
SYSTEM_PROMPT = """You are an autonomous investment agent analyzing cryptocurrency markets.
Your task is to analyze the provided technical data and market news,
then issue ONE trading decision.

RULES:
1. Be cautious — prefer HOLD when signals are mixed.
2. Never suggest transaction values above {max_order_usd} USD.
3. Your response MUST be valid JSON only (no markdown, no comments).
4. Confidence is your decision certainty on a scale of 0.0–1.0.
5. If confidence < {threshold}, decision should be HOLD.

RESPONSE FORMAT (exactly these keys):
{{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": 0.0-1.0,
  "value_usd": 0.0,
  "reasoning": "Brief decision rationale (2-3 sentences)",
  "risk_assessment": "LOW" | "MEDIUM" | "HIGH",
  "key_signals": ["signal1", "signal2"]
}}

If action=HOLD, set value_usd to 0.
If action=BUY or SELL, provide suggested value in USD (max {max_order_usd}).
""".format(
    max_order_usd=config.MAX_ORDER_VALUE_USD,
    threshold=config.SENTIMENT_THRESHOLD,
)


class TradingAgent:
    """Decision-making agent based on OpenAI GPT."""

    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.LLM_MODEL
        self.temperature = config.LLM_TEMPERATURE

    def analyze(
        self,
        market_summary: str,
        news_summary: str,
        portfolio_state: dict,
        open_positions: list[dict],
    ) -> dict:
        """
        Sends context to LLM and returns parsed decision.

        Returns:
            dict with keys: action, confidence, value_usd, reasoning,
                            risk_assessment, key_signals
        """
        user_prompt = self._build_user_prompt(
            market_summary, news_summary, portfolio_state, open_positions
        )

        logger.info("Sending query to LLM (%s)...", self.model)
        logger.debug("User prompt:\n%s", user_prompt)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content.strip()
            logger.debug("LLM raw response: %s", raw)

            decision = self._parse_decision(raw)
            logger.info(
                "LLM Decision: %s (confidence=%.2f, value=%.2f USD)",
                decision["action"], decision["confidence"], decision["value_usd"],
            )
            return decision

        except Exception as e:
            logger.error("LLM communication error: %s", e)
            return self._fallback_decision(str(e))

    def _build_user_prompt(
        self,
        market_summary: str,
        news_summary: str,
        portfolio_state: dict,
        open_positions: list[dict],
    ) -> str:
        """Builds user prompt with complete context."""
        positions_text = "No open positions." if not open_positions else json.dumps(
            open_positions, indent=2, default=str
        )

        prompt = f"""Analyze the data below and make an investment decision.

{market_summary}

{news_summary}

=== PORTFOLIO STATUS ===
Available USDT: {portfolio_state.get('free_usdt', 0):.2f}
Invested USDT: {portfolio_state.get('used_usdt', 0):.2f}
Total USDT: {portfolio_state.get('total_usdt', 0):.2f}

=== OPEN POSITIONS ===
{positions_text}

=== SAFETY LIMITS ===
- Max order value: {config.MAX_ORDER_VALUE_USD} USD
- Max daily loss: {config.MAX_DAILY_LOSS_USD} USD
- Stop-Loss: {config.STOP_LOSS_PCT}%
- Take-Profit: {config.TAKE_PROFIT_PCT}%

Provide your decision as JSON.
"""
        return prompt

    def _parse_decision(self, raw_json: str) -> dict:
        """Parses and validates LLM response."""
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON from LLM: %s", e)
            return self._fallback_decision(f"JSON parse error: {e}")

        # Validate required keys
        required_keys = {"action", "confidence", "value_usd", "reasoning"}
        missing = required_keys - set(data.keys())
        if missing:
            logger.warning("Missing keys in LLM response: %s", missing)
            return self._fallback_decision(f"Missing keys: {missing}")

        # Normalization
        action = data["action"].upper().strip()
        if action not in ("BUY", "SELL", "HOLD"):
            logger.warning("Unknown LLM action: %s → HOLD", action)
            action = "HOLD"

        confidence = max(0.0, min(1.0, float(data["confidence"])))
        value_usd = max(0.0, min(config.MAX_ORDER_VALUE_USD, float(data["value_usd"])))

        # Force HOLD if confidence too low
        if confidence < config.SENTIMENT_THRESHOLD and action != "HOLD":
            logger.info(
                "Confidence (%.2f) < threshold (%.2f) → forcing HOLD",
                confidence, config.SENTIMENT_THRESHOLD,
            )
            action = "HOLD"
            value_usd = 0.0

        return {
            "action": action,
            "confidence": confidence,
            "value_usd": value_usd if action != "HOLD" else 0.0,
            "reasoning": data.get("reasoning", ""),
            "risk_assessment": data.get("risk_assessment", "UNKNOWN"),
            "key_signals": data.get("key_signals", []),
        }

    @staticmethod
    def _fallback_decision(error_msg: str) -> dict:
        """Safe emergency decision — always HOLD."""
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "value_usd": 0.0,
            "reasoning": f"Emergency fallback decision due to error: {error_msg}",
            "risk_assessment": "HIGH",
            "key_signals": ["error_fallback"],
        }
