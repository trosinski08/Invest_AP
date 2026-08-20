"""
engine/agent.py
Decision-making logic based on LLM (OpenAI GPT).
Agent analyzes technical data + news and returns a validated TradingDecision.
"""
import json
import logging
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic model — single source of truth for LLM decision validation
# ---------------------------------------------------------------------------

class TradingDecision(BaseModel):
    """
    Validated and normalised LLM trading decision.

    Validation rules (all enforced here, not scattered in caller code):
    - action normalised to uppercase; unknown values → HOLD
    - confidence clamped to [0.0, 1.0]
    - value_usd clamped to [0.0, MAX_ORDER_VALUE_USD]
    - confidence < SENTIMENT_THRESHOLD forces HOLD
    - HOLD always sets value_usd = 0
    """
    model_config = ConfigDict(extra="ignore")

    action: Literal["BUY", "SELL", "HOLD"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    value_usd: float = Field(..., ge=0.0)
    reasoning: str = ""
    risk_assessment: str = "UNKNOWN"
    key_signals: list[str] = Field(default_factory=list)

    @field_validator("action", mode="before")
    @classmethod
    def normalize_action(cls, v: object) -> str:
        val = str(v).upper().strip()
        if val not in ("BUY", "SELL", "HOLD"):
            logger.warning("Unknown action '%s' from LLM → HOLD", v)
            return "HOLD"
        return val

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: object) -> float:
        return max(0.0, min(1.0, float(v)))

    @field_validator("value_usd", mode="before")
    @classmethod
    def clamp_value_usd(cls, v: object) -> float:
        return max(0.0, min(float(config.MAX_ORDER_VALUE_USD), float(v)))

    @model_validator(mode="after")
    def apply_guardrails(self) -> "TradingDecision":
        """Force HOLD when confidence is below threshold; zero value_usd on HOLD."""
        if self.confidence < config.SENTIMENT_THRESHOLD and self.action != "HOLD":
            logger.info(
                "Confidence %.2f < threshold %.2f → forcing HOLD",
                self.confidence, config.SENTIMENT_THRESHOLD,
            )
            self.action = "HOLD"
        if self.action == "HOLD":
            self.value_usd = 0.0
        return self


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TradingAgent:
    """Decision-making agent based on OpenAI GPT."""

    def __init__(self) -> None:
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
        Sends context to LLM and returns a validated decision dict.

        Returns dict with keys: action, confidence, value_usd, reasoning,
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

        except Exception as exc:
            logger.error("LLM communication error: %s", exc)
            return self._fallback_decision(str(exc))

    def _build_user_prompt(
        self,
        market_summary: str,
        news_summary: str,
        portfolio_state: dict,
        open_positions: list[dict],
    ) -> str:
        """Builds user prompt with complete market context."""
        positions_text = (
            "No open positions."
            if not open_positions
            else json.dumps(open_positions, indent=2, default=str)
        )
        return f"""Analyze the data below and make an investment decision.

{market_summary}

{news_summary}

=== PORTFOLIO STATUS ===
Available USDT: {portfolio_state.get('free_usdt', 0):.2f}
Invested USDT:  {portfolio_state.get('used_usdt', 0):.2f}
Total USDT:     {portfolio_state.get('total_usdt', 0):.2f}

=== OPEN POSITIONS ===
{positions_text}

=== SAFETY LIMITS ===
- Max order value: {config.MAX_ORDER_VALUE_USD} USD
- Max daily loss:  {config.MAX_DAILY_LOSS_USD} USD
- Stop-Loss:       {config.STOP_LOSS_PCT}%
- Take-Profit:     {config.TAKE_PROFIT_PCT}%

Provide your decision as JSON.
"""

    def _parse_decision(self, raw_json: str) -> dict:
        """
        Parses LLM JSON response and validates it through TradingDecision.
        Falls back to a safe HOLD on any parse or validation error.
        """
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse JSON from LLM: %s", exc)
            return self._fallback_decision(f"JSON parse error: {exc}")

        try:
            decision = TradingDecision.model_validate(data)
        except ValidationError as exc:
            logger.warning("LLM response validation failed: %s", exc)
            return self._fallback_decision(str(exc))

        return decision.model_dump()

    @staticmethod
    def _fallback_decision(error_msg: str) -> dict:
        """Safe emergency decision — always HOLD."""
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "value_usd": 0.0,
            "reasoning": f"Emergency fallback due to error: {error_msg}",
            "risk_assessment": "HIGH",
            "key_signals": ["error_fallback"],
        }
