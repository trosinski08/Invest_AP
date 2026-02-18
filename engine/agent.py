"""
engine/agent.py
Logika decyzyjna oparta na LLM (OpenAI GPT).
Agent analizuje dane techniczne + newsy i zwraca decyzję w formacie JSON.
"""
import json
import logging
from typing import Optional

from openai import OpenAI

import config

logger = logging.getLogger(__name__)

# System prompt wymuszający strukturalną odpowiedź JSON
SYSTEM_PROMPT = """Jesteś autonomicznym agentem inwestycyjnym analizującym kryptowaluty.
Twoim zadaniem jest analiza dostarczonych danych technicznych i wiadomości rynkowych,
a następnie wydanie JEDNEJ decyzji handlowej.

ZASADY:
1. Bądź ostrożny — preferuj HOLD gdy sygnały są mieszane.
2. Nigdy nie sugeruj wartości transakcji powyżej {max_order_usd} USD.
3. Twoja odpowiedź MUSI być wyłącznie poprawnym obiektem JSON (bez markdown, bez komentarzy).
4. Confidence to Twoja pewność decyzji w skali 0.0–1.0.
5. Jeśli confidence < {threshold}, decyzja powinna być HOLD.

FORMAT ODPOWIEDZI (dokładnie te klucze):
{{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": 0.0-1.0,
  "value_usd": 0.0,
  "reasoning": "Krótkie uzasadnienie decyzji (2-3 zdania)",
  "risk_assessment": "LOW" | "MEDIUM" | "HIGH",
  "key_signals": ["signal1", "signal2"]
}}

Jeśli action=HOLD, ustaw value_usd na 0.
Jeśli action=BUY lub SELL, podaj sugerowaną wartość w USD (max {max_order_usd}).
""".format(
    max_order_usd=config.MAX_ORDER_VALUE_USD,
    threshold=config.SENTIMENT_THRESHOLD,
)


class TradingAgent:
    """Agent decyzyjny oparty na OpenAI GPT."""

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
        Wysyła kontekst do LLM i zwraca sparsowaną decyzję.

        Returns:
            dict z kluczami: action, confidence, value_usd, reasoning,
                              risk_assessment, key_signals
        """
        user_prompt = self._build_user_prompt(
            market_summary, news_summary, portfolio_state, open_positions
        )

        logger.info("Wysyłam zapytanie do LLM (%s)...", self.model)
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
                "Decyzja LLM: %s (confidence=%.2f, value=%.2f USD)",
                decision["action"], decision["confidence"], decision["value_usd"],
            )
            return decision

        except Exception as e:
            logger.error("Błąd komunikacji z LLM: %s", e)
            return self._fallback_decision(str(e))

    def _build_user_prompt(
        self,
        market_summary: str,
        news_summary: str,
        portfolio_state: dict,
        open_positions: list[dict],
    ) -> str:
        """Buduje prompt użytkownika z całym kontekstem."""
        positions_text = "Brak otwartych pozycji." if not open_positions else json.dumps(
            open_positions, indent=2, default=str
        )

        prompt = f"""Przeanalizuj poniższe dane i podejmij decyzję inwestycyjną.

{market_summary}

{news_summary}

=== STAN PORTFELA ===
Dostępne USDT: {portfolio_state.get('free_usdt', 0):.2f}
Zainwestowane USDT: {portfolio_state.get('used_usdt', 0):.2f}
Łącznie USDT: {portfolio_state.get('total_usdt', 0):.2f}

=== OTWARTE POZYCJE ===
{positions_text}

=== LIMITY BEZPIECZEŃSTWA ===
- Max wartość zlecenia: {config.MAX_ORDER_VALUE_USD} USD
- Max dzienne straty: {config.MAX_DAILY_LOSS_USD} USD
- Stop-Loss: {config.STOP_LOSS_PCT}%
- Take-Profit: {config.TAKE_PROFIT_PCT}%

Podaj swoją decyzję jako JSON.
"""
        return prompt

    def _parse_decision(self, raw_json: str) -> dict:
        """Parsuje i waliduje odpowiedź LLM."""
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as e:
            logger.error("Nie udało się sparsować JSON z LLM: %s", e)
            return self._fallback_decision(f"JSON parse error: {e}")

        # Walidacja wymaganych kluczy
        required_keys = {"action", "confidence", "value_usd", "reasoning"}
        missing = required_keys - set(data.keys())
        if missing:
            logger.warning("Brakujące klucze w odpowiedzi LLM: %s", missing)
            return self._fallback_decision(f"Missing keys: {missing}")

        # Normalizacja
        action = data["action"].upper().strip()
        if action not in ("BUY", "SELL", "HOLD"):
            logger.warning("Nieznana akcja LLM: %s → HOLD", action)
            action = "HOLD"

        confidence = max(0.0, min(1.0, float(data["confidence"])))
        value_usd = max(0.0, min(config.MAX_ORDER_VALUE_USD, float(data["value_usd"])))

        # Wymuszone HOLD jeśli pewność zbyt niska
        if confidence < config.SENTIMENT_THRESHOLD and action != "HOLD":
            logger.info(
                "Confidence (%.2f) < threshold (%.2f) → wymuszam HOLD",
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
        """Bezpieczna decyzja awaryjna — zawsze HOLD."""
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "value_usd": 0.0,
            "reasoning": f"Decyzja awaryjna (fallback) z powodu błędu: {error_msg}",
            "risk_assessment": "HIGH",
            "key_signals": ["error_fallback"],
        }
