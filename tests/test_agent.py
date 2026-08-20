"""
tests/test_agent.py
Unit tests for TradingDecision Pydantic model and TradingAgent._parse_decision.

External services (OpenAI, Binance) are never called — all LLM interaction
is exercised through _parse_decision with pre-baked JSON strings.
"""
import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

# conftest.py stubs heavy deps before this import
import config
from engine.agent import TradingDecision, TradingAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw(**kwargs) -> str:
    """Build a minimal valid LLM JSON payload, overridable per test."""
    base = {
        "action": "BUY",
        "confidence": 0.8,
        "value_usd": 10.0,
        "reasoning": "Strong buy signal.",
        "risk_assessment": "LOW",
        "key_signals": ["rsi_oversold", "macd_cross"],
    }
    base.update(kwargs)
    return json.dumps(base)


def _agent() -> TradingAgent:
    """Return a TradingAgent with the OpenAI client mocked out."""
    agent = TradingAgent.__new__(TradingAgent)
    agent.client = MagicMock()
    agent.model = config.LLM_MODEL
    agent.temperature = config.LLM_TEMPERATURE
    return agent


# ---------------------------------------------------------------------------
# TradingDecision — validation rules
# ---------------------------------------------------------------------------

class TestTradingDecisionValid:
    def test_valid_buy_decision(self):
        d = TradingDecision.model_validate({
            "action": "BUY",
            "confidence": 0.85,
            "value_usd": 15.0,
            "reasoning": "Bullish",
            "risk_assessment": "LOW",
            "key_signals": ["rsi", "macd"],
        })
        assert d.action == "BUY"
        assert d.confidence == 0.85
        assert d.value_usd == 15.0

    def test_valid_sell_decision(self):
        d = TradingDecision.model_validate({
            "action": "SELL",
            "confidence": 0.75,
            "value_usd": 10.0,
        })
        assert d.action == "SELL"

    def test_valid_hold_decision(self):
        d = TradingDecision.model_validate({
            "action": "HOLD",
            "confidence": 0.5,
            "value_usd": 0.0,
        })
        assert d.action == "HOLD"
        assert d.value_usd == 0.0


class TestTradingDecisionNormalization:
    def test_action_lowercased_to_uppercase(self):
        d = TradingDecision.model_validate({"action": "buy", "confidence": 0.9, "value_usd": 5.0})
        assert d.action == "BUY"

    def test_action_mixed_case(self):
        d = TradingDecision.model_validate({"action": "Sell", "confidence": 0.9, "value_usd": 5.0})
        assert d.action == "SELL"

    def test_unknown_action_becomes_hold(self):
        d = TradingDecision.model_validate({"action": "INVEST", "confidence": 0.9, "value_usd": 5.0})
        assert d.action == "HOLD"
        assert d.value_usd == 0.0  # HOLD zeroes value_usd

    def test_confidence_above_1_clamped(self):
        d = TradingDecision.model_validate({"action": "BUY", "confidence": 1.5, "value_usd": 5.0})
        assert d.confidence == 1.0

    def test_confidence_below_0_clamped(self):
        d = TradingDecision.model_validate({"action": "BUY", "confidence": -0.3, "value_usd": 5.0})
        assert d.confidence == 0.0

    def test_value_usd_above_max_clamped(self):
        d = TradingDecision.model_validate({
            "action": "BUY",
            "confidence": 0.9,
            "value_usd": config.MAX_ORDER_VALUE_USD + 999,
        })
        assert d.value_usd == config.MAX_ORDER_VALUE_USD

    def test_value_usd_negative_clamped_to_zero(self):
        d = TradingDecision.model_validate({"action": "BUY", "confidence": 0.9, "value_usd": -5.0})
        assert d.value_usd == 0.0

    def test_extra_keys_ignored(self):
        """LLM sometimes adds extra explanation keys — must not raise."""
        d = TradingDecision.model_validate({
            "action": "HOLD",
            "confidence": 0.5,
            "value_usd": 0.0,
            "extra_llm_field": "ignored",
            "another_extra": 42,
        })
        assert d.action == "HOLD"


class TestTradingDecisionGuardrails:
    def test_low_confidence_forces_hold(self):
        """confidence < SENTIMENT_THRESHOLD (0.7) → action forced to HOLD."""
        threshold = config.SENTIMENT_THRESHOLD
        low_conf = max(0.0, threshold - 0.01)
        d = TradingDecision.model_validate({
            "action": "BUY",
            "confidence": low_conf,
            "value_usd": 15.0,
        })
        assert d.action == "HOLD"
        assert d.value_usd == 0.0

    def test_confidence_at_threshold_allows_trade(self):
        """confidence == SENTIMENT_THRESHOLD is permitted (strict < comparison)."""
        threshold = config.SENTIMENT_THRESHOLD
        d = TradingDecision.model_validate({
            "action": "BUY",
            "confidence": threshold,
            "value_usd": 10.0,
        })
        assert d.action == "BUY"

    def test_hold_always_zeros_value_usd(self):
        d = TradingDecision.model_validate({
            "action": "HOLD",
            "confidence": 0.95,
            "value_usd": 999.0,
        })
        assert d.value_usd == 0.0

    def test_low_confidence_hold_value_usd_zeroed(self):
        d = TradingDecision.model_validate({
            "action": "SELL",
            "confidence": 0.1,
            "value_usd": 10.0,
        })
        assert d.action == "HOLD"
        assert d.value_usd == 0.0


class TestTradingDecisionDefaults:
    def test_reasoning_defaults_to_empty_string(self):
        d = TradingDecision.model_validate({"action": "HOLD", "confidence": 0.5, "value_usd": 0.0})
        assert d.reasoning == ""

    def test_risk_assessment_defaults_to_unknown(self):
        d = TradingDecision.model_validate({"action": "HOLD", "confidence": 0.5, "value_usd": 0.0})
        assert d.risk_assessment == "UNKNOWN"

    def test_key_signals_defaults_to_empty_list(self):
        d = TradingDecision.model_validate({"action": "HOLD", "confidence": 0.5, "value_usd": 0.0})
        assert d.key_signals == []

    def test_model_dump_returns_all_fields(self):
        d = TradingDecision.model_validate({"action": "HOLD", "confidence": 0.5, "value_usd": 0.0})
        data = d.model_dump()
        assert set(data.keys()) == {
            "action", "confidence", "value_usd",
            "reasoning", "risk_assessment", "key_signals",
        }


class TestTradingDecisionMissingRequired:
    def test_missing_action_raises(self):
        with pytest.raises(ValidationError):
            TradingDecision.model_validate({"confidence": 0.8, "value_usd": 5.0})

    def test_missing_confidence_raises(self):
        with pytest.raises(ValidationError):
            TradingDecision.model_validate({"action": "BUY", "value_usd": 5.0})

    def test_missing_value_usd_raises(self):
        with pytest.raises(ValidationError):
            TradingDecision.model_validate({"action": "BUY", "confidence": 0.8})

    def test_empty_dict_raises(self):
        with pytest.raises(ValidationError):
            TradingDecision.model_validate({})


# ---------------------------------------------------------------------------
# TradingAgent._parse_decision
# ---------------------------------------------------------------------------

class TestParseDecision:
    def setup_method(self):
        self.agent = _agent()

    def test_valid_json_returns_decision_dict(self):
        result = self.agent._parse_decision(_make_raw())
        assert result["action"] == "BUY"
        assert result["confidence"] == 0.8
        assert result["value_usd"] == 10.0

    def test_invalid_json_returns_fallback(self):
        result = self.agent._parse_decision("not valid json {{{")
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0
        assert "error_fallback" in result["key_signals"]

    def test_missing_required_keys_returns_fallback(self):
        result = self.agent._parse_decision(json.dumps({"action": "BUY"}))
        assert result["action"] == "HOLD"
        assert "error_fallback" in result["key_signals"]

    def test_low_confidence_raw_json_forces_hold(self):
        raw = _make_raw(action="BUY", confidence=0.1, value_usd=15.0)
        result = self.agent._parse_decision(raw)
        assert result["action"] == "HOLD"
        assert result["value_usd"] == 0.0

    def test_unknown_action_in_raw_json_becomes_hold(self):
        raw = _make_raw(action="MOON", confidence=0.95, value_usd=10.0)
        result = self.agent._parse_decision(raw)
        assert result["action"] == "HOLD"

    def test_value_usd_capped_in_parsed_result(self):
        raw = _make_raw(action="BUY", confidence=0.95, value_usd=99999.0)
        result = self.agent._parse_decision(raw)
        assert result["value_usd"] <= config.MAX_ORDER_VALUE_USD

    def test_extra_llm_fields_stripped(self):
        raw = _make_raw()
        raw_dict = json.loads(raw)
        raw_dict["hallucinated_field"] = "some extra text"
        result = self.agent._parse_decision(json.dumps(raw_dict))
        assert "hallucinated_field" not in result

    def test_result_is_plain_dict(self):
        """_parse_decision must return a plain dict, not a Pydantic model."""
        result = self.agent._parse_decision(_make_raw())
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# TradingAgent._fallback_decision
# ---------------------------------------------------------------------------

class TestFallbackDecision:
    def test_fallback_is_safe_hold(self):
        result = TradingAgent._fallback_decision("test error")
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0
        assert result["value_usd"] == 0.0
        assert result["risk_assessment"] == "HIGH"
        assert "error_fallback" in result["key_signals"]
        assert "test error" in result["reasoning"]
