"""engine/vertex_wrapper.py
Wrapper around Google Gemini SDK (`google-genai`).

Uses the lightweight `google.genai` client which only requires
an API key — no GCP project, service-account or gcloud CLI needed.

Usage:
    from engine.vertex_wrapper import VertexWrapper

    wrapper = VertexWrapper(api_key="...", model_name="gemini-1.5-pro")
    wrapper.init()          # optional — lazy-inits on first generate()
    text = wrapper.generate(prompt, temperature=0.2)

Environment variables (read as fallbacks):
    LLM_API_KEY  - Gemini API key  (required)
    LLM_MODEL    - model name      (default: gemini-1.5-pro)
"""
from __future__ import annotations

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------- SDK import ----------
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    types = None
    GENAI_AVAILABLE = False

# Legacy imports (kept commented for reference)
# from openai import OpenAI
# import google.generativeai as genai_old  # deprecated
# from vertexai.preview.language_models import TextGenerationModel  # requires GCP auth


class VertexWrapper:
    """Thin wrapper around `google.genai` (Gemini SDK)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.model_name = model_name or os.getenv("LLM_MODEL", "gemini-2.0-flash")
        self._client = None
        self._inited = False

    # ---- lifecycle ----

    def init(self) -> None:
        """Create the genai.Client with the API key (idempotent)."""
        if self._inited:
            return
        if not GENAI_AVAILABLE:
            raise RuntimeError(
                "google-genai package is not installed. "
                "Run:  pip install google-genai"
            )
        if not self.api_key:
            raise RuntimeError(
                "LLM_API_KEY is empty. Set it in your .env file."
            )
        self._client = genai.Client(api_key=self.api_key)
        self._inited = True
        logger.debug("google.genai Client created (model=%s)", self.model_name)

    def _ensure_client(self):
        """Lazy-create the Client if not already done."""
        if self._client is None:
            self.init()

    # ---- public API ----

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_output_tokens: int = 4096,
    ) -> str:
        """Send *prompt* to Gemini and return the text response."""
        self._ensure_client()

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
        )

        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )

            # SDK exposes the generated text on `response.text`.
            raw = getattr(response, "text", None)
            if raw is None:
                logger.warning("Gemini response has no .text field: %s", repr(response))
                raw = str(response)

            # Log length and tail to help diagnose truncation/token issues
            try:
                length = len(raw)
            except Exception:
                length = -1
            logger.debug("Gemini response length: %d", length)
            logger.debug("Gemini response tail (last 200 chars): %s", raw[-200:] if length>0 else raw)

            return raw.strip()

        except Exception as e:
            logger.error("VertexWrapper.generate error: %s", e)
            raise


__all__ = ["VertexWrapper"]
