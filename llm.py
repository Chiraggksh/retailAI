"""
Gemini LLM Client module using the official google-genai SDK.

Provides:
1. Reusable Gemini client configured for 'gemini-3.8-flash'.
2. Native Gemini function calling with typed FunctionDeclaration schemas.
3. Multi-turn conversation and function response round-trip handling.
4. Comprehensive error handling (auth, rate limits, timeouts, connection errors).
5. Mock injection support for offline unit testing without external API calls.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

# Load environment variables if available
load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
FALLBACK_GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.1-flash-lite"]


class GeminiConfigurationError(Exception):
    """Raised when GEMINI_API_KEY is missing or invalid."""
    pass


@dataclass
class ToolCallRequest:
    """Represents a function call requested by the Gemini model."""
    name: str
    args: Dict[str, Any]
    call_id: Optional[str] = None


@dataclass
class GeminiResponse:
    """Standardized response from Gemini LLM."""
    text: Optional[str] = None
    tool_calls: List[ToolCallRequest] = field(default_factory=list)
    raw_content: Optional[types.Content] = None
    prompt_tokens: int = 0
    candidate_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = "STOP"
    latency_ms: float = 0.0

    @property
    def has_tool_calls(self) -> bool:
        """Check if model requested one or more tool calls."""
        return len(self.tool_calls) > 0


class GeminiClient:
    """Wrapper around official google-genai client providing function-calling orchestration."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        client: Optional[Any] = None,
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize the Gemini client.

        Args:
            api_key: Gemini API key. Defaults to GEMINI_API_KEY environment variable.
            model: Model name. Defaults to GEMINI_MODEL or 'gemini-3.8-flash'.
            client: Optional pre-configured genai.Client instance (useful for mocking).
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = model or os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
        self.timeout = timeout

        self._client = client

    @property
    def is_configured(self) -> bool:
        """Check if an API key or mock client is available."""
        return bool(self._client is not None or (self.api_key and self.api_key.strip()))

    def _get_client(self) -> genai.Client:
        """Get or initialize the official genai.Client instance."""
        if self._client is not None:
            return self._client

        if not self.api_key or not self.api_key.strip():
            raise GeminiConfigurationError(
                "GEMINI_API_KEY is missing or empty. Please set GEMINI_API_KEY in your .env file "
                "or environment before executing live model calls."
            )

        self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate(
        self,
        contents: List[Any],
        system_instruction: Optional[str] = None,
        tools: Optional[List[types.Tool | types.FunctionDeclaration]] = None,
        temperature: float = 0.0,
    ) -> GeminiResponse:
        """
        Generate content using Gemini with optional function declarations and system instructions.

        Args:
            contents: List of message strings or types.Content objects.
            system_instruction: High-level system instructions.
            tools: List of FunctionDeclaration or Tool instances.
            temperature: Sampling temperature (0.0 for deterministic factual answers).

        Returns:
            Structured GeminiResponse with text, tool calls, and token usage.
        """
        client = self._get_client()

        # Build config
        tool_list = None
        if tools:
            # If function declarations were passed directly, wrap them in a Tool
            if all(isinstance(t, types.FunctionDeclaration) for t in tools):
                tool_list = [types.Tool(function_declarations=tools)]
            else:
                tool_list = tools

        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
            tools=tool_list,
        )

        start_time = time.perf_counter()
        try:
            raw_response = client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
        except APIError as e:
            # On 503 (high demand), try fallback models before giving up
            if e.code == 503:
                logger.warning(f"Model '{self.model}' returned 503 (high demand). Trying fallback models...")
                from llm import FALLBACK_GEMINI_MODELS
                for fallback in FALLBACK_GEMINI_MODELS:
                    if fallback == self.model:
                        continue
                    try:
                        logger.info(f"Attempting fallback to '{fallback}'...")
                        raw_response = client.models.generate_content(
                            model=fallback,
                            contents=contents,
                            config=config,
                        )
                        logger.info(f"Fallback to '{fallback}' succeeded.")
                        break
                    except APIError as fe:
                        logger.warning(f"Fallback model '{fallback}' failed: {fe.code}")
                        continue
                else:
                    logger.error(f"All fallback models exhausted. Original error: {e.code}: {e.message}")
                    raise
            else:
                logger.error(f"Gemini API error ({e.code}): {e.message}")
                raise
        except Exception as e:
            logger.error(f"Error calling Gemini generate_content: {e}")
            raise

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Parse text and tool calls
        text_parts: List[str] = []
        tool_calls: List[ToolCallRequest] = []
        raw_content = None
        finish_reason = "STOP"

        if raw_response and raw_response.candidates:
            candidate = raw_response.candidates[0]
            raw_content = candidate.content
            if candidate.finish_reason:
                finish_reason = str(candidate.finish_reason)

            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if part.text:
                        text_parts.append(part.text)
                    if part.function_call:
                        fc = part.function_call
                        # Ensure args is a standard dict
                        args_dict = dict(fc.args) if fc.args else {}
                        tool_calls.append(
                            ToolCallRequest(
                                name=fc.name,
                                args=args_dict,
                                call_id=getattr(fc, "id", None),
                            )
                        )

        # Parse usage metadata if available
        prompt_tokens = 0
        candidate_tokens = 0
        total_tokens = 0
        if raw_response and hasattr(raw_response, "usage_metadata") and raw_response.usage_metadata:
            meta = raw_response.usage_metadata
            prompt_tokens = getattr(meta, "prompt_token_count", 0) or 0
            candidate_tokens = getattr(meta, "candidates_token_count", 0) or 0
            total_tokens = getattr(meta, "total_token_count", 0) or 0

        full_text = "".join(text_parts).strip() if text_parts else None

        return GeminiResponse(
            text=full_text,
            tool_calls=tool_calls,
            raw_content=raw_content,
            prompt_tokens=prompt_tokens,
            candidate_tokens=candidate_tokens,
            total_tokens=total_tokens,
            finish_reason=finish_reason,
            latency_ms=round(elapsed_ms, 2),
        )

    # -----------------------------------------------------------------------
    # Multi-Turn Message Building Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def create_user_message(text: str) -> types.Content:
        """Create a user content turn."""
        return types.Content(
            role="user",
            parts=[types.Part.from_text(text=text)],
        )

    @staticmethod
    def create_model_message(
        text: Optional[str] = None,
        tool_calls: Optional[List[ToolCallRequest]] = None,
    ) -> types.Content:
        """Create a model content turn (may contain text and/or function calls)."""
        parts: List[types.Part] = []
        if text:
            parts.append(types.Part.from_text(text=text))
        if tool_calls:
            for tc in tool_calls:
                parts.append(
                    types.Part.from_function_call(
                        name=tc.name,
                        args=tc.args,
                    )
                )
        return types.Content(role="model", parts=parts)

    @staticmethod
    def create_function_response_message(name: str, response_dict: Dict[str, Any]) -> types.Content:
        """Create a function response turn returning tool execution results to Gemini."""
        part = types.Part.from_function_response(
            name=name,
            response={"result": response_dict},
        )
        return types.Content(role="user", parts=[part])
