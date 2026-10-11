"""Model providers behind one small interface (``LLMClient.create``).

Providers: Gemini direct (``src/llm_gemini.py``, default ``gemini-3.7-flash`` -
free tier), CloseRouter's OpenAI-compatible endpoint (``src/llm_openai_compat.py``,
default ``google/gemini-3.7-flash``) and Anthropic (this module).

Every provider failure becomes ``LLMUnavailable`` so the orchestrators degrade to
the deterministic path instead of crashing or, worse, guessing.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from src import config
from src.telemetry import RunTelemetryCounter

REFUSAL_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMUnavailable(Exception):
    """The model could not produce a usable response (no credentials, outage, refusal, truncation)."""


@dataclass
class ToolUse:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    content: list[Any]  # provider content blocks, appended to history unchanged
    stop_reason: str | None
    text: str = ""
    tool_uses: list[ToolUse] = field(default_factory=list)

    def json(self) -> dict[str, Any]:
        try:
            value = json.loads(self.text)
        except json.JSONDecodeError as exc:
            raise LLMUnavailable(f"Model returned non-JSON final output: {exc}") from exc
        if not isinstance(value, dict):
            raise LLMUnavailable("Model returned JSON that is not an object")
        return value


class LLMClient(Protocol):
    model: str

    def create(self, *, agent: str, system: str, messages: list[dict[str, Any]],
               tools: list[dict[str, Any]] | None, output_schema: dict[str, Any],
               telemetry: RunTelemetryCounter) -> LLMResponse: ...


def _block_get(block: Any, key: str) -> Any:
    return block.get(key) if isinstance(block, dict) else getattr(block, key, None)


class AnthropicLLM:
    provider = "anthropic"

    def __init__(self, model: str | None = None, effort: str | None = None):
        import anthropic  # imported lazily so the deterministic path works without the SDK

        self._anthropic = anthropic
        self.model = model or config.model_name()
        self.effort = effort or config.effort()
        self.client = anthropic.Anthropic(timeout=float(os.getenv("COPILOT_LLM_TIMEOUT", "120")), max_retries=2)

    def create(self, *, agent: str, system: str, messages: list[dict[str, Any]],
               tools: list[dict[str, Any]] | None, output_schema: dict[str, Any],
               telemetry: RunTelemetryCounter) -> LLMResponse:
        a = self._anthropic
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 16000,
            "system": system,
            "messages": messages,
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": output_schema},
            },
            # Opt in to Anthropic's server-side refusal fallback (routes a policy decline
            # to the recommended substitute model instead of returning an empty answer).
            "betas": [REFUSAL_FALLBACK_BETA],
            "fallbacks": "default",
        }
        if tools:
            kwargs["tools"] = tools
        started = time.perf_counter()
        try:
            response = self.client.beta.messages.create(**kwargs)
        except a.AuthenticationError as exc:
            raise LLMUnavailable("Anthropic authentication failed - set ANTHROPIC_API_KEY in .env") from exc
        except a.PermissionDeniedError as exc:
            raise LLMUnavailable(f"Anthropic permission denied: {exc.message}") from exc
        except a.NotFoundError as exc:
            raise LLMUnavailable(f"Model '{self.model}' not found: {exc.message}") from exc
        except a.BadRequestError as exc:
            raise LLMUnavailable(f"Anthropic rejected the request: {exc.message}") from exc
        except a.RateLimitError as exc:
            raise LLMUnavailable("Anthropic rate limit hit after retries") from exc
        except a.APIStatusError as exc:
            raise LLMUnavailable(f"Anthropic API error {exc.status_code}") from exc
        except a.APIConnectionError as exc:
            raise LLMUnavailable(f"Cannot reach Anthropic API: {type(exc).__name__}") from exc
        except TypeError as exc:  # raised by the SDK when no credential source resolves
            raise LLMUnavailable(f"No Anthropic credentials configured: {exc}") from exc
        ms = (time.perf_counter() - started) * 1000

        usage = getattr(response, "usage", None)
        telemetry.record_llm_call(
            agent,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            ms=ms,
            stop_reason=response.stop_reason,
        )
        if response.stop_reason == "refusal":
            raise LLMUnavailable("Model declined the request (refusal) and no fallback succeeded")
        if response.stop_reason == "max_tokens":
            raise LLMUnavailable("Model output truncated at max_tokens")
        return to_response(response.content, response.stop_reason)


def to_response(content: list[Any], stop_reason: str | None) -> LLMResponse:
    text = "".join(_block_get(b, "text") or "" for b in content if _block_get(b, "type") == "text")
    uses = [
        ToolUse(id=_block_get(b, "id"), name=_block_get(b, "name"), input=_block_get(b, "input") or {})
        for b in content
        if _block_get(b, "type") == "tool_use"
    ]
    return LLMResponse(content=list(content), stop_reason=stop_reason, text=text.strip(), tool_uses=uses)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """First complete JSON object in a reply (tolerates code fences and surrounding prose)."""
    if not text:
        return None
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def has_required_keys(obj: dict[str, Any], schema: dict[str, Any]) -> bool:
    return all(k in obj for k in schema.get("required", []))


PROVIDERS = ("gemini", "groq", "closerouter", "anthropic")
KEY_ENV = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY", "closerouter": "CLOSEROUTER_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def _available() -> dict[str, bool]:
    from src.llm_gemini import gemini_keys

    return {
        "gemini": bool(gemini_keys()),
        "groq": bool(os.getenv("GROQ_API_KEY")),
        "closerouter": bool(os.getenv("CLOSEROUTER_API_KEY")),
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("ANTHROPIC_PROFILE")),
    }


def configured_provider() -> str | None:
    """Which provider would be used, or None (AI off / no credentials).

    ``LLM_PROVIDER`` forces one; ``auto`` (default) picks the first configured of
    Gemini, Groq, CloseRouter, Anthropic. There is deliberately no automatic
    switching to another provider at run time: a run (and an eval) uses one model.
    """
    if config.llm_disabled():
        return None
    available = _available()
    wanted = (os.getenv("LLM_PROVIDER") or "auto").strip().lower()
    if wanted in available:
        return wanted if available[wanted] else None
    return next((p for p in PROVIDERS if available[p]), None)


def provider_config_problem() -> str | None:
    """Human-readable reason AI is off, so a misconfiguration is never silent."""
    if config.llm_disabled():
        return "AI disabled (COPILOT_LLM=off)"
    wanted = (os.getenv("LLM_PROVIDER") or "auto").strip().lower()
    available = _available()
    if wanted not in ("auto", "") and wanted not in available:
        return f"LLM_PROVIDER='{wanted}' is not one of {', '.join(PROVIDERS)}"
    if wanted in available and not available[wanted]:
        return f"LLM_PROVIDER={wanted} but {KEY_ENV[wanted]} is not set"
    if not any(available.values()):
        return "No model provider configured (set GEMINI_API_KEY, GROQ_API_KEY, CLOSEROUTER_API_KEY or ANTHROPIC_API_KEY in .env)"
    return None


def credentials_present() -> bool:
    return configured_provider() is not None


def describe_provider() -> tuple[str | None, str]:
    """(provider, model) for display in the UI and eval reports."""
    provider = configured_provider()
    if provider == "gemini":
        from src.llm_gemini import DEFAULT_MODEL
        return provider, os.getenv("GEMINI_MODEL") or DEFAULT_MODEL
    if provider in ("groq", "closerouter"):
        from src.llm_openai_compat import PRESETS
        p = PRESETS[provider]
        return provider, os.getenv(p.model_env) or p.default_model
    if provider == "anthropic":
        return provider, config.model_name()
    return None, "none (rules-only)"


def default_client() -> LLMClient | None:
    """The configured client, or None when AI is switched off / no credentials are present."""
    provider = configured_provider()
    if provider == "gemini":
        from src.llm_gemini import GeminiLLM
        return GeminiLLM()
    if provider in ("groq", "closerouter"):
        from src.llm_openai_compat import OpenAICompatLLM
        return OpenAICompatLLM(preset=provider)
    if provider == "anthropic":
        return AnthropicLLM()
    return None
