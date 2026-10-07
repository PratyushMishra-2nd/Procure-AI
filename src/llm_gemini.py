"""Gemini provider (Google AI Studio free tier friendly), via the official ``google-genai`` SDK.

Two Gemini specifics shape this adapter:

* ``gemini-2.5-flash`` cannot combine function calling with a JSON response
  schema in one request. Tool turns therefore run without a schema; the final
  answer is parsed from the model's text when it is already a valid object,
  and otherwise a second, tool-free call with ``response_json_schema`` produces it.
* The free tier is rate limited per key (requests per minute and per day).
  Calls are spaced to ``GEMINI_RPM`` per key, several keys can be pooled with
  ``GEMINI_API_KEY_POOL``, and a 429 / 503 rotates to the next key with backoff.
  Keys are never logged; only their index is.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any

from src.llm import LLMResponse, LLMUnavailable, ToolUse, extract_json_object, has_required_keys
from src.telemetry import RunTelemetryCounter

DEFAULT_MODEL = "gemini-2.5-flash"
_THROTTLE_LOCK = threading.Lock()
_LAST_CALL = [0.0]


def gemini_keys() -> list[str]:
    pool = [k.strip() for k in (os.getenv("GEMINI_API_KEY_POOL") or "").split(",") if k.strip()]
    single = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    keys = pool or ([single] if single else [])
    return list(dict.fromkeys(keys))


def _final_answer_hint(schema: dict[str, Any]) -> str:
    return (
        "\n\nFINAL ANSWER FORMAT\nWhen you have the evidence you need, reply with ONLY one JSON object "
        "(no prose, no code fences) that matches this JSON Schema:\n" + json.dumps(schema)
    )


class GeminiLLM:
    provider = "gemini"

    def __init__(self, keys: list[str] | None = None, model: str | None = None):
        from google import genai
        from google.genai import types

        self._genai, self._types = genai, types
        keys = keys or gemini_keys()
        if not keys:
            raise LLMUnavailable("No Gemini key configured (GEMINI_API_KEY or GEMINI_API_KEY_POOL)")
        timeout_ms = int(float(os.getenv("COPILOT_LLM_TIMEOUT", "120")) * 1000)
        self._clients = [genai.Client(api_key=k, http_options=types.HttpOptions(timeout=timeout_ms)) for k in keys]
        self._index = 0
        self.model = model or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL
        rpm = max(1.0, float(os.getenv("GEMINI_RPM", "10")))
        self._min_interval = 60.0 / (rpm * len(self._clients))
        self._max_wait = float(os.getenv("GEMINI_MAX_RETRY_WAIT", "90"))

    # ------------------------------------------------------------------ public

    def create(self, *, agent: str, system: str, messages: list[dict[str, Any]],
               tools: list[dict[str, Any]] | None, output_schema: dict[str, Any],
               telemetry: RunTelemetryCounter) -> LLMResponse:
        t = self._types
        contents = self._to_contents(messages)
        if not tools:
            return self._structured(agent, system, contents, output_schema, telemetry)

        config = t.GenerateContentConfig(
            system_instruction=system + _final_answer_hint(output_schema),
            temperature=0.0,
            tools=[t.Tool(function_declarations=[self._declaration(s) for s in tools])],
            automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True),
        )
        response = self._generate(agent, contents, config, telemetry)
        content = self._candidate_content(response)
        calls = [p.function_call for p in (content.parts or []) if getattr(p, "function_call", None)]
        if calls:
            uses = [ToolUse(id=f"{fc.name}|{fc.id or i}", name=fc.name, input=dict(fc.args or {})) for i, fc in enumerate(calls)]
            return LLMResponse(content=[content], stop_reason="tool_use", tool_uses=uses)

        text = "".join(p.text or "" for p in (content.parts or []) if getattr(p, "text", None) and not getattr(p, "thought", False))
        obj = extract_json_object(text)
        if obj is not None and has_required_keys(obj, output_schema):
            return LLMResponse(content=[content], stop_reason="end_turn", text=json.dumps(obj))
        # The model answered in prose: ask once more, tool-free, with the schema enforced.
        follow_up = contents + [content, t.Content(role="user", parts=[t.Part.from_text(
            text="Now return your final answer as one JSON object matching the required schema.")])]
        return self._structured(agent, system, follow_up, output_schema, telemetry)

    # ------------------------------------------------------------------ internals

    def _structured(self, agent: str, system: str, contents: list, schema: dict[str, Any],
                    telemetry: RunTelemetryCounter) -> LLMResponse:
        t = self._types
        config = t.GenerateContentConfig(
            system_instruction=system, temperature=0.0,
            response_mime_type="application/json", response_json_schema=schema,
        )
        try:
            response = self._generate(agent, contents, config, telemetry)
        except LLMUnavailable as exc:
            if "schema" not in str(exc).lower():
                raise
            # Older models reject some JSON-Schema keywords: fall back to JSON mode + schema in the prompt.
            config = t.GenerateContentConfig(system_instruction=system + _final_answer_hint(schema), temperature=0.0,
                                             response_mime_type="application/json")
            response = self._generate(agent, contents, config, telemetry)
        content = self._candidate_content(response)
        text = "".join(p.text or "" for p in (content.parts or []) if getattr(p, "text", None) and not getattr(p, "thought", False))
        obj = extract_json_object(text)
        if obj is None or not has_required_keys(obj, schema):
            raise LLMUnavailable("Gemini structured output did not match the schema")
        return LLMResponse(content=[content], stop_reason="end_turn", text=json.dumps(obj))

    def _generate(self, agent: str, contents: list, config: Any, telemetry: RunTelemetryCounter) -> Any:
        from google.genai import errors

        deadline = time.monotonic() + self._max_wait
        attempt = 0
        while True:
            attempt += 1
            self._throttle()
            client = self._clients[self._index]
            started = time.perf_counter()
            try:
                response = client.models.generate_content(model=self.model, contents=contents, config=config)
            except errors.APIError as exc:
                code = getattr(exc, "code", None)
                retryable = code in (429, 500, 503, 504)
                telemetry.event("llm_retry" if retryable else "llm_error", agent=agent, code=code, key_index=self._index)
                if not retryable:
                    raise LLMUnavailable(f"Gemini API error {code}: {str(getattr(exc, 'message', '') or exc)[:200]}") from exc
                if len(self._clients) > 1:
                    self._index = (self._index + 1) % len(self._clients)
                wait = self._retry_delay(exc, attempt)
                if time.monotonic() + wait > deadline:
                    raise LLMUnavailable(f"Gemini {code} (quota / overload) - gave up after {attempt} attempts") from exc
                time.sleep(wait)
                continue
            except Exception as exc:  # network-level failures from the HTTP stack
                if type(exc).__name__ in {"ConnectError", "ReadTimeout", "ConnectTimeout", "TimeoutException", "RemoteProtocolError"}:
                    raise LLMUnavailable(f"Cannot reach Gemini API: {type(exc).__name__}") from exc
                raise
            usage = getattr(response, "usage_metadata", None)
            telemetry.record_llm_call(
                agent,
                input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                output_tokens=(getattr(usage, "candidates_token_count", 0) or 0) + (getattr(usage, "thoughts_token_count", 0) or 0),
                ms=(time.perf_counter() - started) * 1000,
                stop_reason=str(getattr((response.candidates or [None])[0], "finish_reason", None)),
            )
            return response

    def _throttle(self) -> None:
        with _THROTTLE_LOCK:
            wait = _LAST_CALL[0] + self._min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            _LAST_CALL[0] = time.monotonic()

    @staticmethod
    def _retry_delay(exc: Exception, attempt: int) -> float:
        match = re.search(r"retry(?:Delay| in)[\"':\s]*([0-9.]+)\s*s", str(exc), flags=re.IGNORECASE)
        if match:
            return min(float(match.group(1)) + 1.0, 60.0)
        return min(2.0 ** attempt, 30.0)

    @staticmethod
    def _candidate_content(response: Any) -> Any:
        candidates = getattr(response, "candidates", None) or []
        if not candidates or candidates[0].content is None or not candidates[0].content.parts:
            reason = getattr(candidates[0], "finish_reason", "no candidates") if candidates else "no candidates"
            raise LLMUnavailable(f"Gemini returned no content (finish_reason={reason})")
        return candidates[0].content

    def _declaration(self, spec: dict[str, Any]) -> Any:
        params = spec["input_schema"]
        kwargs = {"name": spec["name"], "description": spec["description"]}
        if params.get("properties"):
            kwargs["parameters_json_schema"] = params
        return self._types.FunctionDeclaration(**kwargs)

    def _to_contents(self, messages: list[dict[str, Any]]) -> list:
        t = self._types
        out = []
        for m in messages:
            content = m["content"]
            if isinstance(content, str):
                out.append(t.Content(role="user" if m["role"] == "user" else "model", parts=[t.Part.from_text(text=content)]))
            elif m["role"] == "assistant":
                out.extend(content)  # raw SDK Content objects, replayed unchanged (keeps thought signatures)
            else:
                parts = []
                for block in content:
                    name, _, call_id = str(block["tool_use_id"]).partition("|")
                    try:
                        payload = json.loads(block["content"])
                    except (TypeError, json.JSONDecodeError):
                        payload = {"text": block["content"]}
                    response = {"error": payload} if block.get("is_error") else {"output": payload}
                    fr = t.FunctionResponse(name=name, response=response, id=None if call_id.isdigit() else call_id or None)
                    parts.append(t.Part(function_response=fr))
                out.append(t.Content(role="user", parts=parts))
        return out
