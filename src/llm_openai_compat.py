"""OpenAI-compatible chat-completions provider (used for CloseRouter's Gemini models).

Plain ``requests`` so no extra SDK is needed. As with direct Gemini, tool turns
run without a response schema; the final answer is parsed from text and, if it
is not a valid object, re-requested once in JSON mode with the schema in the
system prompt.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import requests

from src.llm import LLMResponse, LLMUnavailable, ToolUse, extract_json_object, has_required_keys
from src.telemetry import RunTelemetryCounter

DEFAULT_BASE_URL = "https://api.closerouter.dev/v1"
DEFAULT_MODEL = "google/gemini-3.7-flash"


def _schema_hint(schema: dict[str, Any]) -> str:
    return ("\n\nFINAL ANSWER FORMAT\nWhen you have the evidence you need, reply with ONLY one JSON object "
            "(no prose, no code fences) that matches this JSON Schema:\n" + json.dumps(schema))


class OpenAICompatLLM:
    provider = "closerouter"

    def __init__(self, api_key: str | None = None, model: str | None = None, base_url: str | None = None):
        self._key = api_key or os.getenv("CLOSEROUTER_API_KEY") or ""
        if not self._key:
            raise LLMUnavailable("No CLOSEROUTER_API_KEY configured")
        self.model = model or os.getenv("CLOSEROUTER_MODEL") or DEFAULT_MODEL
        self._url = (base_url or os.getenv("CLOSEROUTER_BASE_URL") or DEFAULT_BASE_URL).rstrip("/") + "/chat/completions"
        self._timeout = float(os.getenv("COPILOT_LLM_TIMEOUT", "120"))

    def create(self, *, agent: str, system: str, messages: list[dict[str, Any]],
               tools: list[dict[str, Any]] | None, output_schema: dict[str, Any],
               telemetry: RunTelemetryCounter) -> LLMResponse:
        chat = self._to_chat(messages)
        if not tools:
            return self._structured(agent, system, chat, output_schema, telemetry)
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "system", "content": system + _schema_hint(output_schema)}, *chat],
            "tools": [{"type": "function", "function": {"name": s["name"], "description": s["description"],
                                                        "parameters": s["input_schema"]}} for s in tools],
        }
        msg = self._post(agent, payload, telemetry)
        calls = msg.get("tool_calls") or []
        if calls:
            uses = []
            for c in calls:
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                uses.append(ToolUse(id=c["id"], name=c["function"]["name"], input=args if isinstance(args, dict) else {}))
            return LLMResponse(content=[msg], stop_reason="tool_use", tool_uses=uses)
        obj = extract_json_object(msg.get("content") or "")
        if obj is not None and has_required_keys(obj, output_schema):
            return LLMResponse(content=[msg], stop_reason="end_turn", text=json.dumps(obj))
        follow_up = chat + [msg, {"role": "user", "content": "Now return your final answer as one JSON object matching the required schema."}]
        return self._structured(agent, system, follow_up, output_schema, telemetry)

    def _structured(self, agent: str, system: str, chat: list, schema: dict[str, Any],
                    telemetry: RunTelemetryCounter) -> LLMResponse:
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system + _schema_hint(schema)}, *chat],
        }
        msg = self._post(agent, payload, telemetry)
        obj = extract_json_object(msg.get("content") or "")
        if obj is None or not has_required_keys(obj, schema):
            raise LLMUnavailable("Model JSON output did not match the schema")
        return LLMResponse(content=[msg], stop_reason="end_turn", text=json.dumps(obj))

    def _post(self, agent: str, payload: dict[str, Any], telemetry: RunTelemetryCounter) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        for attempt in range(1, 5):
            started = time.perf_counter()
            try:
                r = requests.post(self._url, headers=headers, json=payload, timeout=self._timeout)
            except requests.RequestException as exc:
                raise LLMUnavailable(f"Cannot reach {self._url}: {type(exc).__name__}") from exc
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 4:
                telemetry.event("llm_retry", agent=agent, code=r.status_code)
                time.sleep(min(2.0 ** attempt, 20.0))
                continue
            if r.status_code >= 400:
                raise LLMUnavailable(f"Provider HTTP {r.status_code}: {r.text[:200]}")
            body = r.json()
            usage = body.get("usage") or {}
            choice = (body.get("choices") or [{}])[0]
            telemetry.record_llm_call(agent, input_tokens=usage.get("prompt_tokens", 0) or 0,
                                      output_tokens=usage.get("completion_tokens", 0) or 0,
                                      ms=(time.perf_counter() - started) * 1000, stop_reason=choice.get("finish_reason"))
            msg = choice.get("message")
            if not isinstance(msg, dict):
                raise LLMUnavailable("Provider returned no message")
            return {k: v for k, v in msg.items() if k in {"role", "content", "tool_calls"}}
        raise LLMUnavailable("Provider kept returning 429/5xx")

    @staticmethod
    def _to_chat(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            content = m["content"]
            if isinstance(content, str):
                out.append({"role": m["role"], "content": content})
            elif m["role"] == "assistant":
                out.extend(content)  # raw assistant message dicts, replayed unchanged
            else:
                for block in content:
                    out.append({"role": "tool", "tool_call_id": block["tool_use_id"], "content": block["content"]})
        return out
