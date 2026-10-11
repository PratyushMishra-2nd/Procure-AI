"""OpenAI-compatible chat-completions provider: Groq and CloseRouter presets.

Plain ``requests`` so no extra SDK is needed. Tool turns run without a response
schema; the final answer is parsed from the reply and, if it is not a valid
object, re-requested once in JSON mode with the schema in the system prompt.

Provider quirks handled here:
* **Groq** models sometimes "call" an undeclared ``json`` tool to deliver the
  answer, or the API returns HTTP 400 ``tool_use_failed`` with the attempted
  output in ``failed_generation``. Both are treated as a candidate *answer* and
  validated against the schema; an undeclared tool is never executed.
* **Gemini 3 through an OpenAI-compatible router** returns a thought signature
  in ``extra_content`` that must be replayed on the next turn, so assistant
  messages are replayed with ``extra_content`` intact.
* **Free-tier limits**: calls are paced to requests/tokens per minute
  (``RateLimiter``); a per-day limit, or a retry-after longer than
  ``LLM_MAX_RETRY_WAIT``, puts the provider/model into cooldown and fails fast.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import requests

from src.llm import LLMResponse, LLMUnavailable, ToolUse, extract_json_object, has_required_keys
from src.ratelimit import COOLDOWN, RateLimiter
from src.telemetry import RunTelemetryCounter


@dataclass(frozen=True)
class Preset:
    name: str
    key_env: str
    model_env: str
    base_env: str
    default_model: str
    default_base: str
    default_rpm: float | None
    default_tpm: int | None
    answer_tool: bool = False  # deliver the final answer through a declared `final_answer` tool


PRESETS = {
    "groq": Preset("groq", "GROQ_API_KEY", "GROQ_MODEL", "GROQ_BASE_URL", "qwen/qwen3.8-27b",
                   "https://api.groq.com/openai/v1", 30, 8000, answer_tool=True),
    "closerouter": Preset("closerouter", "CLOSEROUTER_API_KEY", "CLOSEROUTER_MODEL", "CLOSEROUTER_BASE_URL",
                          "google/gemini-3.7-flash", "https://api.closerouter.dev/v1", None, None),
}
# Backwards-compatible names
DEFAULT_BASE_URL = PRESETS["closerouter"].default_base
DEFAULT_MODEL = PRESETS["closerouter"].default_model
FINAL_ANSWER_TOOL = "final_answer"
ANSWER_TOOL_NAMES = {"json", FINAL_ANSWER_TOOL, "answer", "respond", "output"}
_LIMITERS: dict[str, RateLimiter] = {}


def _schema_hint(schema: dict[str, Any]) -> str:
    return ("\n\nFINAL ANSWER FORMAT\nWhen you have the evidence you need, reply with ONLY one JSON object "
            "(no prose, no code fences, no tool call) that matches this JSON Schema:\n" + json.dumps(schema, separators=(",", ":")))


def _answer_tool_hint() -> str:
    return (f"\n\nFINAL ANSWER FORMAT\nWhen you have the evidence you need, call the `{FINAL_ANSWER_TOOL}` tool exactly once "
            "with your complete answer as its arguments. Do not answer in plain text and do not call any other undeclared tool.")


def _answer_tool_spec(schema: dict[str, Any]) -> dict[str, Any]:
    return {"type": "function", "function": {
        "name": FINAL_ANSWER_TOOL,
        "description": "Submit the final structured answer. This records your answer; it performs no action.",
        "parameters": schema}}


def _unwrap(obj: dict[str, Any] | None) -> dict[str, Any] | None:
    """`failed_generation` often holds {"name": "json", "arguments": {...}} - return the inner answer."""
    if isinstance(obj, dict):
        for key in ("arguments", "parameters", "input"):
            inner = obj.get(key)
            if isinstance(inner, str):
                from src.llm import extract_json_object as _x
                inner = _x(inner)
            if isinstance(inner, dict):
                return inner
    return obj


def _float_env(name: str, default: float | None) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else None


def _parse_wait(text: str) -> float | None:
    """Seconds from 'try again in 7m12.5s' / 'retry in 15h3m' / '2.5s' style messages."""
    m = re.search(r"(?:try again|retry)\s+in\s+((?:\d+(?:\.\d+)?[hms]\s*)+)", text, flags=re.IGNORECASE)
    if not m:
        return None
    total = 0.0
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)([hms])", m.group(1)):
        total += float(num) * {"h": 3600, "m": 60, "s": 1}[unit]
    return total


class OpenAICompatLLM:
    def __init__(self, api_key: str | None = None, model: str | None = None, base_url: str | None = None,
                 preset: str = "closerouter"):
        p = PRESETS[preset]
        self.provider = p.name
        self._key = api_key or os.getenv(p.key_env) or ""
        if not self._key:
            raise LLMUnavailable(f"No {p.key_env} configured")
        self.model = model or os.getenv(p.model_env) or p.default_model
        self._url = (base_url or os.getenv(p.base_env) or p.default_base).rstrip("/") + "/chat/completions"
        self._answer_tool = p.answer_tool
        self._timeout = float(os.getenv("COPILOT_LLM_TIMEOUT", "120"))
        self._max_wait = float(os.getenv("LLM_MAX_RETRY_WAIT", "90"))
        rpm = _float_env("LLM_RPM", p.default_rpm)
        tpm = _float_env("LLM_TPM", p.default_tpm)
        key = f"{self.provider}:{self.model}"
        self._cooldown_key = key
        if key not in _LIMITERS:
            _LIMITERS[key] = RateLimiter(rpm=rpm, tpm=int(tpm) if tpm else None)
        self._limiter = _LIMITERS[key]

    # ------------------------------------------------------------------ public

    def create(self, *, agent: str, system: str, messages: list[dict[str, Any]],
               tools: list[dict[str, Any]] | None, output_schema: dict[str, Any],
               telemetry: RunTelemetryCounter) -> LLMResponse:
        chat = self._to_chat(messages)
        if not tools:
            return self._structured(agent, system, chat, output_schema, telemetry)
        declared = {s["name"] for s in tools}
        tool_specs = [{"type": "function", "function": {"name": s["name"], "description": s["description"],
                                                        "parameters": s["input_schema"]}} for s in tools]
        hint = _schema_hint(output_schema)
        if self._answer_tool:
            tool_specs.append(_answer_tool_spec(output_schema))
            hint = _answer_tool_hint()
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "system", "content": system + hint}, *chat],
            "tools": tool_specs,
        }
        msg = self._post(agent, payload, telemetry, output_schema)
        if isinstance(msg, LLMResponse):  # answer recovered from a tool_use_failed error
            return msg
        calls = msg.get("tool_calls") or []
        submitted = self._submitted_answer(msg, output_schema)
        if submitted is not None:
            return LLMResponse(content=[msg], stop_reason="end_turn", text=json.dumps(submitted))
        real = [c for c in calls if c.get("function", {}).get("name") in declared]
        if not real:
            answer = self._answer_from(msg, output_schema)
            if answer is not None:
                if calls:
                    telemetry.event("undeclared_tool_as_answer", agent=agent, name=calls[0].get("function", {}).get("name"))
                return LLMResponse(content=[msg], stop_reason="end_turn", text=json.dumps(answer))
            if not calls:
                follow_up = chat + [msg, {"role": "user", "content": "Now return your final answer as one JSON object matching the required schema."}]
                return self._structured(agent, system, follow_up, output_schema, telemetry)
        uses = []
        for c in calls:  # undeclared names are passed through so the tool layer reports them as errors
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            uses.append(ToolUse(id=c["id"], name=c["function"]["name"], input=args if isinstance(args, dict) else {}))
        return LLMResponse(content=[msg], stop_reason="tool_use", tool_uses=uses)

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _answer_from(msg: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any] | None:
        # Only called when no *declared* tool was requested, so any tool call here is an undeclared
        # one (e.g. Groq's "json"): its arguments are read as a candidate answer, never executed.
        calls = sorted(msg.get("tool_calls") or [], key=lambda c: c.get("function", {}).get("name") not in ANSWER_TOOL_NAMES)
        candidates = [msg.get("content") or ""] + [c.get("function", {}).get("arguments") or "" for c in calls]
        for text in candidates:
            obj = _unwrap(extract_json_object(text)) if isinstance(text, str) else None
            if obj is not None and has_required_keys(obj, schema):
                return obj
        return None

    @staticmethod
    def _submitted_answer(msg: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any] | None:
        for c in msg.get("tool_calls") or []:
            if c.get("function", {}).get("name") == FINAL_ANSWER_TOOL:
                obj = extract_json_object(c["function"].get("arguments") or "")
                if obj is not None and has_required_keys(obj, schema):
                    return obj
        return None

    def _structured(self, agent: str, system: str, chat: list, schema: dict[str, Any],
                    telemetry: RunTelemetryCounter) -> LLMResponse:
        if self._answer_tool:
            payload = {
                "model": self.model,
                "temperature": 0,
                "messages": [{"role": "system", "content": system + _answer_tool_hint()}, *chat],
                "tools": [_answer_tool_spec(schema)],
                "tool_choice": {"type": "function", "function": {"name": FINAL_ANSWER_TOOL}},
            }
            msg = self._post(agent, payload, telemetry, schema)
            if isinstance(msg, LLMResponse):
                return msg
            obj = self._submitted_answer(msg, schema) or self._answer_from(msg, schema)
            if obj is None:
                raise LLMUnavailable("Model answer did not match the schema")
            return LLMResponse(content=[msg], stop_reason="end_turn", text=json.dumps(obj))
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system + _schema_hint(schema)}, *chat],
        }
        msg = self._post(agent, payload, telemetry, schema)
        if isinstance(msg, LLMResponse):
            return msg
        obj = self._answer_from(msg, schema)
        if obj is None:
            raise LLMUnavailable("Model JSON output did not match the schema")
        return LLMResponse(content=[msg], stop_reason="end_turn", text=json.dumps(obj))

    def _post(self, agent: str, payload: dict[str, Any], telemetry: RunTelemetryCounter,
              schema: dict[str, Any]) -> dict[str, Any] | LLMResponse:
        blocked = COOLDOWN.check(self._cooldown_key)
        if blocked:
            raise LLMUnavailable(f"{self.provider} quota exhausted: {blocked}")
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        est_tokens = len(json.dumps(payload)) // 4 + 800  # ~4 chars/token + reply allowance; settled to real usage after
        deadline = time.monotonic() + self._max_wait
        attempt = 0
        while True:
            attempt += 1
            handle = self._limiter.acquire(est_tokens)
            started = time.perf_counter()
            try:
                r = requests.post(self._url, headers=headers, json=payload, timeout=self._timeout)
            except requests.Timeout as exc:
                raise LLMUnavailable(f"{self.provider} request timed out after {self._timeout:.0f}s") from exc
            except requests.RequestException as exc:
                raise LLMUnavailable(f"Cannot reach {self.provider}: {type(exc).__name__}") from exc

            if r.status_code == 429 or r.status_code >= 500:
                self._limiter.settle(handle, 0)
                body = r.text[:600]
                wait = None
                if r.headers.get("retry-after"):
                    try:
                        wait = float(r.headers["retry-after"])
                    except ValueError:
                        wait = None
                wait = _parse_wait(body) or wait
                daily = bool(re.search(r"per day|\bTPD\b|\bRPD\b|daily", body, flags=re.IGNORECASE))
                telemetry.event("llm_retry", agent=agent, code=r.status_code, wait_s=wait, daily=daily)
                if r.status_code == 429 and (daily or (wait is not None and wait > self._max_wait)):
                    COOLDOWN.block(self._cooldown_key, wait or 3600, f"{self.provider} {self.model} hit its {'daily ' if daily else ''}rate limit")
                    raise LLMUnavailable(f"{self.provider} quota exhausted ({'daily limit' if daily else f'retry in {wait:.0f}s'})")
                wait = wait if wait is not None else min(2.0 ** attempt, 20.0)
                if time.monotonic() + wait > deadline or attempt >= 6:
                    raise LLMUnavailable(f"{self.provider} HTTP {r.status_code} persisted after {attempt} attempts")
                time.sleep(wait)
                continue

            if r.status_code >= 400:
                self._limiter.settle(handle, 0)
                err = {}
                try:
                    err = (r.json() or {}).get("error") or {}
                except ValueError:
                    pass
                if r.status_code == 400 and err.get("code") == "tool_use_failed":
                    obj = _unwrap(extract_json_object(err.get("failed_generation") or ""))
                    telemetry.event("tool_use_failed", agent=agent, recovered=bool(obj and has_required_keys(obj, schema)))
                    if obj is not None and has_required_keys(obj, schema):
                        telemetry.record_llm_call(agent, ms=(time.perf_counter() - started) * 1000, stop_reason="tool_use_failed_recovered")
                        return LLMResponse(content=[{"role": "assistant", "content": json.dumps(obj)}], stop_reason="end_turn", text=json.dumps(obj))
                    if attempt < 2:
                        continue
                if r.status_code in (401, 403):
                    raise LLMUnavailable(f"{self.provider} rejected the API key (HTTP {r.status_code}) - check {self.provider.upper()}_API_KEY")
                raise LLMUnavailable(f"{self.provider} HTTP {r.status_code}: {str(err.get('message') or r.text)[:200]}")

            try:
                body = r.json()
            except ValueError as exc:
                raise LLMUnavailable(f"{self.provider} returned a non-JSON body") from exc
            usage = body.get("usage") or {}
            self._limiter.settle(handle, (usage.get("total_tokens") or 0) or None)
            choice = (body.get("choices") or [{}])[0]
            telemetry.record_llm_call(agent, input_tokens=usage.get("prompt_tokens", 0) or 0,
                                      output_tokens=usage.get("completion_tokens", 0) or 0,
                                      ms=(time.perf_counter() - started) * 1000, stop_reason=choice.get("finish_reason"))
            if choice.get("finish_reason") == "length":
                raise LLMUnavailable(f"{self.provider} output truncated (finish_reason=length)")
            msg = choice.get("message")
            if not isinstance(msg, dict):
                raise LLMUnavailable(f"{self.provider} returned no message")
            # keep extra_content: Gemini-3-via-OpenAI thought signatures must be replayed next turn
            return {k: v for k, v in msg.items() if k in {"role", "content", "tool_calls", "extra_content"}}

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
