"""Provider adapters exercised end-to-end with mocked transports (no network, no keys)."""
from __future__ import annotations

import json

import pytest
from google.genai import errors, types

from src import llm
from src.data_access import get_request
from src.llm_gemini import GeminiLLM
from src.llm_openai_compat import OpenAICompatLLM
from src.solution import analyze_request
from tests.conftest import ALL_TOOLS, decision_draft, evidence_pack


def gemini_response(parts: list[types.Part]) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts), finish_reason="STOP")],
        usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=1000, candidates_token_count=100),
    )


def fc(name: str, args: dict | None = None) -> types.Part:
    return types.Part(function_call=types.FunctionCall(name=name, args=args or {}))


class ScriptedModels:
    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    def generate_content(self, *, model, contents, config):
        self.requests.append({"model": model, "contents": list(contents), "config": config})
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


class FakeClient:
    def __init__(self, models):
        self.models = models


def make_gemini(monkeypatch, *scripts) -> tuple[GeminiLLM, list[ScriptedModels]]:
    monkeypatch.setenv("GEMINI_RPM", "100000")  # no throttling in tests
    g = GeminiLLM(keys=[f"k{i}" for i in range(len(scripts))], model="gemini-3.7-flash")
    models = [ScriptedModels(s) for s in scripts]
    g._clients = [FakeClient(m) for m in models]
    return g, models


def test_gemini_single_agent_tool_loop_then_json_text(monkeypatch):
    draft = decision_draft(action="route_for_review", required_approvals=["Department Head", "Finance", "Procurement", "Security"],
                           risk_flags=["security_review_required", "existing_tool_overlap"], cited_evidence_ids=["E1"])
    g, (m,) = make_gemini(monkeypatch, [
        gemini_response([fc(n, {"query": None, "category": None} if n == "search_software_catalog" else ({"vendor_name": None} if n == "get_vendor_risk" else {})) for n in ALL_TOOLS]),
        gemini_response([types.Part(text="```json\n" + json.dumps(draft) + "\n```")]),
    ])
    d = analyze_request(get_request("REQ-1003"), "single", llm=g)
    assert d.ai_status == "ok", d.ai_error
    assert d.telemetry.llm_calls == 2 and d.telemetry.tool_calls == 5
    assert d.guard_adjustments == []
    # tool turn: declarations sent, no response schema mixed in
    first_cfg = m.requests[0]["config"]
    assert first_cfg.tools and first_cfg.response_json_schema is None
    names = [f.name for f in first_cfg.tools[0].function_declarations]
    assert names == list(ALL_TOOLS)
    # second turn replays the model's function-call content and returns function responses
    second = m.requests[1]["contents"]
    assert second[1].role == "model" and second[1].parts[0].function_call.name == "lookup_requester"
    assert second[2].role == "user" and second[2].parts[0].function_response.name == "lookup_requester"
    assert "output" in second[2].parts[0].function_response.response


def test_gemini_prose_answer_triggers_structured_follow_up(monkeypatch):
    draft = decision_draft(action="route_for_approval", required_approvals=["Manager"])
    g, (m,) = make_gemini(monkeypatch, [
        gemini_response([fc("evaluate_policy_rules")]),
        gemini_response([types.Part(text="Looks fine, route it to the manager.")]),
        gemini_response([types.Part(text=json.dumps(draft))]),
    ])
    d = analyze_request(get_request("REQ-1001"), "single", llm=g)
    assert d.ai_status == "ok"
    assert d.telemetry.llm_calls == 3
    final_cfg = m.requests[2]["config"]
    assert final_cfg.response_mime_type == "application/json" and final_cfg.response_json_schema is not None
    assert final_cfg.tools is None


def test_gemini_staged_reviewer_uses_structured_call(monkeypatch):
    g, (m,) = make_gemini(monkeypatch, [
        gemini_response([fc("evaluate_policy_rules")]),
        gemini_response([types.Part(text=json.dumps(evidence_pack()))]),
        gemini_response([types.Part(text=json.dumps(decision_draft(action="route_for_review",
                                                                   required_approvals=["Department Head", "Procurement", "Finance", "Security", "Legal"])))]),
    ])
    d = analyze_request(get_request("REQ-1002"), "staged", llm=g)
    assert d.ai_status == "ok" and d.telemetry.llm_calls == 3
    assert m.requests[2]["config"].response_json_schema is not None


def test_gemini_429_rotates_key_and_retries(monkeypatch):
    monkeypatch.setattr("src.llm_gemini.time.sleep", lambda s: None)
    quota = errors.ClientError(429, {"error": {"code": 429, "message": "RESOURCE_EXHAUSTED retryDelay: 2s", "status": "RESOURCE_EXHAUSTED"}})
    draft = decision_draft(action="route_for_approval", required_approvals=["Manager"])
    g, (m0, m1) = make_gemini(monkeypatch,
                              [quota],
                              [gemini_response([types.Part(text=json.dumps(draft))])])
    d = analyze_request(get_request("REQ-1001"), "single", llm=g)
    assert d.ai_status == "ok"
    assert len(m0.requests) == 1 and len(m1.requests) == 1
    assert any(e["kind"] == "llm_retry" and e["key_index"] == 0 for e in d.diagnostics["trace"])


def test_gemini_auth_error_degrades_to_rules(monkeypatch):
    bad_key = errors.ClientError(400, {"error": {"code": 400, "message": "API key not valid", "status": "INVALID_ARGUMENT"}})
    g, _ = make_gemini(monkeypatch, [bad_key])
    d = analyze_request(get_request("REQ-1009"), "single", llm=g)
    assert d.ai_status == "unavailable" and "400" in d.ai_error
    assert {"Security", "Legal"} <= set(d.required_approvals)


def test_gemini_blocked_response_degrades(monkeypatch):
    blocked = types.GenerateContentResponse(candidates=[types.Candidate(content=None, finish_reason="SAFETY")])
    g, _ = make_gemini(monkeypatch, [blocked])
    d = analyze_request(get_request("REQ-1001"), "single", llm=g)
    assert d.ai_status == "unavailable" and "SAFETY" in d.ai_error


# ---------------------------------------------------------------- OpenAI-compatible (CloseRouter)


class FakeHTTP:
    def __init__(self, replies):
        self.replies = list(replies)
        self.payloads = []

    def __call__(self, url, headers, json, timeout):
        self.payloads.append(json)
        status, body = self.replies.pop(0)

        class R:
            status_code = status
            text = str(body)

            def json(self_inner):
                return body
        return R()


def chat(message: dict) -> dict:
    return {"choices": [{"message": message, "finish_reason": "stop"}], "usage": {"prompt_tokens": 900, "completion_tokens": 80}}


def test_openai_compat_tool_loop(monkeypatch):
    monkeypatch.setattr("src.llm_openai_compat.time.sleep", lambda s: None)
    draft = decision_draft(action="reuse_existing_tool", required_approvals=["Department Head", "Procurement"], risk_flags=["existing_tool_overlap"])
    http = FakeHTTP([
        (429, {"error": "slow down"}),
        (200, chat({"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "evaluate_policy_rules", "arguments": "{}"}},
            {"id": "c2", "type": "function", "function": {"name": "search_software_catalog", "arguments": "{\"query\": \"task tracker\", \"category\": null}"}}]})),
        (200, chat({"role": "assistant", "content": json.dumps(draft)})),
    ])
    monkeypatch.setattr("src.llm_openai_compat.requests.post", http)
    client = OpenAICompatLLM(api_key="x", model="google/gemini-3.7-flash", base_url="https://example.invalid/v1")
    d = analyze_request(get_request("REQ-1008"), "single", llm=client)
    assert d.ai_status == "ok" and d.action == "reuse_existing_tool"
    assert d.telemetry.llm_calls == 2 and d.telemetry.tool_calls == 2
    tool_msgs = [m for m in http.payloads[2]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2"]


# ---------------------------------------------------------------- provider selection


@pytest.mark.parametrize("env, expected", [
    ({"GEMINI_API_KEY": "g"}, "gemini"),
    ({"GEMINI_API_KEY_POOL": "a,b"}, "gemini"),
    ({"CLOSEROUTER_API_KEY": "c"}, "closerouter"),
    ({"ANTHROPIC_API_KEY": "a"}, "anthropic"),
    ({"GEMINI_API_KEY": "g", "ANTHROPIC_API_KEY": "a"}, "gemini"),
    ({"GEMINI_API_KEY": "g", "ANTHROPIC_API_KEY": "a", "LLM_PROVIDER": "anthropic"}, "anthropic"),
    ({"LLM_PROVIDER": "gemini"}, None),
    ({"GEMINI_API_KEY": "g", "COPILOT_LLM": "off"}, None),
    ({}, None),
])
def test_provider_selection(monkeypatch, env, expected):
    for k in ("GEMINI_API_KEY", "GEMINI_API_KEY_POOL", "GOOGLE_API_KEY", "CLOSEROUTER_API_KEY", "ANTHROPIC_API_KEY",
              "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "LLM_PROVIDER", "COPILOT_LLM"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert llm.configured_provider() == expected


@pytest.mark.parametrize("text, ok", [
    ('{"a": 1}', True), ('```json\n{"a": 1}\n```', True), ('Sure! {"a": 1} done', True), ("no json", False), ("[1,2]", False),
])
def test_extract_json_object(text, ok):
    assert (llm.extract_json_object(text) is not None) == ok
