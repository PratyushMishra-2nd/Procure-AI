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
        reply = self.replies.pop(0)
        status, body = reply[0], reply[1]
        resp_headers = reply[2] if len(reply) > 2 else {}

        class R:
            status_code = status
            text = body if isinstance(body, str) else __import__("json").dumps(body)
            headers = resp_headers

            def json(self_inner):
                return body
        return R()


def chat(message: dict) -> dict:
    return {"choices": [{"message": message, "finish_reason": "stop"}], "usage": {"prompt_tokens": 900, "completion_tokens": 80}}


def test_openai_compat_tool_loop(monkeypatch):
    monkeypatch.setattr("src.llm_openai_compat.time.sleep", lambda s: None)
    draft = decision_draft(action="reuse_existing_tool", overlap_assessment="existing_tool_likely_sufficient", required_approvals=["Department Head", "Procurement"], risk_flags=["existing_tool_overlap"])
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
    ({"GROQ_API_KEY": "q"}, "groq"),
    ({"GROQ_API_KEY": "q", "CLOSEROUTER_API_KEY": "c"}, "groq"),
    ({"GEMINI_API_KEY": "g", "GROQ_API_KEY": "q", "LLM_PROVIDER": "groq"}, "groq"),
    ({"ANTHROPIC_API_KEY": "a"}, "anthropic"),
    ({"GEMINI_API_KEY": "g", "ANTHROPIC_API_KEY": "a"}, "gemini"),
    ({"GEMINI_API_KEY": "g", "ANTHROPIC_API_KEY": "a", "LLM_PROVIDER": "anthropic"}, "anthropic"),
    ({"LLM_PROVIDER": "gemini"}, None),
    ({"GEMINI_API_KEY": "g", "COPILOT_LLM": "off"}, None),
    ({}, None),
])
def test_provider_selection(monkeypatch, env, expected):
    for k in ("GEMINI_API_KEY", "GEMINI_API_KEY_POOL", "GOOGLE_API_KEY", "GROQ_API_KEY", "CLOSEROUTER_API_KEY", "ANTHROPIC_API_KEY",
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


def test_gemini_daily_quota_fails_fast_and_skips_later_calls(monkeypatch):
    slept = []
    monkeypatch.setattr("src.llm_gemini.time.sleep", lambda s: slept.append(s))
    daily = errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message":
        "Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20. Please retry in 15h36m20.8s."}})
    g, (m,) = make_gemini(monkeypatch, [daily])
    d1 = analyze_request(get_request("REQ-1001"), "single", llm=g)
    d2 = analyze_request(get_request("REQ-1002"), "single", llm=g)
    assert "daily request quota" in d1.ai_error and "daily request quota" in d2.ai_error
    assert len(m.requests) == 1  # second request never hit the API
    assert not [s for s in slept if s > 1]  # no long backoff waits


# ---------------------------------------------------------------- Groq quirks, pacing, cooldown


def groq_client(monkeypatch, http):
    from src import ratelimit
    import src.llm_openai_compat as oc

    monkeypatch.setattr(oc.requests, "post", http)
    monkeypatch.setattr(oc.time, "sleep", lambda s: None)
    monkeypatch.setattr(oc, "_LIMITERS", {})
    monkeypatch.setattr(oc, "COOLDOWN", ratelimit.QuotaCooldown())
    monkeypatch.setenv("LLM_RPM", "100000")
    monkeypatch.setenv("LLM_TPM", "0")
    return OpenAICompatLLM(api_key="x", preset="groq")


def test_groq_undeclared_json_tool_is_read_as_answer_not_executed(monkeypatch):
    draft = decision_draft(action="route_for_approval", required_approvals=["Manager"])
    http = FakeHTTP([
        (200, chat({"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "evaluate_policy_rules", "arguments": "{}"}}]})),
        (200, chat({"role": "assistant", "content": None, "tool_calls": [
            {"id": "c2", "type": "function", "function": {"name": "json", "arguments": json.dumps(draft)}}]})),
    ])
    d = analyze_request(get_request("REQ-1001"), "single", llm=groq_client(monkeypatch, http))
    assert d.ai_status == "ok" and d.provider == "groq"
    assert "json" not in d.telemetry.tool_names
    assert any(e["kind"] == "undeclared_tool_as_answer" for e in d.diagnostics["trace"])


def test_groq_undeclared_tool_without_answer_is_reported_as_error(monkeypatch):
    draft = decision_draft(action="route_for_approval", required_approvals=["Manager"])
    http = FakeHTTP([
        (200, chat({"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "approve_purchase", "arguments": "{}"}}]})),
        (200, chat({"role": "assistant", "content": json.dumps(draft)})),
    ])
    d = analyze_request(get_request("REQ-1001"), "single", llm=groq_client(monkeypatch, http))
    assert d.ai_status == "ok"
    tool_msg = [m for m in http.payloads[1]["messages"] if m["role"] == "tool"][0]
    assert "Unknown tool" in tool_msg["content"]


def test_groq_tool_use_failed_recovers_failed_generation(monkeypatch):
    draft = decision_draft(action="route_for_approval", required_approvals=["Manager"])
    http = FakeHTTP([(400, {"error": {"code": "tool_use_failed", "message": "Failed to call a function",
                                      "failed_generation": "<json>" + json.dumps(draft) + "</json>"}})])
    d = analyze_request(get_request("REQ-1001"), "single", llm=groq_client(monkeypatch, http))
    assert d.ai_status == "ok" and d.action == "route_for_approval"


def test_groq_daily_limit_cools_down_and_fails_fast(monkeypatch):
    body = {"error": {"message": "Rate limit reached for model `openai/gpt-oss-120b` on tokens per day (TPD): Limit 200000, Used 199000. Please try again in 7m12.5s.", "code": "rate_limit_exceeded"}}
    http = FakeHTTP([(429, body, {"retry-after": "433"})])
    client = groq_client(monkeypatch, http)
    d1 = analyze_request(get_request("REQ-1001"), "single", llm=client)
    d2 = analyze_request(get_request("REQ-1002"), "single", llm=client)
    assert "quota exhausted" in d1.ai_error and "quota exhausted" in d2.ai_error
    assert len(http.payloads) == 1  # second request never sent


def test_groq_per_minute_limit_waits_and_retries(monkeypatch):
    draft = decision_draft(action="route_for_approval", required_approvals=["Manager"])
    body = {"error": {"message": "Rate limit reached on tokens per minute (TPM): Limit 8000. Please try again in 2.5s.", "code": "rate_limit_exceeded"}}
    http = FakeHTTP([(429, body), (200, chat({"role": "assistant", "content": json.dumps(draft)}))])
    d = analyze_request(get_request("REQ-1001"), "single", llm=groq_client(monkeypatch, http))
    assert d.ai_status == "ok" and len(http.payloads) == 2


def test_auth_error_is_not_retried(monkeypatch):
    http = FakeHTTP([(401, {"error": {"message": "Invalid API Key"}})])
    d = analyze_request(get_request("REQ-1001"), "single", llm=groq_client(monkeypatch, http))
    assert d.ai_status == "unavailable" and "rejected the API key" in d.ai_error and len(http.payloads) == 1


def test_extra_content_thought_signature_is_replayed(monkeypatch):
    draft = decision_draft(action="route_for_approval", required_approvals=["Manager"])
    sig = {"google": {"thought_signature": "sig-123"}}
    http = FakeHTTP([
        (200, chat({"role": "assistant", "content": None, "extra_content": sig, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "check_budget", "arguments": "{}"}, "extra_content": sig}]})),
        (200, chat({"role": "assistant", "content": json.dumps(draft)})),
    ])
    analyze_request(get_request("REQ-1001"), "single", llm=groq_client(monkeypatch, http))
    replayed = [m for m in http.payloads[1]["messages"] if m["role"] == "assistant"][0]
    assert replayed["extra_content"] == sig and replayed["tool_calls"][0]["extra_content"] == sig


def test_no_arg_tools_get_valid_schemas_for_every_provider():
    from src.tools import TOOL_SPECS
    g = GeminiLLM.__new__(GeminiLLM)
    g._types = types
    for spec in TOOL_SPECS:
        decl = g._declaration(spec)
        if spec["input_schema"]["properties"]:
            assert decl.parameters_json_schema["required"] == list(spec["input_schema"]["properties"])
        else:
            assert decl.parameters_json_schema is None  # Gemini rejects an empty-properties object
        assert spec["input_schema"]["type"] == "object"  # OpenAI-style providers get an explicit empty object


def test_misconfigured_forced_provider_is_explained(monkeypatch):
    for k in ("GEMINI_API_KEY", "GEMINI_API_KEY_POOL", "GOOGLE_API_KEY", "GROQ_API_KEY", "CLOSEROUTER_API_KEY", "ANTHROPIC_API_KEY", "COPILOT_LLM"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    assert llm.provider_config_problem() == "LLM_PROVIDER=groq but GROQ_API_KEY is not set"


def test_rate_limiter_respects_tokens_per_minute():
    from src.ratelimit import RateLimiter
    now = [0.0]
    slept = []

    def sleep(s):
        slept.append(s)
        now[0] += s
    rl = RateLimiter(rpm=None, tpm=8000, sleep=sleep, clock=lambda: now[0])
    rl.acquire(5000)
    rl.acquire(5000)  # would exceed 8000 within the minute -> waits ~60s
    assert slept and 59 <= sum(slept) <= 61


def test_groq_answers_through_declared_final_answer_tool(monkeypatch):
    draft = decision_draft(action="route_for_approval", required_approvals=["Manager"])
    http = FakeHTTP([
        (200, chat({"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "evaluate_policy_rules", "arguments": "{}"}}]})),
        (200, chat({"role": "assistant", "content": None, "tool_calls": [
            {"id": "c2", "type": "function", "function": {"name": "final_answer", "arguments": json.dumps(draft)}}]})),
    ])
    d = analyze_request(get_request("REQ-1001"), "single", llm=groq_client(monkeypatch, http))
    assert d.ai_status == "ok" and d.telemetry.llm_calls == 2
    assert "final_answer" not in d.telemetry.tool_names  # submitted, never executed
    sent = [t["function"]["name"] for t in http.payloads[0]["tools"]]
    assert sent[-1] == "final_answer" and "evaluate_policy_rules" in sent


def test_groq_structured_call_forces_final_answer_tool(monkeypatch):
    pack = evidence_pack()
    review = decision_draft(action="route_for_review", required_approvals=["Department Head", "Procurement", "Finance", "Security", "Legal"])
    http = FakeHTTP([
        (200, chat({"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "final_answer", "arguments": json.dumps(pack)}}]})),
        (200, chat({"role": "assistant", "content": None, "tool_calls": [
            {"id": "c2", "type": "function", "function": {"name": "final_answer", "arguments": json.dumps(review)}}]})),
    ])
    d = analyze_request(get_request("REQ-1002"), "staged", llm=groq_client(monkeypatch, http))
    assert d.ai_status == "ok"
    assert http.payloads[1]["tool_choice"]["function"]["name"] == "final_answer"


def test_groq_failed_generation_wrapper_is_unwrapped(monkeypatch):
    draft = decision_draft(action="route_for_approval", required_approvals=["Manager"])
    wrapped = json.dumps({"name": "json", "arguments": draft})
    http = FakeHTTP([(400, {"error": {"code": "tool_use_failed", "message": "attempted to call tool 'json' which was not in request.tools",
                                      "failed_generation": wrapped}})])
    d = analyze_request(get_request("REQ-1001"), "single", llm=groq_client(monkeypatch, http))
    assert d.ai_status == "ok" and d.action == "route_for_approval"
