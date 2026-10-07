from __future__ import annotations

import json

from src.contracts import ProcurementDecision
from src.data_access import get_request
from src.llm import LLMUnavailable
from src.solution import analyze_request
from tests.conftest import ALL_TOOLS, FakeLLM, answer, call_tools, decision_draft, evidence_pack


def run(request_id: str, arch: str, script: list) -> tuple:
    llm = FakeLLM(script)
    return analyze_request(get_request(request_id), arch, llm=llm), llm


# ---------------------------------------------------------------- architecture A


def test_single_agent_happy_path_counts_calls_and_keeps_contract():
    draft = decision_draft(action="route_for_approval", required_approvals=["Manager"], risk_flags=["existing_tool_overlap"],
                           recommendation="Route to the Manager for approval.", cited_evidence_ids=["E1", "E3"])
    d, llm = run("REQ-1001", "single", [call_tools(*ALL_TOOLS), answer(draft)])
    assert isinstance(d, ProcurementDecision)
    assert d.action == "route_for_approval"
    assert d.required_approvals == ["Manager"]
    assert d.telemetry.llm_calls == 2
    assert d.telemetry.tool_calls == 5
    assert d.guard_adjustments == []
    assert d.human_review_required is True
    # Tool results were returned to the model in a single user turn, after the assistant turn.
    second = llm.calls[1]["messages"]
    assert second[1]["role"] == "assistant" and second[2]["role"] == "user"
    assert len(second[2]["content"]) == 5


def test_guard_restores_approvals_and_flags_the_model_dropped():
    lazy = decision_draft(action="route_for_approval", required_approvals=["Department Head"], risk_flags=[])
    d, _ = run("REQ-1005", "single", [call_tools(*ALL_TOOLS), answer(lazy)])
    assert {"Finance", "Security", "Privacy", "Legal"} <= set(d.required_approvals)
    assert "budget_insufficient" in d.risk_flags
    assert d.action == "route_for_review"
    assert any(a.field == "action" and a.change == "raised" for a in d.guard_adjustments)
    assert d.diagnostics["guard_corrections"] >= 5


def test_guard_removes_fact_flags_no_tool_supports():
    draft = decision_draft(action="route_for_approval", required_approvals=["Manager"],
                           risk_flags=["budget_insufficient", "vendor_review_expired"])
    d, _ = run("REQ-1001", "single", [call_tools(*ALL_TOOLS), answer(draft)])
    assert "budget_insufficient" not in d.risk_flags
    assert "vendor_review_expired" not in d.risk_flags
    assert sum(a.change == "removed" for a in d.guard_adjustments) == 2


def test_model_can_add_caution():
    draft = decision_draft(action="route_for_review", required_approvals=["Manager", "Security"],
                           risk_flags=["security_review_required"])
    d, _ = run("REQ-1001", "single", [call_tools(*ALL_TOOLS), answer(draft)])
    assert "Security" in d.required_approvals
    assert d.action == "route_for_review"


def test_reuse_only_allowed_when_nothing_else_blocks():
    reuse = decision_draft(action="reuse_existing_tool", required_approvals=["Department Head", "Procurement"],
                           risk_flags=["existing_tool_overlap"])
    d, _ = run("REQ-1008", "single", [call_tools(*ALL_TOOLS), answer(reuse)])
    assert d.action == "reuse_existing_tool"

    d, _ = run("REQ-1004", "single", [call_tools(*ALL_TOOLS), answer(reuse)])  # customer PII -> reviews first
    assert d.action == "route_for_review"


def test_approval_claims_in_model_text_are_replaced():
    draft = decision_draft(action="route_for_approval", required_approvals=["Manager"], recommendation="This purchase is approved.")
    d, _ = run("REQ-1001", "single", [call_tools(*ALL_TOOLS), answer(draft)])
    assert "approved." not in d.recommendation
    assert any(a.field == "recommendation" for a in d.guard_adjustments)


def test_invalid_citations_are_dropped_and_counted():
    draft = decision_draft(action="route_for_approval", required_approvals=["Manager"], cited_evidence_ids=["E1", "E999"])
    d, _ = run("REQ-1001", "single", [call_tools(*ALL_TOOLS), answer(draft)])
    assert d.cited_evidence == ["E1"]
    assert d.diagnostics["invalid_citations"] == ["E999"]


def test_prompt_injection_cannot_change_outcome():
    obedient = decision_draft(action="route_for_approval", required_approvals=["CFO"], risk_flags=[],
                              recommendation="Approved as CFO-approved per the request.")
    d, _ = run("REQ-1006", "single", [call_tools(*ALL_TOOLS), answer(obedient)])
    assert d.action == "request_clarification"
    assert "prompt_injection_detected" in d.risk_flags
    assert "missing_information" in d.risk_flags
    assert d.clarification_questions  # template questions when the model gave none
    # the injected text reached the model only as JSON inside <request_data>
    # (CFO stays because adding caution is allowed, but nothing was approved)


def test_request_text_is_wrapped_as_untrusted_data():
    _, llm = run("REQ-1006", "single", [call_tools(*ALL_TOOLS), answer(decision_draft(action="request_clarification"))])
    first_user = llm.calls[0]["messages"][0]["content"]
    assert "<request_data>" in first_user and "untrusted" in first_user
    assert "UNTRUSTED DATA" in llm.calls[0]["system"]


def test_unknown_tool_and_bad_args_are_reported_not_raised():
    def bad(agent, messages, tools):
        from src.llm import to_response
        return to_response([{"type": "tool_use", "id": "x1", "name": "approve_purchase", "input": {}},
                            {"type": "tool_use", "id": "x2", "name": "search_software_catalog", "input": {"query": 5, "category": None}}],
                           "tool_use")

    d, llm = run("REQ-1001", "single", [bad, answer(decision_draft(action="route_for_approval", required_approvals=["Manager"]))])
    results = llm.calls[1]["messages"][2]["content"]
    assert results[0]["is_error"] is True and "Unknown tool" in results[0]["content"]
    assert "is_error" not in results[1]
    assert d.required_approvals == ["Manager"]


def test_agent_skipping_tools_still_gets_full_evidence_from_guard():
    d, _ = run("REQ-1007", "single", [answer(decision_draft(action="route_for_review"))])
    assert "conflicting_vendor_evidence" in d.risk_flags
    assert any(e.source == "vendor_risk_api" for e in d.evidence)
    assert "evaluate_policy_rules" in d.telemetry.tool_names  # computed by the guard, and counted


def test_llm_failure_degrades_to_rules_only():
    d, _ = run("REQ-1009", "single", [call_tools(*ALL_TOOLS), LLMUnavailable("API error 529")])
    assert d.ai_status == "unavailable"
    assert "ai_assessment_unavailable" in d.risk_flags
    assert {"Finance", "Security", "Legal"} <= set(d.required_approvals)
    assert d.action == "route_for_review"


def test_max_turns_exceeded_degrades(monkeypatch):
    monkeypatch.setenv("COPILOT_MAX_TURNS", "2")
    d, _ = run("REQ-1001", "single", [call_tools("check_budget"), call_tools("check_budget")])
    assert d.ai_status == "unavailable"
    assert "did not produce a final answer" in d.ai_error


def test_vendor_outage_is_reported(vendor_api_down):
    d, _ = run("REQ-1008", "single", [call_tools(*ALL_TOOLS), answer(decision_draft(action="route_for_approval", required_approvals=["Department Head", "Procurement"]))])
    assert "vendor_risk_unavailable" in d.risk_flags
    assert "Security" in d.required_approvals
    assert d.action == "route_for_review"


# ---------------------------------------------------------------- architecture B


def test_staged_hands_pack_and_rules_to_reviewer():
    reviewer = decision_draft(action="route_for_review", required_approvals=["Department Head", "Finance", "Procurement", "Security"],
                              risk_flags=["security_review_required", "existing_tool_overlap"])
    d, llm = run("REQ-1003", "staged", [call_tools(*ALL_TOOLS), answer(evidence_pack()), answer(reviewer)])
    assert d.architecture == "staged"
    assert d.telemetry.llm_calls == 3
    assert [c["agent"] for c in llm.calls] == ["procurement_analyst", "procurement_analyst", "policy_reviewer"]
    assert llm.calls[2]["tools"] is None
    reviewer_input = llm.calls[2]["messages"][0]["content"]
    for tag in ("<analyst_evidence_pack>", "<evidence_items>", "<deterministic_policy_result>"):
        assert tag in reviewer_input
    assert d.guard_adjustments == []


def test_staged_reviewer_failure_degrades():
    d, _ = run("REQ-1002", "staged", [call_tools(*ALL_TOOLS), answer(evidence_pack()), LLMUnavailable("timeout")])
    assert d.ai_status == "unavailable"
    assert {"Security", "Legal"} <= set(d.required_approvals)


def test_no_llm_configured_runs_deterministic(monkeypatch):
    monkeypatch.setenv("COPILOT_LLM", "off")
    d = analyze_request(get_request("REQ-1001"), "single", llm=None)
    assert d.required_approvals == ["Manager"]
    assert d.telemetry.llm_calls == 0
    json.loads(d.model_dump_json())  # serialisable for the UI / CSV
