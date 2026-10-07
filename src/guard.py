"""Policy guard: the last step of both architectures.

The model's draft is merged with the deterministic policy result under fixed
rules, and every place the guard had to step in is recorded as a
``GuardAdjustment``. Those adjustments are the eval's "policy failures caught"
signal: they show what the model would have got wrong without the guard.

Merge rules
- Approvals, risk flags and missing information: union. The model can add caution,
  never remove a requirement.
- Fact flags (budget, vendor freshness/conflict/outage, onboarding) are only kept if
  the deterministic checks raised them - a model cannot assert a fact the tools did
  not return.
- Action: the model's choice if it is at least as cautious as the rule floor.
- Human review is always required (policy section 11).
"""
from __future__ import annotations

import re
from typing import Any

from src import policy_engine as pe
from src.contracts import CopilotDecision, GuardAdjustment, RunTelemetry
from src.tools import ToolContext

FACT_FLAGS = {
    "budget_insufficient",
    "budget_unverified",
    "vendor_review_expired",
    "conflicting_vendor_evidence",
    "vendor_risk_unavailable",
    "vendor_not_onboarded",
}
FLAG_TO_APPROVAL = {
    "security_review_required": "Security",
    "privacy_review_required": "Privacy",
    "legal_review_required": "Legal",
}
SPECIALISTS = ("Security", "Privacy", "Legal")
SHORT_FIELD = {
    "requester": "requester / department",
    "product_vendor": "product / vendor",
    "annual_cost": "annual cost",
    "user_count": "number of users / seats",
    "business_purpose": "business purpose",
    "data_access_level": "data-access level",
    "integrations": "required integrations",
}
QUESTION_FOR_FIELD = {
    "requester": "Which employee ID and department is this request for?",
    "product_vendor": "Which exact product and vendor are you requesting?",
    "annual_cost": "What is the annual cost (or a reasonable annual estimate) from the vendor quote?",
    "user_count": "How many users / licenses are needed?",
    "business_purpose": "What concrete business problem will this solve, and why can't an existing approved tool cover it?",
    "data_access_level": "Which data will the tool access (e.g. none, internal documents, confidential documents, employee PII, customer PII, source code, production systems)?",
    "integrations": "Which systems must it integrate with (or confirm none)?",
}
APPROVAL_CLAIM = re.compile(r"\b(auto-?approved?|approved for purchase|purchase (is )?approved|has been approved|is approved|i approve|we approve)\b", re.I)


def _join(items: list[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _business_approvers(approvals: list[str]) -> list[str]:
    return [a for a in approvals if a not in SPECIALISTS]


def template_recommendation(action: str, approvals: list[str], flags: list[str], missing: dict[str, str],
                            overlap_names: list[str]) -> str:
    specialists = [a for a in approvals if a in SPECIALISTS]
    business = _business_approvers(approvals)
    if action == "request_clarification":
        return f"Return to the requester for clarification before routing: missing {_join([SHORT_FIELD.get(k, k) for k in missing])}."
    if action == "reuse_existing_tool":
        return f"Check whether the existing {_join(overlap_names) or 'approved tool'} already covers this need before buying; route to {_join(business)} only if a documented gap remains."
    if action == "route_for_review":
        parts = []
        if specialists:
            parts.append(f"{_join(specialists)} review")
        if "budget_insufficient" in flags or "budget_unverified" in flags:
            parts.append("a Finance budget-exception review")
        if "vendor_risk_unavailable" in flags or "conflicting_vendor_evidence" in flags:
            parts.append("manual verification of the vendor's risk status")
        return f"Route for {_join(parts) or 'manual review'} before {_join(business) or 'the business approvers'} decide."
    return f"Route to {_join(business)} for approval; no specialist review is triggered."


def template_next_step(action: str, approvals: list[str], facts: dict[str, Any], overlap_names: list[str]) -> str:
    business = _business_approvers(approvals)
    specialists = [a for a in approvals if a in SPECIALISTS]
    requester = (facts.get("requester") or {}).get("requester") or {}
    who = f" ({requester['name']})" if requester.get("name") else ""
    if action == "request_clarification":
        return f"Procurement sends the clarification questions to the requester{who}; re-run the copilot once answered. No approval routing until then."
    if action == "reuse_existing_tool":
        return f"Procurement confirms with the owner of {_join(overlap_names) or 'the existing tool'} whether spare seats or scope cover the need; if not, send the evidence pack to {_join(business)}."
    if action == "route_for_review":
        first = _join(specialists) if specialists else "Finance / Procurement"
        return f"Procurement sends this evidence pack to {first}; {_join(business) or 'the approvers'} decide only after those reviews complete."
    return f"Send this evidence pack to {_join(business)} for a decision."


def finalize(ctx: ToolContext, draft: dict[str, Any] | None, *, architecture: str, ai_status: str,
             ai_error: str | None, model: str | None, extra_diagnostics: dict[str, Any] | None = None) -> CopilotDecision:
    policy = ctx.policy(caller="guard")
    facts = policy.facts
    adjustments: list[GuardAdjustment] = []

    def adjust(field: str, change: str, detail: str) -> None:
        adjustments.append(GuardAdjustment(field=field, change=change, detail=detail))

    # Evidence back-fill: every deterministic finding is shown even if an agent skipped a tool.
    for finding in policy.findings:
        ctx.add_evidence(finding)

    draft = draft or {}
    has_draft = bool(draft)

    # ---- approvals
    approval_reasons = {a: list(r) for a, r in policy.approval_reasons.items()}
    draft_approvals = [a for a in draft.get("required_approvals", []) if a in pe.APPROVAL_ORDER]
    if has_draft:
        for a in policy.required_approvals:
            if a not in draft_approvals:
                adjust("required_approvals", "added", f"{a} - {policy.approval_reasons[a][0]}")
        for a in draft_approvals:
            if a not in approval_reasons:
                approval_reasons[a] = ["Added by AI review as extra caution"]

    # ---- risk flags
    flag_reasons = {f: list(r) for f, r in policy.flag_reasons.items()}
    draft_flags = [f for f in draft.get("risk_flags", []) if f in pe.RISK_FLAGS]
    if draft.get("prompt_injection_observed") and "prompt_injection_detected" not in draft_flags:
        draft_flags.append("prompt_injection_detected")
    if has_draft:
        for f in policy.risk_flags:
            if f not in draft_flags:
                adjust("risk_flags", "added", f"{f} - {policy.flag_reasons[f][0]}")
    for f in draft_flags:
        if f in flag_reasons:
            continue
        if f in FACT_FLAGS:
            adjust("risk_flags", "removed", f"{f} - asserted by the model but not supported by any tool result")
            continue
        if f == "missing_information" and not draft.get("insufficient_fields"):
            adjust("risk_flags", "removed", "missing_information - no missing field identified")
            continue
        flag_reasons[f] = ["Raised by AI review"]
        if f in FLAG_TO_APPROVAL and FLAG_TO_APPROVAL[f] not in approval_reasons:
            approval_reasons[FLAG_TO_APPROVAL[f]] = [f"AI review raised {f}"]

    # ---- missing information
    missing = dict(policy.missing_fields)
    for key in draft.get("insufficient_fields", []):
        if key in pe.REQUIRED_FIELDS and key not in missing:
            missing[key] = pe.REQUIRED_FIELDS[key] + " (judged insufficient by AI review)"
    if has_draft:
        for key in policy.missing_fields:
            if key not in draft.get("insufficient_fields", []):
                adjust("missing_information", "added", SHORT_FIELD.get(key, key))
    if missing and "missing_information" not in flag_reasons:
        flag_reasons["missing_information"] = ["; ".join(missing.values())]

    approvals = [a for a in pe.APPROVAL_ORDER if a in approval_reasons]
    flags = [f for f in pe.RISK_FLAGS if f in flag_reasons]

    # ---- action
    floor = "request_clarification" if missing else policy.action_floor
    overlap_names = [m["product_name"] for m in facts["catalog"]["overlapping"]]
    action = floor
    draft_action = draft.get("action")
    if draft_action in pe.ACTIONS:
        if draft_action == "reuse_existing_tool":
            # Reuse is only a valid outcome when overlap exists and nothing else blocks routing.
            if "existing_tool_overlap" in flags and floor == "route_for_approval":
                action = draft_action
            else:
                adjust("action", "raised", f"reuse_existing_tool -> {floor}: required reviews or missing information come first")
        elif pe.ACTION_SEVERITY[draft_action] >= pe.ACTION_SEVERITY[floor]:
            action = draft_action
        else:
            adjust("action", "raised", f"{draft_action} -> {floor}: rules require a more cautious action")

    # ---- texts
    recommendation = draft.get("recommendation", "").strip() if action == draft_action else ""
    next_step = draft.get("next_step", "").strip() if action == draft_action else ""
    if recommendation and APPROVAL_CLAIM.search(recommendation):
        adjust("recommendation", "replaced", "draft wording implied an approval decision")
        recommendation = ""
    recommendation = recommendation or template_recommendation(action, approvals, flags, missing, overlap_names)
    next_step = next_step or template_next_step(action, approvals, facts, overlap_names)

    questions = [q for q in draft.get("clarification_questions", []) if isinstance(q, str) and q.strip()]
    if action == "request_clarification" and not questions:
        questions = [QUESTION_FOR_FIELD[k] for k in missing if k in QUESTION_FOR_FIELD]

    # ---- grounding
    known_ids = set(ctx.evidence_ids)
    cited = [c for c in draft.get("cited_evidence_ids", []) if isinstance(c, str)]
    invalid_citations = [c for c in cited if c not in known_ids]
    valid_citations = [c for c in dict.fromkeys(cited) if c in known_ids]

    # ---- approver names
    names: dict[str, str] = {}
    r = facts.get("requester") or {}
    if r.get("manager") and "Manager" in approvals:
        names["Manager"] = f"{r['manager']['name']} ({r['manager']['employee_id']})"
    if r.get("department_head") and "Department Head" in approvals:
        names["Department Head"] = f"{r['department_head']['name']} ({r['department_head']['employee_id']})"

    tel = ctx.telemetry
    diagnostics = {
        "rule_floor": policy.action_floor,
        "draft_action": draft_action,
        "draft_approvals": draft_approvals,
        "draft_flags": draft_flags,
        "guard_corrections": sum(1 for a in adjustments if a.change in {"added", "raised", "removed", "replaced"}),
        "invalid_citations": invalid_citations,
        "cited_count": len(valid_citations),
        "external_api_calls": tel.external_api_calls,
        "input_tokens": tel.input_tokens,
        "output_tokens": tel.output_tokens,
        "llm_ms": round(tel.llm_ms),
        "trace": tel.trace,
        "reference_date": facts["reference_date"],
        **(extra_diagnostics or {}),
    }

    if has_draft:
        rationale = draft.get("rationale", "")
        need = draft.get("need_summary", "")
        overlap = f"{draft.get('overlap_assessment', '')}: {draft.get('overlap_explanation', '')}".strip(": ")
    else:
        rationale = "AI assessment unavailable - this recommendation was produced by the deterministic policy rules only. " + (
            "Overlap with existing tools was detected but whether the request shows a credible gap needs human judgement."
            if overlap_names else "")
        need = ""
        overlap = (f"Overlaps with {', '.join(overlap_names)} - gap not assessed (AI unavailable)" if overlap_names else "No overlap found in catalog")

    return CopilotDecision(
        request_id=str(ctx.request.get("request_id") or "AD-HOC"),
        recommendation=recommendation,
        evidence=list(ctx.evidence),
        required_approvals=approvals,
        missing_information=list(missing.values()),
        risk_flags=flags + ([] if ai_status == "ok" else ["ai_assessment_unavailable"]),
        next_step=next_step,
        human_review_required=True,
        telemetry=RunTelemetry(llm_calls=tel.llm_calls, tool_calls=tel.tool_calls, tool_names=list(tel.tool_names)),
        architecture=architecture,
        action=action,
        rationale=rationale.strip(),
        need_summary=need.strip(),
        overlap_assessment=overlap,
        clarification_questions=questions,
        approval_reasons={a: approval_reasons[a] for a in approvals},
        flag_reasons={f: flag_reasons[f] for f in flags},
        approver_names=names,
        cited_evidence=valid_citations,
        guard_adjustments=adjustments,
        ai_status=ai_status,  # type: ignore[arg-type]
        ai_error=ai_error,
        model=model,
        diagnostics=diagnostics,
    )
