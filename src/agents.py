"""Architecture A (single agent) and Architecture B (analyst -> reviewer).

Both share the same tools, the same deterministic engine and the same policy
guard, so the comparison isolates the one thing that differs: how the model work
is split.

A  UI -> Procurement Agent (tools loop, decides) -> policy guard -> decision
B  UI -> Procurement Analyst (tools loop, evidence pack)
        -> Policy / Risk Reviewer (no tools, decides) -> policy guard -> decision
"""
from __future__ import annotations

from typing import Any

from src import config, prompts
from src.guard import finalize
from src.contracts import CopilotDecision
from src.llm import LLMClient, LLMUnavailable
from src.tools import TOOL_SPECS, ToolContext


def tool_loop(llm: LLMClient, agent: str, system: str, user_text: str, ctx: ToolContext,
              schema: dict[str, Any]) -> dict[str, Any]:
    """Run one agent until it returns schema-valid JSON instead of tool calls.

    History is append-only: assistant turns are appended exactly as returned
    (thinking blocks included) and tool results follow in one user message.
    """
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_text}]
    for _ in range(config.max_agent_turns()):
        resp = llm.create(agent=agent, system=system, messages=messages, tools=TOOL_SPECS,
                          output_schema=schema, telemetry=ctx.telemetry)
        messages.append({"role": "assistant", "content": resp.content})
        if resp.tool_uses:
            results = []
            for use in resp.tool_uses:
                text, is_error = ctx.run_tool(use.name, use.input, caller=agent)
                block = {"type": "tool_result", "tool_use_id": use.id, "content": text}
                if is_error:
                    block["is_error"] = True
                results.append(block)
            messages.append({"role": "user", "content": results})
            continue
        if resp.stop_reason == "pause_turn":
            continue
        return resp.json()
    raise LLMUnavailable(f"{agent} did not produce a final answer within {config.max_agent_turns()} turns")


def run_single(request: dict[str, Any], llm: LLMClient | None, disabled: bool = False) -> CopilotDecision:
    ctx = ToolContext(request=request)
    if llm is None:
        return _deterministic(ctx, "single", disabled=disabled)
    try:
        draft = tool_loop(llm, "procurement_agent", prompts.single_agent_system(), prompts.request_message(request),
                          ctx, prompts.DECISION_SCHEMA)
    except LLMUnavailable as exc:
        return _deterministic(ctx, "single", error=str(exc), model=llm.model)
    return finalize(ctx, draft, architecture="single", ai_status="ok", ai_error=None, model=llm.model)


def run_staged(request: dict[str, Any], llm: LLMClient | None, disabled: bool = False) -> CopilotDecision:
    ctx = ToolContext(request=request)
    if llm is None:
        return _deterministic(ctx, "staged", disabled=disabled)
    try:
        pack = tool_loop(llm, "procurement_analyst", prompts.analyst_system(), prompts.request_message(request),
                         ctx, prompts.EVIDENCE_PACK_SCHEMA)
        policy = ctx.policy(caller="orchestrator")
        for finding in policy.findings:  # reviewer sees every deterministic fact, citable by ID
            ctx.add_evidence(finding)
        evidence = [{"id": eid, **item.model_dump()} for eid, item in zip(ctx.evidence_ids, ctx.evidence)]
        policy_view = {
            "required_approvals": policy.required_approvals,
            "approval_reasons": policy.approval_reasons,
            "risk_flags": policy.risk_flags,
            "flag_reasons": policy.flag_reasons,
            "missing_information": list(policy.missing_fields.values()),
            "missing_field_keys": list(policy.missing_fields),
            "least_cautious_permissible_action": policy.action_floor,
            "financial_tier": policy.financial_tier,
        }
        resp = llm.create(
            agent="policy_reviewer",
            system=prompts.reviewer_system(),
            messages=[{"role": "user", "content": prompts.reviewer_message(request, pack, evidence, policy_view)}],
            tools=None,
            output_schema=prompts.DECISION_SCHEMA,
            telemetry=ctx.telemetry,
        )
        draft = resp.json()
    except LLMUnavailable as exc:
        return _deterministic(ctx, "staged", error=str(exc), model=llm.model)
    return finalize(ctx, draft, architecture="staged", ai_status="ok", ai_error=None, model=llm.model,
                    extra_diagnostics={"analyst_pack": pack})


def _deterministic(ctx: ToolContext, architecture: str, error: str | None = None, model: str | None = None,
                   disabled: bool = False) -> CopilotDecision:
    """Degraded mode: rules only, clearly labelled, never more permissive than the rules.

    ``error`` None means no model was configured for this run (AI switched off or no credentials).
    """
    ctx.gather_all(caller="fallback")
    if error:
        status, why = "unavailable", error
    elif disabled or config.llm_disabled():
        status, why = "disabled", "AI disabled - rules-only run"
    else:
        status, why = "unavailable", "No model provider configured (set GEMINI_API_KEY, CLOSEROUTER_API_KEY or ANTHROPIC_API_KEY in .env)"
    return finalize(ctx, None, architecture=architecture, ai_status=status, ai_error=why, model=model)
