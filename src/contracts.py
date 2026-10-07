from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    source: str = Field(description="Tool/data source name")
    finding: str = Field(description="Concise factual finding")
    reference: str | None = Field(default=None, description="Optional record ID / policy section / endpoint")


class RunTelemetry(BaseModel):
    llm_calls: int | None = None
    tool_calls: int | None = None
    tool_names: list[str] = Field(default_factory=list)


class ProcurementDecision(BaseModel):
    request_id: str
    recommendation: str = Field(description="Short recommendation label or sentence")
    evidence: list[EvidenceItem] = Field(default_factory=list)
    required_approvals: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    next_step: str
    human_review_required: bool = True
    telemetry: RunTelemetry | None = None


Architecture = Literal["single", "staged"]

Action = Literal["request_clarification", "reuse_existing_tool", "route_for_review", "route_for_approval"]


class GuardAdjustment(BaseModel):
    """One place where the deterministic policy guard overrode or extended the model's draft."""

    field: str
    change: str
    detail: str


class CopilotDecision(ProcurementDecision):
    """`ProcurementDecision` plus the context the product UI and the evaluation need.

    It is a subclass, so the eval harness's `isinstance(raw, ProcurementDecision)`
    check and `model_validate` both keep working.
    """

    architecture: str = "single"
    action: Action = "route_for_review"
    rationale: str = ""
    need_summary: str = ""
    overlap_assessment: str = ""
    clarification_questions: list[str] = Field(default_factory=list)
    approval_reasons: dict[str, list[str]] = Field(default_factory=dict)
    flag_reasons: dict[str, list[str]] = Field(default_factory=dict)
    approver_names: dict[str, str] = Field(default_factory=dict)
    cited_evidence: list[str] = Field(default_factory=list)
    guard_adjustments: list[GuardAdjustment] = Field(default_factory=list)
    ai_status: Literal["ok", "unavailable", "disabled"] = "ok"
    ai_error: str | None = None
    model: str | None = None
    latency_ms: float | None = None
    diagnostics: dict = Field(default_factory=dict)

# Suggested approval names for consistency in evaluation:
# Manager, Department Head, Procurement, Finance, CFO, Security, Privacy, Legal
#
# Suggested risk-flag taxonomy (you may add others):
# existing_tool_overlap
# budget_insufficient
# security_review_required
# privacy_review_required
# legal_review_required
# vendor_review_expired
# conflicting_vendor_evidence
# vendor_risk_unavailable
# prompt_injection_detected
# missing_information
