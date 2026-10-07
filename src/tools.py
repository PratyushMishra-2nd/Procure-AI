"""Agent-visible tools.

All five tools are deterministic code over the business data and the mock vendor
service. The model chooses *which* to call and interprets the results; it never
computes a threshold, a date difference or a budget comparison itself.

Every tool returns JSON with an ``evidence`` list. Each evidence item gets a
stable ID (``E1``, ``E2``...) that the model must cite, which is how the eval
measures grounding: a citation to an ID no tool produced is a grounding failure.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from src import policy_engine as pe
from src import vendor_client
from src.contracts import EvidenceItem
from src.telemetry import RunTelemetryCounter


def _schema(properties: dict[str, Any] | None = None) -> dict[str, Any]:
    properties = properties or {}
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


NULLABLE_STRING = {"anyOf": [{"type": "string"}, {"type": "null"}]}

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "lookup_requester",
        "description": "Look up the requester in the employee directory: department, level, manager, and the resolved Manager / Department Head approvers. Takes no arguments; it always uses the requester on the current request.",
        "input_schema": _schema(),
    },
    {
        "name": "check_budget",
        "description": "Deterministic budget check: compares the request's annual cost with the requester department's AVAILABLE software budget. Returns within_budget / insufficient / cost_unknown / no_budget_record. Takes no arguments.",
        "input_schema": _schema(),
    },
    {
        "name": "search_software_catalog",
        "description": "Search the approved software catalog and purchase history for overlap: same product, same category, same vendor, or a keyword/use-case match. With both arguments null it searches by the request's own product, vendor and category. Pass `query` with use-case words (e.g. 'task tracker', 'design templates') to look for an existing tool that could satisfy the need under a different category.",
        "input_schema": _schema({"query": NULLABLE_STRING, "category": NULLABLE_STRING}),
    },
    {
        "name": "get_vendor_risk",
        "description": "Get the vendor's internal registry record AND the external vendor-risk service record, with the security-review freshness computed against the policy reference date and any registry/service conflicts. If the external service is unavailable the result says so; never infer a favourable status. `vendor_name` null means the request's vendor.",
        "input_schema": _schema({"vendor_name": NULLABLE_STRING}),
    },
    {
        "name": "evaluate_policy_rules",
        "description": "Run the deterministic policy engine (policy sections 1-10) on the current request: financial approval tier, budget, overlap, security/privacy/legal triggers, vendor-review freshness, missing required information and prompt-injection scan. Returns the MINIMUM required approvals, risk flags, missing information and the least-cautious permissible action. Its output is authoritative: you may add caution but cannot remove anything it requires.",
        "input_schema": _schema(),
    },
]
for _spec in TOOL_SPECS:
    _spec["strict"] = True

TOOL_NAMES = [s["name"] for s in TOOL_SPECS]


@dataclass
class ToolContext:
    """State for one analysis run: the request, evidence registry, caches and telemetry."""

    request: dict[str, Any]
    telemetry: RunTelemetryCounter = field(default_factory=RunTelemetryCounter)
    evidence: list[EvidenceItem] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    _vendor_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    _policy: pe.PolicyResult | None = None

    # ------------------------------------------------------------------ evidence

    def add_evidence(self, finding: pe.Finding) -> str:
        for eid, item in zip(self.evidence_ids, self.evidence):
            if item.source == finding.source and item.finding == finding.finding:
                return eid
        eid = f"E{len(self.evidence) + 1}"
        self.evidence.append(EvidenceItem(source=finding.source, finding=finding.finding, reference=finding.reference))
        self.evidence_ids.append(eid)
        return eid

    def _pack(self, findings: list[pe.Finding], data: dict[str, Any]) -> dict[str, Any]:
        items = []
        for f in findings:
            eid = self.add_evidence(f)
            items.append({"id": eid, "source": f.source, "finding": f.finding, "reference": f.reference})
        return {"evidence": items, "data": data}

    def evidence_by_id(self) -> dict[str, EvidenceItem]:
        return dict(zip(self.evidence_ids, self.evidence))

    # ------------------------------------------------------------------ shared lookups

    def vendor_api(self, vendor_name: str | None) -> dict[str, Any]:
        """Call the external service once per vendor per run."""
        key = (vendor_name or "").strip()
        if key in self._vendor_cache:
            return self._vendor_cache[key]
        if not key:
            result = {"status": "not_found", "error": "No vendor named on the request"}
        else:
            try:
                result = {"status": "ok", "record": vendor_client.get_vendor_risk(key)}
            except vendor_client.VendorRiskError as exc:
                result = {"status": exc.status, "error": str(exc)}
            self.telemetry.record_external_call(f"GET /vendor-risk/{key}", result["status"] == "ok", result.get("error", ""))
        self._vendor_cache[key] = result
        return result

    def policy(self, caller: str) -> pe.PolicyResult:
        """Deterministic policy result, computed once; a computation counts as a tool call."""
        if self._policy is None:
            self._policy = pe.evaluate(self.request, self.vendor_api(self.request.get("vendor_name")))
            if caller != "agent":
                self.telemetry.record_tool_call("evaluate_policy_rules", caller=caller)
        return self._policy

    # ------------------------------------------------------------------ tools

    def lookup_requester(self) -> dict[str, Any]:
        r = pe.assess_requester(self.request.get("requester_id"))
        return self._pack(pe.describe_requester(r), r)

    def check_budget(self) -> dict[str, Any]:
        r = pe.assess_requester(self.request.get("requester_id"))
        dept = r["requester"]["department"] if r.get("found") else None
        b = pe.assess_budget(dept, pe.parse_amount(self.request.get("annual_cost_usd")))
        return self._pack(pe.describe_budget(b), b)

    def search_software_catalog(self, query: str | None = None, category: str | None = None) -> dict[str, Any]:
        c = pe.match_catalog(self.request, query=query, category=category)
        slim = {
            "matches": [{k: m[k] for k in ("software_id", "product_name", "vendor_name", "category", "status", "scope", "licensed_seats", "annual_cost_usd", "match_types")} for m in c["matches"]],
            "overlapping_product_ids": [m["software_id"] for m in c["overlapping"]],
            "seat_utilisation_data_available": False,
        }
        return self._pack(pe.describe_catalog(c, self.request), slim)

    def get_vendor_risk(self, vendor_name: str | None = None) -> dict[str, Any]:
        name = vendor_name or self.request.get("vendor_name")
        v = pe.assess_vendor(name, self.vendor_api(name))
        slim = {k: v[k] for k in ("vendor_name", "in_registry", "api_status", "security_assessment", "expired",
                                   "conflicts", "is_new_vendor", "legal_terms_status", "processes_personal_data",
                                   "stores_data_outside_region", "risk_level", "reference_date")}
        return self._pack(pe.describe_vendor(v), slim)

    def evaluate_policy_rules(self) -> dict[str, Any]:
        p = self.policy(caller="agent")
        rule_findings = [
            pe.Finding("policy_engine", f"Requires {a}: {'; '.join(reasons)}", "procurement_policy.md")
            for a, reasons in p.approval_reasons.items()
        ] + [
            pe.Finding("policy_engine", f"Missing required information: {label}", "Policy section 1")
            for label in p.missing_fields.values()
        ] + [f for f in p.findings if f.source in {"policy_engine", "injection_scanner"}]
        data = {
            "required_approvals": p.required_approvals,
            "risk_flags": p.risk_flags,
            "flag_reasons": p.flag_reasons,
            "missing_information": list(p.missing_fields.values()),
            "missing_field_keys": list(p.missing_fields),
            "least_cautious_permissible_action": p.action_floor,
            "financial_tier": p.financial_tier,
        }
        return self._pack(rule_findings, data)

    # ------------------------------------------------------------------ dispatch

    def dispatch(self) -> dict[str, Callable[..., dict[str, Any]]]:
        return {
            "lookup_requester": self.lookup_requester,
            "check_budget": self.check_budget,
            "search_software_catalog": self.search_software_catalog,
            "get_vendor_risk": self.get_vendor_risk,
            "evaluate_policy_rules": self.evaluate_policy_rules,
        }

    def run_tool(self, name: str, args: dict[str, Any] | None, caller: str = "agent") -> tuple[str, bool]:
        """Execute a tool by name; returns (JSON text, is_error). Never raises for bad input."""
        fn = self.dispatch().get(name)
        args = args if isinstance(args, dict) else {}
        if fn is None:
            self.telemetry.record_tool_call(name, caller=caller, args=args, ok=False)
            return json.dumps({"error": f"Unknown tool '{name}'. Available: {TOOL_NAMES}"}), True
        allowed = set(next(s for s in TOOL_SPECS if s["name"] == name)["input_schema"]["properties"])
        clean = {k: v for k, v in args.items() if k in allowed and (v is None or isinstance(v, str))}
        try:
            result = fn(**clean)
        except Exception as exc:  # a tool failure is reported to the model, not raised
            self.telemetry.record_tool_call(name, caller=caller, args=clean, ok=False)
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"}), True
        self.telemetry.record_tool_call(name, caller=caller, args=clean, ok=True)
        return json.dumps(result, default=str), False

    def gather_all(self, caller: str) -> dict[str, Any]:
        """Run every tool once (used by deterministic fallback and evidence back-fill)."""
        return {name: json.loads(self.run_tool(name, {}, caller=caller)[0]) for name in TOOL_NAMES}
