"""Deterministic procurement rules (policy sections 1-10).

Nothing in this module calls a model. Every function is a pure function of the
request and the business data, so its output is reproducible and unit-testable.
The agents call these through ``src/tools.py``; the policy guard
(``src/guard.py``) re-applies them after the model has spoken, so a model can add
caution but can never remove an approval, a risk flag or a missing-information
item that the rules require.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Any, Iterable

from src import data_access as da
from src.config import reference_date

REVIEW_VALIDITY_DAYS = 365

APPROVAL_ORDER = [
    "Manager",
    "Department Head",
    "Procurement",
    "Finance",
    "CFO",
    "Security",
    "Privacy",
    "Legal",
]

RISK_FLAGS = [
    "existing_tool_overlap",
    "budget_insufficient",
    "budget_unverified",
    "security_review_required",
    "privacy_review_required",
    "legal_review_required",
    "vendor_review_expired",
    "conflicting_vendor_evidence",
    "vendor_risk_unavailable",
    "vendor_not_onboarded",
    "prompt_injection_detected",
    "missing_information",
    "urgency_pressure",
]

ACTIONS = ["request_clarification", "reuse_existing_tool", "route_for_review", "route_for_approval"]
ACTION_SEVERITY = {
    "route_for_approval": 0,
    "reuse_existing_tool": 1,
    "route_for_review": 2,
    "request_clarification": 3,
}

# Policy section 1 fields. Keys are what the model may cite in `insufficient_fields`.
REQUIRED_FIELDS = {
    "requester": "Requester and department (requester ID not found in employee records)",
    "product_vendor": "Product / vendor name",
    "annual_cost": "Annual cost or a reasonable annual cost estimate (price)",
    "user_count": "Number of users / licenses (seats)",
    "business_purpose": "Business purpose (a concrete business need)",
    "data_access_level": "Intended data-access level (which data classes the tool will touch)",
    "integrations": "Required integrations (list them, or state that none are needed)",
}

# --------------------------------------------------------------------------- data classes

KNOWN_DATA_LEVELS: dict[str, set[str]] = {
    "none": set(),
    "public": set(),
    "internal": set(),
    "internal_documents": set(),
    "internal_marketing": set(),
    "internal_finance": set(),
    "production_telemetry": {"production"},
    "source_code": {"source_code"},
    "confidential_documents": {"confidential"},
    "employee_pii": {"employee_pii"},
    "customer_pii": {"customer_pii"},
    "credentials": {"credentials"},
    "secrets": {"credentials"},
}
MISSING_DATA_LEVELS = {"", "unknown", "tbd", "n/a", "na", "not sure", "unspecified"}

SECURITY_CLASSES = {"source_code", "production", "confidential", "employee_pii", "customer_pii", "credentials", "unclassified"}
PII_CLASSES = {"employee_pii", "customer_pii"}
SENSITIVE_CLASSES = PII_CLASSES | {"confidential", "source_code", "credentials"}

CLASS_LABELS = {
    "source_code": "source code access",
    "production": "production / cloud-account access",
    "confidential": "confidential documents",
    "employee_pii": "employee PII",
    "customer_pii": "customer PII",
    "credentials": "credentials / secrets",
    "unclassified": "an unrecognised data class",
}

INTEGRATION_RULES: list[tuple[str, str]] = [
    (r"\bprod(uction)?\b|cloud account|\baws\b|\bazure\b|\bgcp\b|kubernetes|\bk8s\b", "production"),
    (r"\bgit\b|github|gitlab|bitbucket|source code|code repositor", "source_code"),
    (r"document repositor|sharepoint|google drive|contract repositor|\bdms\b", "confidential"),
    (r"\bcrm\b|salesforce|hubspot|help ?desk|zendesk|ticket", "customer_pii"),
    (r"\bhris\b|workday|payroll|bamboohr", "employee_pii"),
    (r"vault|secret|credential|password manager", "credentials"),
]

# --------------------------------------------------------------------------- prompt injection

INJECTION_PATTERNS: list[tuple[str, str]] = [
    (r"\b(ignore|disregard|forget|override|bypass)\b[^.]{0,40}\b(rules?|polic(y|ies)|instructions?|controls?|guardrails?|checks?|reviews?)\b", "instructs the copilot to ignore or bypass rules"),
    (r"\btreat (this|the|it)\b[^.]{0,40}\bas\b[^.]{0,30}\bapproved\b", "asserts an approval that is not on record"),
    (r"\b(pre-?approved|already approved|has been approved|is approved)\b[^.]{0,40}\b(by|from)\b", "claims a prior approval"),
    (r"\b(cfo|ceo|vp|security|legal|finance)[- ]approved\b", "claims a prior approval"),
    (r"\bapprove (it|this|the request|immediately|now)\b", "instructs the copilot to approve"),
    (r"\b(skip|waive|no need for|do not (run|perform|require|flag|escalate))\b[^.]{0,30}\b(security|privacy|legal|review|approval|checks?)\b", "asks to skip a required control"),
    (r"\b(system prompt|you are now|new instructions|as an ai|assistant:|developer mode)\b", "attempts to re-prompt the model"),
    (r"\b(reveal|print|show|expose)\b[^.]{0,30}\b(api key|secret|password|token|system prompt)\b", "attempts to extract secrets"),
]

URGENCY_VALUES = {"urgent", "asap", "critical", "immediate"}

# --------------------------------------------------------------------------- results


@dataclass
class Finding:
    """A fact produced by a deterministic check, with where it came from."""

    source: str
    finding: str
    reference: str | None = None


@dataclass
class Requirement:
    name: str
    reasons: list[str] = field(default_factory=list)


@dataclass
class PolicyResult:
    required_approvals: list[str]
    approval_reasons: dict[str, list[str]]
    risk_flags: list[str]
    flag_reasons: dict[str, list[str]]
    missing_fields: dict[str, str]
    action_floor: str
    financial_tier: str
    findings: list[Finding]
    facts: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- helpers


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _tokens(text: Any) -> set[str]:
    stop = {"the", "a", "an", "and", "for", "of", "to", "pro", "plus", "add-on", "addon", "team", "teams",
            "enterprise", "business", "expansion", "advanced", "workspace", "suite", "pack", "tool"}
    return {t for t in re.findall(r"[a-z0-9]+", _norm(text)) if len(t) > 2 and t not in stop}


def _money(value: Any) -> str:
    if value is None:
        return "unknown"
    return f"${float(value):,.2f}".replace(".00", "")


def parse_amount(value: Any) -> float | None:
    """Return a non-negative annual amount or None when absent / not a number."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if value >= 0 else None
    match = re.fullmatch(r"\s*\$?\s*([0-9][0-9,]*(\.[0-9]+)?)\s*", str(value))
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


def parse_count(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        count = int(float(value))
    except (TypeError, ValueError):
        return None
    return count if count > 0 else None


# --------------------------------------------------------------------------- individual checks


def financial_tier(amount: float | None) -> tuple[str, list[str]]:
    """Policy section 4 - minimum business approvals by annual amount."""
    if amount is None:
        return "unknown", []
    if amount <= 1_000:
        return "up_to_1k", ["Manager"]
    if amount <= 10_000:
        return "1k_to_10k", ["Department Head", "Procurement"]
    if amount <= 25_000:
        return "10k_to_25k", ["Department Head", "Finance", "Procurement"]
    return "above_25k", ["Department Head", "Finance", "CFO", "Procurement"]


def assess_requester(requester_id: str | None) -> dict[str, Any]:
    employee = da.find_employee(requester_id)
    if employee is None:
        return {"found": False, "requester_id": requester_id}
    employees = {e["employee_id"]: e for e in da.load_employees()}
    manager = employees.get(employee["manager_id"]) if employee.get("manager_id") else None

    # Department Head: a Director/VP in the requester's own department; otherwise the
    # first Director/VP up the reporting line (e.g. Sales/CS report into Go To Market).
    head = next(
        (e for e in employees.values()
         if e["department"] == employee["department"] and e["level"] in {"Director", "VP"}
         and e["employee_id"] != employee["employee_id"]),
        None,
    )
    head_basis = "same department"
    if head is None:
        cursor, seen = manager, set()
        while cursor and cursor["employee_id"] not in seen:
            seen.add(cursor["employee_id"])
            if cursor["level"] in {"Director", "VP"}:
                head, head_basis = cursor, "reporting line (no Director in requester's department)"
                break
            cursor = employees.get(cursor["manager_id"]) if cursor.get("manager_id") else None

    def brief(e: dict | None) -> dict | None:
        return None if e is None else {k: e[k] for k in ("employee_id", "name", "department", "level")}

    return {
        "found": True,
        "requester": brief(employee),
        "country": employee.get("country"),
        "manager": brief(manager),
        "department_head": brief(head),
        "department_head_basis": head_basis if head else None,
    }


def assess_budget(department: str | None, amount: float | None) -> dict[str, Any]:
    """Policy section 2."""
    budget = da.find_budget(department)
    if budget is None:
        return {"status": "no_budget_record", "department": department, "requested": amount}
    available = float(budget["available_usd"])
    base = {
        "department": budget["department"],
        "annual_budget": budget["annual_software_budget_usd"],
        "committed": budget["committed_usd"],
        "available": available,
        "requested": amount,
    }
    if amount is None:
        return {**base, "status": "cost_unknown"}
    if amount > available:
        return {**base, "status": "insufficient", "shortfall": round(amount - available, 2)}
    return {**base, "status": "within_budget", "remaining_after": round(available - amount, 2)}


def match_catalog(request: dict[str, Any], query: str | None = None, category: str | None = None) -> dict[str, Any]:
    """Policy section 3 - same product, same category, same vendor, or keyword match."""
    product = request.get("product_name")
    vendor = request.get("vendor_name")
    req_category = category or request.get("category")
    product_tokens = _tokens(product)
    query_tokens = _tokens(query) if query else set()

    matches = []
    for item in da.load_software_catalog():
        reasons = []
        same_category = bool(req_category) and _norm(item["category"]) == _norm(req_category)
        # A vendor's training pack is not "the same product" as its e-signature tool, so a
        # name-token match only counts when vendor and category also line up.
        if _norm(item["product_name"]) == _norm(product) or (
            product_tokens & _tokens(item["product_name"])
            and _norm(item["vendor_name"]) == _norm(vendor)
            and same_category
        ):
            reasons.append("same_product")
        if same_category:
            reasons.append("same_category")
        if vendor and _norm(item["vendor_name"]) == _norm(vendor):
            reasons.append("same_vendor")
        if query_tokens:
            haystack = _tokens(" ".join(str(item[k] or "") for k in ("product_name", "category", "notes", "scope")))
            if query_tokens & haystack:
                reasons.append("keyword")
        if reasons:
            matches.append({**item, "match_types": reasons})

    history = [
        p for p in da.load_purchase_history()
        if _norm(p["vendor_name"]) == _norm(vendor)
        or any(_norm(p["product_name"]) == _norm(m["product_name"]) for m in matches)
    ]
    overlap = [m for m in matches if {"same_product", "same_category"} & set(m["match_types"])]
    return {"matches": matches, "overlapping": overlap, "purchase_history": history}


def assess_vendor(vendor_name: str | None, api_result: dict[str, Any]) -> dict[str, Any]:
    """Combine the internal registry with the external risk service (policy sections 5, 7, 10).

    ``api_result`` is ``{"status": "ok", "record": {...}}`` or
    ``{"status": "unavailable" | "not_found", "error": "..."}`` from the vendor tool.
    """
    ref = reference_date()
    registry = da.find_vendor(vendor_name)
    api_ok = api_result.get("status") == "ok"
    api = api_result.get("record") if api_ok else None

    def review_state(status: str | None, review_date: str | None) -> tuple[str, int | None]:
        status = _norm(status)
        age = None
        if review_date:
            try:
                age = (ref - date.fromisoformat(review_date)).days
            except ValueError:
                return "invalid_date", None
        if status in {"approved", "current"}:
            if age is None:
                return "missing_date", None
            return ("current" if age <= REVIEW_VALIDITY_DAYS else "expired"), age
        if status == "expired":
            return "expired", age
        if status in {"pending", "not_completed", "in_progress"}:
            return "not_completed", age
        return "unknown", age

    reg_state, reg_age = review_state(
        registry.get("security_status") if registry else None,
        registry.get("security_review_date") if registry else None,
    )
    api_state, api_age = (None, None)
    if api is not None:
        api_state, api_age = review_state(api.get("security_review_status"), api.get("last_review_date"))

    conflicts: list[str] = []
    if registry and api is not None:
        reg_label = _norm(registry.get("security_status"))
        api_label = _norm(api.get("security_review_status"))
        if reg_label == "approved" and api_label in {"expired", "not_completed"}:
            conflicts.append(f"registry security status is 'Approved' but the vendor-risk service reports '{api_label}'")
        if reg_label in {"pending", "not_completed"} and api_label == "approved":
            conflicts.append(f"registry security status is '{registry.get('security_status')}' but the vendor-risk service reports 'approved'")
        reg_date, api_date = registry.get("security_review_date"), api.get("last_review_date")
        if reg_date and api_date and reg_date != api_date:
            conflicts.append(f"review dates differ (registry {reg_date}, vendor-risk service {api_date})")

    states = {s for s in (reg_state, api_state) if s}
    if conflicts:
        assessment = "conflicting"
    elif "expired" in states:
        assessment = "expired"
    elif api is None and reg_state == "current":
        assessment = "unverified"  # registry alone says current; external confirmation failed
    elif states == {"current"}:
        assessment = "current"
    elif "not_completed" in states:
        assessment = "not_completed"
    else:
        assessment = "missing"

    expired_by_date = any(a is not None and a > REVIEW_VALIDITY_DAYS for a in (reg_age, api_age))
    is_new = registry is None or _norm(registry.get("procurement_status")) != "approved"
    return {
        "vendor_name": vendor_name,
        "in_registry": registry is not None,
        "registry": registry,
        "api_status": api_result.get("status"),
        "api_error": api_result.get("error"),
        "api": api,
        "registry_review_state": reg_state,
        "registry_review_age_days": reg_age,
        "api_review_state": api_state,
        "api_review_age_days": api_age,
        "security_assessment": assessment,
        "expired": assessment == "expired" or expired_by_date or api_state == "expired",
        "conflicts": conflicts,
        "is_new_vendor": is_new,
        "legal_terms_approved": bool(registry) and _norm(registry.get("legal_terms_status")) == "approved",
        "legal_terms_status": registry.get("legal_terms_status") if registry else None,
        "processes_personal_data": api.get("processes_personal_data") if api else None,
        "stores_data_outside_region": api.get("stores_data_outside_region") if api else None,
        "risk_level": api.get("risk_level") if api else None,
        "reference_date": ref.isoformat(),
        "review_validity_days": REVIEW_VALIDITY_DAYS,
    }


def classify_data(request: dict[str, Any]) -> dict[str, Any]:
    """Map the declared data-access level and integrations to policy data classes."""
    raw = request.get("data_access_level")
    level = _norm(raw).replace(" ", "_")
    classes: dict[str, str] = {}
    missing = level in MISSING_DATA_LEVELS
    if not missing:
        if level in KNOWN_DATA_LEVELS:
            for c in KNOWN_DATA_LEVELS[level]:
                classes[c] = f"declared data-access level '{raw}'"
        else:
            # Unrecognised label: classify by keyword, otherwise escalate rather than assume safe.
            guessed = {
                "pii": "customer_pii" if "customer" in level else "employee_pii",
                "personal": "customer_pii" if "customer" in level else "employee_pii",
                "source": "source_code", "code": "source_code", "confidential": "confidential",
                "restricted": "confidential", "secret": "credentials", "credential": "credentials",
                "password": "credentials", "prod": "production",
            }
            hits = {cls for key, cls in guessed.items() if key in level}
            for c in hits or {"unclassified"}:
                classes[c] = f"declared data-access level '{raw}'"

    integrations = request.get("requested_integrations")
    for integ in integrations or []:
        for pattern, cls in INTEGRATION_RULES:
            if re.search(pattern, _norm(integ)) and cls not in classes:
                classes[cls] = f"integration '{integ}'"
    return {"declared_level": raw, "level_missing": missing, "classes": classes}


def scan_for_injection(texts: Iterable[tuple[str, Any]]) -> list[dict[str, str]]:
    """Find instruction-like content inside business data (policy section 9)."""
    hits = []
    for location, text in texts:
        if not text:
            continue
        for pattern, why in INJECTION_PATTERNS:
            m = re.search(pattern, str(text), flags=re.IGNORECASE)
            if m:
                hits.append({"location": location, "excerpt": m.group(0)[:120], "why": why})
    return hits


def _strip_injection(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text or "")
    keep = [s for s in sentences if not scan_for_injection([("s", s)])]
    return " ".join(keep).strip()


def check_completeness(request: dict[str, Any], requester: dict[str, Any], data: dict[str, Any]) -> dict[str, str]:
    """Policy section 1. Returns {field_key: human-readable label}."""
    missing: dict[str, str] = {}
    if not requester.get("found"):
        missing["requester"] = REQUIRED_FIELDS["requester"]
    if not (request.get("product_name") and request.get("vendor_name")):
        missing["product_vendor"] = REQUIRED_FIELDS["product_vendor"]
    if parse_amount(request.get("annual_cost_usd")) is None:
        missing["annual_cost"] = REQUIRED_FIELDS["annual_cost"]
    if parse_count(request.get("user_count")) is None:
        missing["user_count"] = REQUIRED_FIELDS["user_count"]
    purpose = _strip_injection(str(request.get("business_justification") or ""))
    if len(re.findall(r"[A-Za-z]{2,}", purpose)) < 5:
        missing["business_purpose"] = REQUIRED_FIELDS["business_purpose"] + " - the justification is missing or too vague once instruction-like text is removed"
    if data["level_missing"]:
        missing["data_access_level"] = REQUIRED_FIELDS["data_access_level"]
    if request.get("requested_integrations") is None:
        missing["integrations"] = REQUIRED_FIELDS["integrations"]
    return missing


# --------------------------------------------------------------------------- the rule set


def evaluate(request: dict[str, Any], vendor_api_result: dict[str, Any]) -> PolicyResult:
    """Apply every deterministic rule. ``vendor_api_result`` comes from the vendor tool."""
    approvals: dict[str, list[str]] = {}
    flags: dict[str, list[str]] = {}
    findings: list[Finding] = []

    def need(approver: str, reason: str) -> None:
        approvals.setdefault(approver, [])
        if reason not in approvals[approver]:
            approvals[approver].append(reason)

    def flag(name: str, reason: str) -> None:
        flags.setdefault(name, [])
        if reason not in flags[name]:
            flags[name].append(reason)

    amount = parse_amount(request.get("annual_cost_usd"))
    requester = assess_requester(request.get("requester_id"))
    department = requester["requester"]["department"] if requester.get("found") else None
    budget = assess_budget(department, amount)
    catalog = match_catalog(request)
    vendor = assess_vendor(request.get("vendor_name"), vendor_api_result)
    data = classify_data(request)
    missing = check_completeness(request, requester, data)
    injection = scan_for_injection(
        [("request.business_justification", request.get("business_justification")),
         ("request.product_name", request.get("product_name")),
         ("vendor_registry.notes", (vendor.get("registry") or {}).get("notes")),
         ("vendor_risk_service.notes", (vendor.get("api") or {}).get("notes"))]
    )

    # ---- section 4: financial tier
    tier, tier_approvers = financial_tier(amount)
    for a in tier_approvers:
        need(a, f"Annual amount {_money(amount)} falls in tier '{tier}' (policy section 4)")
    if tier == "unknown":
        need("Procurement", "Annual cost unknown - financial approval tier cannot be determined; Procurement triages the incomplete request (policy sections 1 and 4)")

    # ---- section 2: budget
    if budget["status"] == "insufficient":
        flag("budget_insufficient", f"{_money(amount)} exceeds {budget['department']} available software budget of {_money(budget['available'])} (shortfall {_money(budget['shortfall'])})")
        need("Finance", "Budget exception review - request exceeds available department budget (policy section 2)")
    elif budget["status"] == "no_budget_record" and requester.get("found"):
        flag("budget_unverified", f"No software budget record for department '{department}'")
        need("Finance", "Department budget could not be verified (policy sections 2 and 10)")

    # ---- section 3: overlap
    if catalog["overlapping"]:
        names = ", ".join(f"{m['product_name']} ({m['scope']}, {m['licensed_seats']} seats)" for m in catalog["overlapping"])
        flag("existing_tool_overlap", f"Approved catalog already contains: {names} (policy section 3)")

    # ---- section 5: security
    for cls, origin in data["classes"].items():
        if cls in SECURITY_CLASSES:
            need("Security", f"Request involves {CLASS_LABELS[cls]} via {origin} (policy section 5)")
            flag("security_review_required", f"{CLASS_LABELS[cls]} via {origin}")
    assessment = vendor["security_assessment"]
    if assessment != "current":
        reason = {
            "conflicting": "registry and vendor-risk service disagree about the security assessment",
            "expired": f"vendor security assessment is older than {REVIEW_VALIDITY_DAYS} days at reference date {vendor['reference_date']}",
            "not_completed": "vendor security assessment is pending / not completed",
            "missing": "no vendor security assessment on record",
            "unverified": "vendor security status could not be confirmed with the vendor-risk service",
        }[assessment]
        need("Security", f"Vendor {reason} (policy sections 5 and 10)")
        flag("security_review_required", reason)
    if vendor["expired"]:
        flag("vendor_review_expired", f"Security review is past the {REVIEW_VALIDITY_DAYS}-day validity window at {vendor['reference_date']}")
    if vendor["conflicts"]:
        for c in vendor["conflicts"]:
            flag("conflicting_vendor_evidence", c)
    if vendor["api_status"] == "unavailable":
        flag("vendor_risk_unavailable", f"Vendor-risk service unavailable: {vendor['api_error']}")
    elif vendor["api_status"] == "not_found":
        flag("vendor_risk_unavailable", "Vendor-risk service has no record for this vendor")
    if vendor["is_new_vendor"]:
        status = (vendor["registry"] or {}).get("procurement_status") or "not in vendor registry"
        flag("vendor_not_onboarded", f"Vendor procurement status: {status}")
        need("Procurement", "New vendor onboarding (vendor not yet approved in the registry)")

    # ---- section 6: privacy
    pii = [c for c in data["classes"] if c in PII_CLASSES]
    for cls in pii:
        need("Privacy", f"Tool will process {CLASS_LABELS[cls]} (policy section 6)")
        flag("privacy_review_required", f"{CLASS_LABELS[cls]} via {data['classes'][cls]}")
    sensitive = [c for c in data["classes"] if c in SENSITIVE_CLASSES]
    outside = vendor["stores_data_outside_region"]
    if sensitive and outside is True:
        need("Privacy", "Vendor stores data outside the operating region and the request involves sensitive data (policy section 6)")
        flag("privacy_review_required", "sensitive data may be stored outside the operating region")
    elif sensitive and outside is None:
        need("Privacy", "Data-residency could not be verified for sensitive data - vendor-risk record unavailable (policy sections 6 and 10)")
        flag("privacy_review_required", "data-residency unverifiable for sensitive data")

    # ---- section 7: legal
    if vendor["is_new_vendor"] and amount is not None and amount >= 10_000:
        need("Legal", f"New vendor with annual spend {_money(amount)} >= $10,000 (policy section 7)")
        flag("legal_review_required", "new vendor at or above $10,000")
    if not vendor["legal_terms_approved"]:
        need("Legal", f"Vendor legal terms are '{vendor['legal_terms_status'] or 'unknown'}', not approved/standard (policy section 7)")
        flag("legal_review_required", "legal terms not approved")
    if sensitive and outside is True:
        need("Legal", "Material cross-region data-processing issue (policy section 7)")
        flag("legal_review_required", "cross-region processing of sensitive data")

    # ---- sections 1 and 9
    if missing:
        flag("missing_information", "; ".join(missing.values()))
    for hit in injection:
        flag("prompt_injection_detected", f"{hit['location']}: \"{hit['excerpt']}\" ({hit['why']}) - ignored")
    if _norm(request.get("urgency")) in URGENCY_VALUES:
        flag("urgency_pressure", f"Urgency marked '{request.get('urgency')}' - urgency does not waive any control")

    # ---- findings (evidence the guard can always show, even if an agent skipped a tool)
    findings.extend(describe_requester(requester))
    findings.extend(describe_budget(budget))
    findings.extend(describe_catalog(catalog, request))
    findings.extend(describe_vendor(vendor))
    findings.append(Finding(
        "policy_engine",
        f"Financial tier '{tier}' for annual amount {_money(amount)} -> minimum approvals: {', '.join(tier_approvers) or 'undetermined'}",
        "Policy section 4",
    ))
    if data["classes"]:
        findings.append(Finding(
            "policy_engine",
            "Data classes in scope: " + "; ".join(f"{CLASS_LABELS.get(c, c)} ({o})" for c, o in data["classes"].items()),
            "Policy sections 5-6",
        ))
    for hit in injection:
        findings.append(Finding("injection_scanner", f"Instruction-like text in {hit['location']}: \"{hit['excerpt']}\" - treated as data, not instructions", "Policy section 9"))

    ordered_approvals = [a for a in APPROVAL_ORDER if a in approvals]
    ordered_flags = [f for f in RISK_FLAGS if f in flags] + [f for f in flags if f not in RISK_FLAGS]

    if missing:
        floor = "request_clarification"
    elif any(a in approvals for a in ("Security", "Privacy", "Legal", "CFO")) or any(
        f in flags for f in ("budget_insufficient", "budget_unverified", "vendor_risk_unavailable",
                             "conflicting_vendor_evidence", "vendor_review_expired", "prompt_injection_detected")
    ) or (budget["status"] == "insufficient"):
        floor = "route_for_review"
    else:
        floor = "route_for_approval"

    return PolicyResult(
        required_approvals=ordered_approvals,
        approval_reasons={a: approvals[a] for a in ordered_approvals},
        risk_flags=ordered_flags,
        flag_reasons={f: flags[f] for f in ordered_flags},
        missing_fields=missing,
        action_floor=floor,
        financial_tier=tier,
        findings=findings,
        facts={
            "amount": amount,
            "requester": requester,
            "budget": budget,
            "catalog": catalog,
            "vendor": vendor,
            "data": data,
            "injection": injection,
            "reference_date": reference_date().isoformat(),
        },
    )


# --------------------------------------------------------------------------- evidence text


def describe_requester(r: dict[str, Any]) -> list[Finding]:
    if not r.get("found"):
        return [Finding("employee_directory", f"Requester '{r.get('requester_id')}' not found in employee records", "employees.csv")]
    req = r["requester"]
    out = [Finding("employee_directory", f"Requester {req['name']} ({req['employee_id']}), {req['level']}, {req['department']}", f"employees.csv:{req['employee_id']}")]
    if r.get("manager"):
        m = r["manager"]
        out.append(Finding("employee_directory", f"Manager approver: {m['name']} ({m['employee_id']}, {m['level']}, {m['department']})", f"employees.csv:{m['employee_id']}"))
    if r.get("department_head"):
        h = r["department_head"]
        out.append(Finding("employee_directory", f"Department Head approver: {h['name']} ({h['employee_id']}, {h['department']}) - resolved via {r['department_head_basis']}", f"employees.csv:{h['employee_id']}"))
    return out


def describe_budget(b: dict[str, Any]) -> list[Finding]:
    ref = f"department_budgets.csv:{b.get('department')}"
    status = b["status"]
    if status == "no_budget_record":
        return [Finding("budget_check", f"No software budget record for department '{b.get('department')}'", "department_budgets.csv")]
    if status == "cost_unknown":
        return [Finding("budget_check", f"{b['department']} has {_money(b['available'])} available, but the request has no annual cost to compare", ref)]
    if status == "insufficient":
        return [Finding("budget_check", f"Requested {_money(b['requested'])} exceeds {b['department']} available budget {_money(b['available'])} (shortfall {_money(b['shortfall'])})", ref)]
    return [Finding("budget_check", f"Requested {_money(b['requested'])} is within {b['department']} available budget {_money(b['available'])} ({_money(b['remaining_after'])} left after) - a positive budget check is not an approval", ref)]


def describe_catalog(c: dict[str, Any], request: dict[str, Any]) -> list[Finding]:
    out = []
    for m in c["matches"]:
        out.append(Finding(
            "software_catalog",
            f"{m['product_name']} by {m['vendor_name']} [{m['status']}] - {m['category']}, scope {m['scope']}, "
            f"{m['licensed_seats']} seats, {_money(m['annual_cost_usd'])}/yr; match: {', '.join(m['match_types'])}; note: {m['notes']}",
            f"software_catalog.csv:{m['software_id']}",
        ))
    if not c["matches"]:
        out.append(Finding("software_catalog", f"No approved catalog product matches '{request.get('product_name')}', vendor '{request.get('vendor_name')}' or category '{request.get('category')}'", "software_catalog.csv"))
    for p in c["purchase_history"]:
        out.append(Finding("purchase_history", f"{p['purchase_date']}: {p['department']} bought {p['product_name']} from {p['vendor_name']} for {_money(p['annual_amount_usd'])} ({p['status']}; {p['notes']})", f"purchase_history.csv:{p['purchase_id']}"))
    return out


def describe_vendor(v: dict[str, Any]) -> list[Finding]:
    out = []
    reg = v.get("registry")
    if reg:
        out.append(Finding(
            "vendor_registry",
            f"{reg['vendor_name']}: procurement {reg['procurement_status']}, security {reg['security_status']} "
            f"(review date {reg['security_review_date'] or 'none'}), legal terms {reg['legal_terms_status']}; note: {reg['notes']}",
            f"vendors.csv:{reg['vendor_id']}",
        ))
    else:
        out.append(Finding("vendor_registry", f"Vendor '{v['vendor_name']}' is not in the internal vendor registry", "vendors.csv"))
    api = v.get("api")
    if api is not None:
        out.append(Finding(
            "vendor_risk_api",
            f"Risk {api.get('risk_level')}, security review {api.get('security_review_status')} "
            f"(last review {api.get('last_review_date') or 'none'}), processes personal data: {api.get('processes_personal_data')}, "
            f"stores data outside region: {api.get('stores_data_outside_region')}; note: {api.get('notes')}",
            f"GET /vendor-risk/{v['vendor_name']}",
        ))
    else:
        out.append(Finding("vendor_risk_api", f"Vendor-risk lookup failed ({v['api_status']}): {v.get('api_error')} - status NOT inferred", f"GET /vendor-risk/{v['vendor_name']}"))
    out.append(Finding(
        "vendor_risk_api" if api is not None else "vendor_registry",
        f"Security assessment status at reference date {v['reference_date']}: {v['security_assessment']}"
        + (f" (conflicts: {'; '.join(v['conflicts'])})" if v["conflicts"] else ""),
        "Policy section 5 (365-day validity)",
    ))
    return out
