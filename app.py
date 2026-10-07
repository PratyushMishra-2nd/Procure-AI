from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

import requests
import streamlit as st

from src import config
from src.contracts import CopilotDecision
from src.data_access import load_employees, load_requests
from src.llm import describe_provider
from src.solution import analyze_request

ROOT = Path(__file__).resolve().parent
DECISIONS_LOG = ROOT / "runs" / "human_decisions.jsonl"

ACTION_STYLE = {
    "route_for_approval": ("Route for approval", "#1f7a4d"),
    "reuse_existing_tool": ("Check existing tool first", "#8a6d00"),
    "route_for_review": ("Specialist review required", "#b35c00"),
    "request_clarification": ("Clarification needed", "#a12a2a"),
}
FLAG_TONE = {
    "budget_insufficient": "red", "conflicting_vendor_evidence": "red", "vendor_risk_unavailable": "red",
    "prompt_injection_detected": "red", "vendor_review_expired": "red", "missing_information": "red",
    "ai_assessment_unavailable": "red", "existing_tool_overlap": "orange", "urgency_pressure": "orange",
}

st.set_page_config(page_title="Procurement Request Copilot", page_icon=":clipboard:", layout="wide")
st.markdown(
    """<style>
    .pill{display:inline-block;padding:2px 10px;margin:2px 4px 2px 0;border-radius:999px;font-size:0.82rem;
          border:1px solid rgba(128,128,128,.35)}
    .action{padding:14px 18px;border-radius:10px;color:#fff;margin-bottom:8px}
    .action h3{margin:0;color:#fff} .action p{margin:4px 0 0 0;color:#fff}
    .untrusted{border-left:3px solid #b35c00;padding:6px 10px;background:rgba(179,92,0,.08);border-radius:4px}
    </style>""",
    unsafe_allow_html=True,
)


def api_healthy() -> bool:
    try:
        return requests.get(f"{config.vendor_risk_base_url()}/health", timeout=0.5).ok
    except requests.RequestException:
        return False


# ------------------------------------------------------------------ sidebar
REQUESTS = load_requests()
BY_ID = {r["request_id"]: r for r in REQUESTS}

with st.sidebar:
    st.header("Request")
    source = st.radio("Source", ["Request queue", "New request"], horizontal=True, label_visibility="collapsed")
    st.header("Architecture")
    arch_label = st.radio(
        "Architecture",
        ["A - Single agent", "B - Analyst + Reviewer", "Rules only (no AI)"],
        label_visibility="collapsed",
        help="A and B use the same tools, rules and guard; only the split of model work differs.",
    )
    architecture = {"A - Single agent": "single", "B - Analyst + Reviewer": "staged"}.get(arch_label, "rules_only")
    st.divider()
    provider, model = describe_provider()
    ai_line = f"AI: **{provider}** · `{model}`" if provider else "AI: not configured - rules-only fallback (set GEMINI_API_KEY in .env)"
    st.caption(
        f"{ai_line}  \n"
        f"Vendor-risk API: {'up' if api_healthy() else 'DOWN - lookups will report unavailable'}  \n"
        f"Policy reference date: {config.reference_date().isoformat()}"
    )

# ------------------------------------------------------------------ request selection / form
st.title("Procurement Request Copilot")
st.caption("Gathers evidence, applies policy rules and recommends the next action. Approvals stay with people.")

if source == "Request queue":
    request_id = st.selectbox("Request", list(BY_ID), format_func=lambda rid: f"{rid} - {BY_ID[rid]['product_name']}")
    request = BY_ID[request_id]
else:
    employees = {e["employee_id"]: e for e in load_employees()}
    with st.form("new_request"):
        c1, c2, c3 = st.columns(3)
        requester_id = c1.selectbox("Requester", list(employees) + ["(unknown)"],
                                    format_func=lambda e: f"{e} - {employees[e]['name']} ({employees[e]['department']})" if e in employees else e)
        product = c2.text_input("Product", "")
        vendor = c3.text_input("Vendor", "")
        c4, c5, c6 = st.columns(3)
        category = c4.text_input("Category", "")
        cost = c5.text_input("Annual cost (USD)", "", help="Leave blank if unknown")
        users = c6.text_input("Users / seats", "")
        c7, c8 = st.columns(2)
        data_level = c7.selectbox("Data access", ["none", "internal_documents", "internal_marketing", "confidential_documents",
                                                  "employee_pii", "customer_pii", "source_code", "production_telemetry", "credentials", "unknown"])
        integrations = c8.text_input("Integrations (comma separated)", "")
        justification = st.text_area("Business justification", "")
        urgency = st.select_slider("Urgency", ["normal", "high", "urgent"])
        submitted = st.form_submit_button("Use this request")
    if submitted:
        st.session_state["adhoc"] = {
            "request_id": f"ADHOC-{datetime.now(timezone.utc):%H%M%S}",
            "requester_id": None if requester_id == "(unknown)" else requester_id,
            "product_name": product or None, "vendor_name": vendor or None, "category": category or None,
            "annual_cost_usd": cost or None, "user_count": users or None,
            "business_justification": justification, "data_access_level": data_level,
            "requested_integrations": [i.strip() for i in integrations.split(",") if i.strip()], "urgency": urgency,
        }
    request = st.session_state.get("adhoc")
    if request is None:
        st.info("Fill in the form and press **Use this request**.")
        st.stop()

result_key = f"result::{request['request_id']}::{architecture}"

# ------------------------------------------------------------------ layout
left, right = st.columns([0.9, 1.1], gap="large")

with left:
    st.subheader("1 · Request")
    st.markdown(
        f"**{request.get('product_name') or '—'}** from **{request.get('vendor_name') or '—'}**  \n"
        f"Category: {request.get('category') or '—'} · Urgency: {request.get('urgency') or '—'}"
    )
    amount = request.get("annual_cost_usd")
    try:
        cost_text = f"${float(amount):,.2f}".replace(".00", "") if amount not in (None, "") else "**missing**"
    except (TypeError, ValueError):
        cost_text = f"**unparseable** ({amount})"
    st.markdown(
        f"Annual cost: {cost_text} · Users: {request.get('user_count') or '**missing**'} · "
        f"Requester: `{request.get('requester_id') or 'missing'}`"
    )
    st.markdown(f"Data access: `{request.get('data_access_level')}` · Integrations: {', '.join(request.get('requested_integrations') or []) or 'none'}")
    st.markdown("Business justification *(untrusted text - shown as data)*")
    justification = html.escape(request.get("business_justification") or "") or "<i>empty</i>"
    st.markdown(f"<div class='untrusted'>{justification}</div>", unsafe_allow_html=True)
    st.write("")
    if st.button("Run analysis", type="primary", width="stretch"):
        with st.spinner("Gathering evidence and checking policy..."):
            if architecture == "rules_only":
                decision = analyze_request(request, "single", llm=None)
            else:
                decision = analyze_request(request, architecture)  # type: ignore[arg-type]
        st.session_state[result_key] = decision.model_dump()

decision_dict = st.session_state.get(result_key)
with right:
    st.subheader("2 · Recommendation")
    if not decision_dict:
        st.info("Run the analysis to see the recommendation, evidence and required approvals.")
        st.stop()
    d = CopilotDecision.model_validate(decision_dict)
    label, color = ACTION_STYLE.get(d.action, (d.action, "#555"))
    st.markdown(f"<div class='action' style='background:{color}'><h3>{label}</h3><p>{html.escape(d.recommendation)}</p></div>", unsafe_allow_html=True)
    if d.ai_status != "ok":
        st.warning(f"AI assessment unavailable - rules-only result. {d.ai_error or ''}")
    st.markdown(f"**Next step:** {d.next_step}")
    if d.rationale:
        st.markdown(f"**Why:** {d.rationale}")
    if d.need_summary:
        st.caption(f"Understood need: {d.need_summary}")
    if d.overlap_assessment:
        st.caption(f"Overlap: {d.overlap_assessment}")

    st.markdown("**Approvals required**")
    st.markdown("".join(
        f"<span class='pill'><b>{a}</b>{' · ' + d.approver_names[a] if a in d.approver_names else ''}</span>"
        for a in d.required_approvals) or "—", unsafe_allow_html=True)
    with st.expander("Why each approval is required"):
        for a, reasons in d.approval_reasons.items():
            st.markdown(f"**{a}**  \n" + "  \n".join(f"- {r}" for r in reasons))

    st.markdown("**Risk flags**")
    st.markdown("".join(f"<span class='pill' style='border-color:{FLAG_TONE.get(f, 'gray')}'>{f}</span>" for f in d.risk_flags) or "none",
                unsafe_allow_html=True)
    with st.expander("Flag details"):
        for f, reasons in d.flag_reasons.items():
            st.markdown(f"**{f}**  \n" + "  \n".join(f"- {r}" for r in reasons))

    if d.missing_information:
        st.markdown("**Missing information**")
        for m in d.missing_information:
            st.markdown(f"- {m}")
    if d.clarification_questions:
        st.markdown("**Questions for the requester**")
        for q in d.clarification_questions:
            st.markdown(f"- {q}")

# ------------------------------------------------------------------ evidence + controls
st.divider()
st.subheader("3 · Evidence")
cited = set(d.cited_evidence)
rows = [{"": "★" if f"E{i}" in cited else "", "ID": f"E{i}", "Source": e.source, "Finding": e.finding, "Reference": e.reference or ""}
        for i, e in enumerate(d.evidence, start=1)]
st.dataframe(rows, hide_index=True, width="stretch",
             column_config={"Finding": st.column_config.TextColumn(width="large")})
st.caption("★ = cited by the AI recommendation. Every row comes from a tool or rule result, not from model text.")

if d.guard_adjustments:
    with st.expander(f"Policy guard corrected the AI draft {len(d.guard_adjustments)} time(s)"):
        for a in d.guard_adjustments:
            st.markdown(f"- **{a.field}** {a.change}: {a.detail}")

tel = d.telemetry
m = st.columns(5)
m[0].metric("Latency", f"{(d.latency_ms or 0) / 1000:.1f} s")
m[1].metric("LLM calls", tel.llm_calls if tel else 0)
m[2].metric("Tool calls", tel.tool_calls if tel else 0)
m[3].metric("Tokens in/out", f"{d.diagnostics.get('input_tokens', 0):,}/{d.diagnostics.get('output_tokens', 0):,}")
m[4].metric("Architecture", {"single": "A", "staged": "B"}.get(d.architecture, d.architecture) if d.ai_status == "ok" else "rules")
with st.expander("Agent trace"):
    for ev in d.diagnostics.get("trace", []):
        if ev["kind"] == "llm":
            st.markdown(f"`{ev['t_ms']:>6} ms` 🧠 **{ev['agent']}** model call - {ev['ms']} ms, {ev['input_tokens']} in / {ev['output_tokens']} out, stop `{ev['stop_reason']}`")
        elif ev["kind"] == "tool":
            st.markdown(f"`{ev['t_ms']:>6} ms` 🔧 {ev['caller']} → `{ev['name']}`({json.dumps(ev['args']) if ev['args'] else ''}){'' if ev['ok'] else ' ❌'}")
        else:
            st.markdown(f"`{ev['t_ms']:>6} ms` 🌐 {ev['name']} {'ok' if ev['ok'] else '- ' + ev['detail']}")
    if d.diagnostics.get("analyst_pack"):
        st.markdown("**Analyst → Reviewer evidence pack**")
        st.json(d.diagnostics["analyst_pack"])
with st.expander("Raw ProcurementDecision JSON"):
    st.json(json.loads(d.model_dump_json(exclude={"diagnostics"})))

# ------------------------------------------------------------------ human review
st.divider()
st.subheader("4 · Human review")
st.caption("The copilot never approves, purchases or changes budgets. A named person records the decision here.")
with st.form(f"review::{result_key}"):
    c1, c2 = st.columns([1, 2])
    reviewer = c1.text_input("Reviewer name / role")
    choice = c1.radio("Decision", ["Accept routing as recommended", "Return to requester", "Escalate / override routing", "Reject request"])
    comment = c2.text_area("Comment (required for overrides)", height=140)
    record = st.form_submit_button("Record decision")
if record:
    if not reviewer.strip():
        st.error("Enter the reviewer's name or role.")
    elif choice != "Accept routing as recommended" and not comment.strip():
        st.error("A comment is required when the decision differs from the recommendation.")
    else:
        DECISIONS_LOG.parent.mkdir(exist_ok=True)
        entry = {
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "request_id": d.request_id,
            "architecture": d.architecture, "copilot_action": d.action, "required_approvals": d.required_approvals,
            "reviewer": reviewer.strip(), "decision": choice, "comment": comment.strip(),
        }
        with DECISIONS_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        st.success(f"Recorded: {choice} by {reviewer.strip()}.")
if DECISIONS_LOG.exists():
    history = [json.loads(line) for line in DECISIONS_LOG.read_text(encoding="utf-8").splitlines() if line.strip()]
    mine = [h for h in history if h["request_id"] == d.request_id]
    if mine:
        st.markdown("**Decision log for this request**")
        st.dataframe(mine[::-1], hide_index=True, width="stretch")
