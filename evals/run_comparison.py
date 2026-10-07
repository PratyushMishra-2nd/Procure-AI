"""Run the same case set through every architecture and score the results.

    python evals/run_comparison.py                      # single + staged + rules-only, all cases
    python evals/run_comparison.py --arch single staged --repeats 3
    python evals/run_comparison.py --cases public

Scoring is stricter than the public runner: it checks the exact approval set, required
and forbidden risk flags, missing-information fields and the recommended action, and it
reports how often the policy guard had to correct the model (raw model policy misses).

Outputs (committed so results are reproducible and reviewable):
    evals/results/comparison_runs.csv     one row per case x architecture x repeat
    evals/results/comparison_summary.md   aggregated table + per-case failures
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.mock_service import ensure_mock_api  # noqa: E402
from evals.run_public_evals import evaluate as public_evaluate  # noqa: E402
from src.contracts import CopilotDecision  # noqa: E402
from src.data_access import get_request  # noqa: E402
from src.llm import describe_provider  # noqa: E402
from src.policy_engine import ACTION_SEVERITY, REQUIRED_FIELDS  # noqa: E402
from src.solution import analyze_request  # noqa: E402

RESULTS = ROOT / "evals" / "results"
# USD per million tokens (input, output) - Claude API list prices.
# Gemini on the AI Studio free tier costs $0; models not listed are reported as $0.
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-4-5": (1.0, 5.0)}
SPECIALIST = {"Security", "Privacy", "Legal", "Finance", "CFO"}


def load_cases(which: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    if which in {"public", "all"}:
        for c in json.loads((ROOT / "evals" / "public_cases.json").read_text(encoding="utf-8")):
            cases.append({**c, "suite": "public"})
    if which in {"extended", "all"}:
        for c in json.loads((ROOT / "evals" / "extended_cases.json").read_text(encoding="utf-8")):
            cases.append({**c, "suite": "extended"})
    return cases


def run_case(case: dict[str, Any], arch: str) -> CopilotDecision:
    request = case.get("request") or get_request(case["request_id"])
    previous = os.environ.get("VENDOR_RISK_BASE_URL")
    if case.get("fault") == "vendor_api_down":
        os.environ["VENDOR_RISK_BASE_URL"] = "http://127.0.0.1:9"  # nothing listens here
    try:
        if arch == "rules_only":
            return analyze_request(request, "single", llm=None)
        return analyze_request(request, arch)  # type: ignore[arg-type]
    finally:
        if previous is None:
            os.environ.pop("VENDOR_RISK_BASE_URL", None)
        else:
            os.environ["VENDOR_RISK_BASE_URL"] = previous


def missing_keys(decision: CopilotDecision) -> set[str]:
    keys = set()
    for text in decision.missing_information:
        for key, label in REQUIRED_FIELDS.items():
            if text.startswith(label):
                keys.add(key)
    return keys


def score(case: dict[str, Any], d: CopilotDecision) -> dict[str, Any]:
    if case["suite"] == "public":
        failures = public_evaluate(d, case["expectations"])
        return {"strict_pass": not failures, "public_min_pass": not failures, "failures": failures,
                "action_ok": None, "approvals_exact": None, "escalation_ok": None}

    e = case["expect"]
    failures: list[str] = []
    got = set(d.required_approvals)
    approvals_exact = None
    escalation_ok = True
    if e.get("approvals") is not None:
        want = set(e["approvals"])
        miss, extra = want - got, got - want
        approvals_exact = not miss and not extra
        if miss:
            failures.append(f"missing approvals {sorted(miss)}")
            if miss & SPECIALIST:
                escalation_ok = False
        if extra:
            failures.append(f"extra approvals {sorted(extra)}")
    for a in e.get("approvals_forbidden", []):
        if a in got:
            failures.append(f"forbidden approval {a}")
    for f in e.get("flags_required", []):
        if f not in d.risk_flags:
            failures.append(f"missing flag {f}")
            if f.endswith("review_required") or f in {"budget_insufficient", "vendor_risk_unavailable", "conflicting_vendor_evidence"}:
                escalation_ok = False
    for f in e.get("flags_forbidden", []):
        if f in d.risk_flags:
            failures.append(f"unexpected flag {f}")
    keys = missing_keys(d)
    for k in e.get("missing_required", []):
        if k not in keys:
            failures.append(f"missing-info item absent: {k}")
    if e.get("max_missing") is not None and len(d.missing_information) > e["max_missing"]:
        failures.append(f"too many missing-info items ({len(d.missing_information)})")
    action_ok = d.action in e["actions"]
    if not action_ok:
        failures.append(f"action {d.action} not in {e['actions']}")
        # Under-escalation = recommending routing/purchase when review or clarification was required.
        needed = min(ACTION_SEVERITY[a] for a in e["actions"])
        if needed >= ACTION_SEVERITY["route_for_review"] and ACTION_SEVERITY[d.action] < needed:
            escalation_ok = False
    if not d.human_review_required:
        failures.append("human_review_required is False")
        escalation_ok = False
    return {"strict_pass": not failures, "public_min_pass": None, "failures": failures,
            "action_ok": action_ok, "approvals_exact": approvals_exact, "escalation_ok": escalation_ok}


def cost_usd(d: CopilotDecision) -> float:
    pin, pout = PRICES.get(d.model or "", (0.0, 0.0))
    return (d.diagnostics.get("input_tokens", 0) * pin + d.diagnostics.get("output_tokens", 0) * pout) / 1_000_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", nargs="+", default=["single", "staged", "rules_only"],
                        choices=["single", "staged", "rules_only"])
    parser.add_argument("--cases", default="all", choices=["public", "extended", "all"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--tag", default="", help="suffix for output files, e.g. an effort level")
    args = parser.parse_args()

    ensure_mock_api()
    cases = load_cases(args.cases)
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    provider, model = describe_provider()
    print(f"provider={provider} model={model} cases={len(cases)} repeats={args.repeats}\n")

    for arch in args.arch:
        for rep in range(args.repeats):
            for case in cases:
                started = time.perf_counter()
                try:
                    d = run_case(case, arch)
                except Exception as exc:  # an eval must record crashes, not stop
                    rows.append({"case_id": case["case_id"], "suite": case["suite"], "architecture": arch, "repeat": rep,
                                 "strict_pass": False, "failures": f"CRASH {type(exc).__name__}: {exc}",
                                 "latency_ms": round((time.perf_counter() - started) * 1000, 1)})
                    print(f"CRASH {arch:10} {case['case_id']}: {exc}")
                    continue
                s = score(case, d)
                invalid = d.diagnostics.get("invalid_citations", [])
                row = {
                    "case_id": case["case_id"],
                    "suite": case["suite"],
                    "architecture": arch,
                    "repeat": rep,
                    "title": case["title"],
                    "strict_pass": s["strict_pass"],
                    "public_min_pass": s["public_min_pass"],
                    "correct_next_action": s["action_ok"],
                    "approvals_exact": s["approvals_exact"],
                    "human_escalation_correct": s["escalation_ok"],
                    "grounded_evidence": (not invalid) and (d.ai_status != "ok" or d.diagnostics.get("cited_count", 0) > 0),
                    "policy_followed": s["strict_pass"],
                    "action": d.action,
                    "ai_status": d.ai_status,
                    "guard_corrections": d.diagnostics.get("guard_corrections", 0) if d.ai_status == "ok" else "",
                    "guard_details": " | ".join(f"{a.field}:{a.change}:{a.detail}" for a in d.guard_adjustments),
                    "invalid_citations": len(invalid),
                    "cited_evidence": d.diagnostics.get("cited_count", 0),
                    "evidence_items": len(d.evidence),
                    "latency_ms": d.latency_ms,
                    "llm_calls": d.telemetry.llm_calls if d.telemetry else 0,
                    "tool_calls": d.telemetry.tool_calls if d.telemetry else 0,
                    "input_tokens": d.diagnostics.get("input_tokens", 0),
                    "output_tokens": d.diagnostics.get("output_tokens", 0),
                    "cost_usd": round(cost_usd(d), 5),
                    "ai_error": d.ai_error or "",
                    "failures": " | ".join(s["failures"]),
                    "recommendation": d.recommendation,
                }
                rows.append(row)
                mark = "PASS" if s["strict_pass"] else "FAIL"
                print(f"{mark}  {arch:10} {case['case_id']:7} {d.action:22} {d.latency_ms:>8.0f} ms  llm={row['llm_calls']} tools={row['tool_calls']}"
                      f"  guard={row['guard_corrections']}  {row['failures']}")

    suffix = f"_{args.tag}" if args.tag else ""
    out_csv = RESULTS / f"comparison_runs{suffix}.csv"
    fields = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    out_md = RESULTS / f"comparison_summary{suffix}.md"
    out_md.write_text(summarise(rows, args), encoding="utf-8")
    print(f"\nWrote {out_csv.relative_to(ROOT)} and {out_md.relative_to(ROOT)}")
    print(out_md.read_text(encoding="utf-8").split("## Per-case")[0])


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, round(0.95 * (len(values) - 1)))]


def summarise(rows: list[dict[str, Any]], args: argparse.Namespace) -> str:
    by_arch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_arch[r["architecture"]].append(r)
    archs = list(by_arch)

    def pct(rs, key):
        vals = [r[key] for r in rs if r.get(key) is not None and r.get(key) != ""]
        return f"{sum(bool(v) for v in vals)}/{len(vals)}" if vals else "-"

    lines = [
        "# Architecture comparison",
        "",
        f"Provider `{describe_provider()[0]}`, model `{describe_provider()[1]}`, repeats {args.repeats}, cases `{args.cases}`. "
        "Same cases, same tools, same deterministic engine and guard for every column; `rules_only` is the deterministic engine with no model.",
        "",
        "| Metric | " + " | ".join(archs) + " |",
        "|---|" + "---:|" * len(archs),
    ]
    metrics = [
        ("Public cases meeting minimum expectations", lambda rs: pct([r for r in rs if r["suite"] == "public"], "public_min_pass")),
        ("Extended cases passing strict checks", lambda rs: pct([r for r in rs if r["suite"] == "extended"], "strict_pass")),
        ("Correct next action (extended)", lambda rs: pct([r for r in rs if r["suite"] == "extended"], "correct_next_action")),
        ("Exact approval set (extended)", lambda rs: pct([r for r in rs if r["suite"] == "extended"], "approvals_exact")),
        ("Human escalation correct (extended)", lambda rs: pct([r for r in rs if r["suite"] == "extended"], "human_escalation_correct")),
        ("Runs with only valid evidence citations", lambda rs: pct(rs, "grounded_evidence")),
        ("Raw model policy misses caught by guard (total)", lambda rs: str(sum(int(r["guard_corrections"]) for r in rs if str(r.get("guard_corrections", "")).isdigit())) if any(r.get("ai_status") == "ok" for r in rs) else "n/a"),
        ("Runs needing >=1 guard correction", lambda rs: (f"{sum(1 for r in rs if str(r.get('guard_corrections','')).isdigit() and int(r['guard_corrections'])>0)}/{sum(1 for r in rs if r.get('ai_status')=='ok')}") if any(r.get("ai_status") == "ok" for r in rs) else "n/a"),
        ("AI fallbacks (model unavailable)", lambda rs: str(sum(1 for r in rs if r.get("ai_status") == "unavailable"))),
        ("Avg latency (ms)", lambda rs: f"{_mean([r['latency_ms'] for r in rs if r.get('latency_ms') is not None]):,.0f}"),
        ("p95 latency (ms)", lambda rs: f"{_p95([r['latency_ms'] for r in rs if r.get('latency_ms') is not None]):,.0f}"),
        ("Avg LLM calls", lambda rs: f"{_mean([r['llm_calls'] for r in rs if 'llm_calls' in r]):.2f}"),
        ("Avg tool calls", lambda rs: f"{_mean([r['tool_calls'] for r in rs if 'tool_calls' in r]):.2f}"),
        ("Avg tokens in / out", lambda rs: f"{_mean([r['input_tokens'] for r in rs if 'input_tokens' in r]):,.0f} / {_mean([r['output_tokens'] for r in rs if 'output_tokens' in r]):,.0f}"),
        ("Avg cost per request (USD)", lambda rs: f"{_mean([r['cost_usd'] for r in rs if 'cost_usd' in r]):.4f}"),
    ]
    for name, fn in metrics:
        lines.append(f"| {name} | " + " | ".join(fn(by_arch[a]) for a in archs) + " |")

    lines += ["", "## Per-case results", "", "| Case | " + " | ".join(archs) + " |", "|---|" + "---|" * len(archs)]
    case_ids = list(dict.fromkeys(r["case_id"] for r in rows))
    for cid in case_ids:
        cells = []
        for a in archs:
            rs = [r for r in by_arch[a] if r["case_id"] == cid]
            passed = sum(bool(r["strict_pass"]) for r in rs)
            fails = sorted({f for r in rs for f in str(r.get("failures", "")).split(" | ") if f})
            cell = f"{passed}/{len(rs)}"
            if fails:
                cell += " - " + "; ".join(fails)[:160]
            cells.append(cell)
        lines.append(f"| {cid} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
