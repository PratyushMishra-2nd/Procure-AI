"""Run the same case set through every architecture and score the results.

    python evals/run_comparison.py                          # single + staged + rules-only, all cases
    python evals/run_comparison.py --resume                 # continue after a quota stop / crash
    python evals/run_comparison.py --arch rules_only        # offline baseline (separate files)
    python evals/run_comparison.py --repeats 3 --case-delay 5

Scoring is stricter than the public runner: exact approval set, required and forbidden risk
flags, missing-information fields and the recommended action; it also counts how often the
policy guard had to correct the model (raw model policy misses).

Results go to evals/results/<stem>.* where <stem> is ``live_<provider>_<model>`` or
``rules_only``, so an offline run can never overwrite live results and two models never mix:
    <stem>.jsonl         progress, one line per finished run (written as it goes; --resume reads it)
    <stem>_runs.csv      one row per case x architecture x repeat
    <stem>_summary.md    aggregated table, how each metric is measured, per-case failures
Live runs that fell back to rules-only are excluded from quality metrics and re-run on --resume.
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
from src.llm import describe_provider, provider_config_problem  # noqa: E402
from src.policy_engine import ACTION_SEVERITY, REQUIRED_FIELDS  # noqa: E402
from src.solution import analyze_request  # noqa: E402

RESULTS = ROOT / "evals" / "results"
# USD per million tokens (input, output) - Claude API list prices.
# Gemini on the AI Studio free tier costs $0; models not listed are reported as $0.
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-4-5": (1.0, 5.0)}
SPECIALIST = {"Security", "Privacy", "Legal", "Finance", "CFO"}


def load_cases(which: str) -> list[dict[str, Any]]:
    """public (6, starter) + extended (20, used while building) + heldout (8, written before any model saw them)."""
    cases: list[dict[str, Any]] = []
    for suite, filename in (("public", "public_cases.json"), ("extended", "extended_cases.json"), ("heldout", "heldout_cases.json")):
        if which in {suite, "all"}:
            for c in json.loads((ROOT / "evals" / filename).read_text(encoding="utf-8")):
                cases.append({**c, "suite": suite})
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


def build_row(case: dict[str, Any], arch: str, rep: int, d: CopilotDecision) -> dict[str, Any]:
    s = score(case, d)
    invalid = d.diagnostics.get("invalid_citations", [])
    live = arch != "rules_only"
    return {
        "case_id": case["case_id"],
        "suite": case["suite"],
        "architecture": arch,
        "repeat": rep,
        "title": case["title"],
        "live_valid": (not live) or d.ai_status == "ok",
        "strict_pass": s["strict_pass"],
        "public_min_pass": s["public_min_pass"],
        "correct_next_action": s["action_ok"],
        "approvals_exact": s["approvals_exact"],
        "human_escalation_correct": s["escalation_ok"],
        "grounded_evidence": (not invalid) and (d.ai_status != "ok" or d.diagnostics.get("cited_count", 0) > 0),
        "policy_followed": s["strict_pass"],
        "action": d.action,
        "ai_status": d.ai_status,
        "provider": d.provider or "",
        "model": d.model or "",
        "guard_corrections": d.diagnostics.get("guard_corrections", 0) if d.ai_status == "ok" else "",
        "schema_issues": len(d.diagnostics.get("schema_issues", [])),
        "guard_details": " | ".join(f"{a.field}:{a.change}:{a.detail}" for a in d.guard_adjustments),
        "invalid_citations": len(invalid),
        "cited_evidence": d.diagnostics.get("cited_count", 0),
        "evidence_items": len(d.evidence),
        "latency_ms": d.latency_ms,
        "llm_ms": d.diagnostics.get("llm_ms", 0),  # time inside model calls only (excludes rate-limit pacing)
        "llm_calls": d.telemetry.llm_calls if d.telemetry else 0,
        "tool_calls": d.telemetry.tool_calls if d.telemetry else 0,
        "input_tokens": d.diagnostics.get("input_tokens", 0),
        "output_tokens": d.diagnostics.get("output_tokens", 0),
        "cost_usd": round(cost_usd(d), 5),
        "ai_error": d.ai_error or "",
        "failures": " | ".join(s["failures"]),
        "recommendation": d.recommendation,
    }


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in text).strip("-").lower()


def output_stem(archs: list[str], provider: str | None, model: str, tag: str) -> str:
    """Live and rules-only results never share files, and different models never share files."""
    stem = "rules_only" if set(archs) == {"rules_only"} else f"live_{_slug(provider or 'none')}_{_slug(model)}"
    return stem + (f"_{_slug(tag)}" if tag else "")


def load_progress(path: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if not path.exists():
        return None, []
    meta, rows = None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if "meta" in item:
            meta = item["meta"]
        else:
            rows.append(item)
    return meta, rows


def write_outputs(stem: str, rows: list[dict[str, Any]], meta: dict[str, Any], planned: int) -> Path:
    out_csv = RESULTS / f"{stem}_runs.csv"
    if rows:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    out_md = RESULTS / f"{stem}_summary.md"
    out_md.write_text(summarise(rows, meta, planned), encoding="utf-8")
    return out_md


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", nargs="+", default=["single", "staged", "workflow", "rules_only"],
                        choices=["single", "staged", "workflow", "rules_only"])
    parser.add_argument("--cases", default="all", choices=["public", "extended", "heldout", "all"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--tag", default="", help="suffix for output files")
    parser.add_argument("--resume", action="store_true", help="continue a stopped run (same provider/model only)")
    parser.add_argument("--fresh", action="store_true", help="discard saved progress for this provider/model and start over")
    parser.add_argument("--case-delay", type=float, default=0.0, help="seconds to wait between live cases")
    parser.add_argument("--summarize-only", action="store_true", help="rebuild CSV + summary from saved progress")
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252; never crash on model text
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    provider, model = describe_provider()
    live_archs = [a for a in args.arch if a != "rules_only"]
    if live_archs and provider is None:
        sys.exit(f"No model provider configured ({provider_config_problem()}). Run with --arch rules_only for the offline baseline.")
    stem = output_stem(args.arch, provider, model, args.tag)
    RESULTS.mkdir(parents=True, exist_ok=True)
    progress = RESULTS / f"{stem}.jsonl"
    meta = {"provider": provider if live_archs else None, "model": model if live_archs else "none (rules-only)",
            "archs": args.arch, "cases": args.cases, "repeats": args.repeats}

    saved_meta, saved_rows = load_progress(progress)
    if saved_meta and not args.fresh:
        if not (args.resume or args.summarize_only):
            sys.exit(f"{progress.relative_to(ROOT)} already has results. Use --resume to continue, --fresh to start over, "
                     "or --summarize-only to rebuild the report.")
        if (saved_meta.get("provider"), saved_meta.get("model")) != (meta["provider"], meta["model"]):
            sys.exit(f"Refusing to mix models: saved run used {saved_meta.get('provider')}/{saved_meta.get('model')}, "
                     f"current config is {meta['provider']}/{meta['model']}.")
    else:
        saved_rows = []

    cases = load_cases(args.cases)
    plan = [(rep, case, arch) for rep in range(args.repeats) for case in cases for arch in args.arch]  # interleaved per case
    # keep only finished rows; fallback / crashed rows are re-run on resume
    kept = [r for r in saved_rows if r.get("live_valid") and not str(r.get("failures", "")).startswith("CRASH")]
    done = {(r["case_id"], r["architecture"], r["repeat"]) for r in kept}
    rows: list[dict[str, Any]] = list(kept)

    if args.summarize_only:
        out = write_outputs(stem, rows, saved_meta or meta, len(plan))
        print(out.read_text(encoding="utf-8"))
        return

    ensure_mock_api()
    with progress.open("w", encoding="utf-8") as f:  # rewrite: meta + kept rows
        f.write(json.dumps({"meta": meta}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")

    todo = [p for p in plan if (p[1]["case_id"], p[2], p[0]) not in done]
    print(f"provider={meta['provider']} model={meta['model']} planned={len(plan)} already-done={len(plan) - len(todo)} to-run={len(todo)}\n")
    stopped = None
    try:
        for i, (rep, case, arch) in enumerate(todo):
            if arch != "rules_only" and args.case_delay and i:
                time.sleep(args.case_delay)
            started = time.perf_counter()
            try:
                d = run_case(case, arch)
                row = build_row(case, arch, rep, d)
            except Exception as exc:  # an eval must record crashes, not stop
                row = {"case_id": case["case_id"], "suite": case["suite"], "architecture": arch, "repeat": rep,
                       "live_valid": False, "strict_pass": False, "failures": f"CRASH {type(exc).__name__}: {exc}",
                       "latency_ms": round((time.perf_counter() - started) * 1000, 1)}
            rows.append(row)
            with progress.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            mark = "PASS" if row["strict_pass"] else "FAIL"
            if not row["live_valid"]:
                mark = "FALLBACK" if not str(row.get("failures", "")).startswith("CRASH") else "CRASH"
            print(f"{mark:8} {arch:10} {case['case_id']:7} {row.get('action', ''):22} {row.get('latency_ms') or 0:>8.0f} ms"
                  f"  llm={row.get('llm_calls', '')} tools={row.get('tool_calls', '')} guard={row.get('guard_corrections', '')}"
                  f"  {row.get('ai_error', '') if not row['live_valid'] else row.get('failures', '')}", flush=True)
            if not row["live_valid"] and "quota" in str(row.get("ai_error", "")).lower():
                stopped = f"provider quota exhausted after {i + 1}/{len(todo)} runs: {row['ai_error']}"
                break
    except KeyboardInterrupt:
        stopped = "interrupted"

    out = write_outputs(stem, rows, meta, len(plan))
    print(f"\nProgress: {progress.relative_to(ROOT)}  Summary: {out.relative_to(ROOT)}")
    if stopped:
        print(f"STOPPED - {stopped}. Completed runs are saved; continue later with the same command plus --resume.")
    print(out.read_text(encoding="utf-8").split("## Per-case")[0])


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, round(0.95 * (len(values) - 1)))]


HOW_MEASURED = """## How each metric is measured

- **Valid runs**: live runs where the model actually answered (`ai_status == ok`). A live run that fell back to rules-only (quota, outage, bad output) is *excluded* from every quality metric below and counted separately - it would otherwise score the rules engine as if it were the model.
- **Public minimum checks**: the starter harness's own `evaluate()` on the 6 public cases.
- **Held-out**: 8 cases written from the policy text before any model was run on them and never used to tune prompts or rules (`evals/heldout_cases.json`). Extended cases *were* used while building, so held-out is the honest generalisation check.
- **Strict checks (extended / held-out)**: exact approval set, required + forbidden risk flags, missing-information fields and the recommended action, against hand-labelled expectations in `evals/extended_cases.json`.
- **Human escalation correct**: no required Security / Privacy / Legal / Finance / CFO approval missing and the action is not less cautious than review / clarification when those are expected.
- **Guard corrections**: times the deterministic policy guard had to add, remove, raise, correct or replace something in the model's draft (raw model policy misses). Final outputs include the corrections.
- **Valid citations**: every evidence ID the model cited exists in the tool results of that run.
- **Latency**: wall-clock time of `analyze_request`, *including* client-side waiting to stay under the free tier's tokens-per-minute cap - so on Groq it mostly measures the rate limit. **Model time** is the sum of the model calls themselves and is the fairer speed comparison. **Model / tool calls and tokens**: from the run's own telemetry (provider-reported token usage).
"""


def summarise(rows: list[dict[str, Any]], meta: dict[str, Any], planned: int) -> str:
    by_arch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_arch[r["architecture"]].append(r)
    archs = [a for a in ("single", "staged", "workflow", "rules_only") if a in by_arch]
    valid = {a: [r for r in by_arch[a] if r.get("live_valid")] for a in archs}

    def pct(rs, key):
        vals = [r[key] for r in rs if r.get(key) is not None and r.get(key) != ""]
        return f"{sum(bool(v) for v in vals)}/{len(vals)}" if vals else "-"

    finished = sum(1 for r in rows if r.get("live_valid"))
    lines = ["# Architecture comparison", ""]
    if finished < planned:
        lines += [f"> **INCOMPLETE:** {finished} of {planned} planned runs finished. Numbers below cover finished runs only.", ""]
    lines += [
        f"Provider `{meta.get('provider')}`, model `{meta.get('model')}`, repeats {meta.get('repeats')}, cases `{meta.get('cases')}`. "
        "Same cases, tools, deterministic engine and guard for every column; `rules_only` is the deterministic engine with no model (offline baseline, says nothing about model reasoning).",
        "",
        "| Metric | " + " | ".join(archs) + " |",
        "|---|" + "---:|" * len(archs),
    ]
    ext = lambda rs: [r for r in rs if r["suite"] == "extended"]  # noqa: E731
    live = lambda a: a != "rules_only"  # noqa: E731
    metrics = [
        ("Valid runs (live: model answered)", lambda a, rs: f"{len(rs)}/{len(by_arch[a])}"),
        ("Runs that fell back to rules-only (excluded)", lambda a, rs: str(len(by_arch[a]) - len(rs)) if live(a) else "n/a"),
        ("Public cases meeting minimum expectations", lambda a, rs: pct([r for r in rs if r["suite"] == "public"], "public_min_pass")),
        ("Extended cases passing strict checks", lambda a, rs: pct(ext(rs), "strict_pass")),
        ("**Held-out cases passing strict checks**", lambda a, rs: pct([r for r in rs if r["suite"] == "heldout"], "strict_pass")),
        ("Correct next action (held-out)", lambda a, rs: pct([r for r in rs if r["suite"] == "heldout"], "correct_next_action")),
        ("Correct next action (extended)", lambda a, rs: pct(ext(rs), "correct_next_action")),
        ("Exact approval set (extended)", lambda a, rs: pct(ext(rs), "approvals_exact")),
        ("Human escalation correct (extended)", lambda a, rs: pct(ext(rs), "human_escalation_correct")),
        ("Runs with only valid evidence citations", lambda a, rs: pct(rs, "grounded_evidence") if live(a) else "n/a"),
        ("Guard corrections (total raw model misses)", lambda a, rs: str(sum(int(r["guard_corrections"]) for r in rs if str(r.get("guard_corrections", "")).isdigit())) if live(a) else "n/a"),
        ("Runs needing >=1 guard correction", lambda a, rs: f"{sum(1 for r in rs if str(r.get('guard_corrections', '')).isdigit() and int(r['guard_corrections']) > 0)}/{len(rs)}" if live(a) else "n/a"),
        ("Avg latency (ms)", lambda a, rs: f"{_mean([r['latency_ms'] for r in rs if r.get('latency_ms') is not None]):,.0f}"),
        ("Avg model time (ms, excl. pacing)", lambda a, rs: f"{_mean([r['llm_ms'] for r in rs if r.get('llm_ms') is not None]):,.0f}" if any('llm_ms' in r for r in rs) else "-"),
        ("p95 latency (ms)", lambda a, rs: f"{_p95([r['latency_ms'] for r in rs if r.get('latency_ms') is not None]):,.0f}"),
        ("Avg LLM calls", lambda a, rs: f"{_mean([r.get('llm_calls', 0) for r in rs]):.2f}"),
        ("Avg tool calls", lambda a, rs: f"{_mean([r.get('tool_calls', 0) for r in rs]):.2f}"),
        ("Avg tokens in / out", lambda a, rs: f"{_mean([r.get('input_tokens', 0) for r in rs]):,.0f} / {_mean([r.get('output_tokens', 0) for r in rs]):,.0f}"),
        ("Avg cost per request (USD)", lambda a, rs: f"{_mean([r.get('cost_usd', 0) for r in rs]):.4f}"),
    ]
    for name, fn in metrics:
        lines.append(f"| {name} | " + " | ".join(fn(a, valid[a]) for a in archs) + " |")

    lines += ["", HOW_MEASURED, "## Per-case results (valid runs)", "", "| Case | " + " | ".join(archs) + " |", "|---|" + "---|" * len(archs)]
    for cid in dict.fromkeys(r["case_id"] for r in rows):
        cells = []
        for a in archs:
            rs = [r for r in valid[a] if r["case_id"] == cid]
            if not rs:
                cells.append("not run / fell back")
                continue
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
