"""Eval runner integrity: live/offline separation, fallback exclusion, resume."""
from __future__ import annotations

import json
import sys

import pytest

import evals.run_comparison as rc


def test_offline_and_live_results_never_share_files():
    assert rc.output_stem(["rules_only"], None, "x", "") == "rules_only"
    assert rc.output_stem(["single", "staged", "rules_only"], "groq", "openai/gpt-oss-120b", "") == "live_groq_openai-gpt-oss-120b"
    assert rc.output_stem(["single"], "gemini", "gemini-3.7-flash", "") != rc.output_stem(["single"], "groq", "openai/gpt-oss-120b", "")


def _row(case, arch, valid=True, passed=True, rep=0):
    return {"case_id": case, "suite": "extended", "architecture": arch, "repeat": rep, "live_valid": valid,
            "strict_pass": passed, "correct_next_action": passed, "approvals_exact": True, "human_escalation_correct": True,
            "grounded_evidence": True, "guard_corrections": 0 if valid else "", "latency_ms": 10, "llm_calls": 2,
            "tool_calls": 5, "input_tokens": 1, "output_tokens": 1, "cost_usd": 0, "failures": "", "ai_status": "ok" if valid else "unavailable"}


def test_fallback_runs_are_excluded_from_live_quality_metrics():
    rows = [_row("EXT-01", "single"), _row("EXT-02", "single", valid=False, passed=True)]
    md = rc.summarise(rows, {"provider": "groq", "model": "m", "repeats": 1, "cases": "all"}, planned=2)
    assert "INCOMPLETE" in md
    assert "| Valid runs (live: model answered) | 1/2 |" in md
    assert "| Extended cases passing strict checks | 1/1 |" in md  # the fallback "pass" is not counted
    assert "| Runs that fell back to rules-only (excluded) | 1 |" in md


def test_resume_refuses_to_mix_models_and_reruns_fallbacks(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "RESULTS", tmp_path)
    monkeypatch.setattr(rc, "ROOT", tmp_path)
    monkeypatch.setattr(rc, "ensure_mock_api", lambda: None)
    monkeypatch.setattr(rc, "describe_provider", lambda: ("groq", "m1"))
    cases = [{"case_id": "C1", "suite": "extended", "title": "t", "expect": {}}, {"case_id": "C2", "suite": "extended", "title": "t", "expect": {}}]
    monkeypatch.setattr(rc, "load_cases", lambda which: cases)
    progress = tmp_path / "live_groq_m1.jsonl"
    progress.write_text("\n".join(json.dumps(x) for x in [
        {"meta": {"provider": "groq", "model": "m1"}}, _row("C1", "single"), _row("C2", "single", valid=False)]) + "\n")
    ran = []

    def fake_build(case, arch, rep, d):
        ran.append((case["case_id"], arch))
        return _row(case["case_id"], arch)
    monkeypatch.setattr(rc, "run_case", lambda case, arch: None)
    monkeypatch.setattr(rc, "build_row", fake_build)
    monkeypatch.setattr(sys, "argv", ["x", "--arch", "single", "--resume"])
    rc.main()
    assert ran == [("C2", "single")]  # C1 kept, the fallback C2 re-run
    saved = [json.loads(l) for l in progress.read_text().splitlines()]
    assert sum(1 for x in saved if x.get("case_id") == "C2" and x["live_valid"]) == 1

    monkeypatch.setattr(rc, "describe_provider", lambda: ("groq", "m2"))
    monkeypatch.setattr(rc, "output_stem", lambda *a: "live_groq_m1")  # force same file, different model
    with pytest.raises(SystemExit, match="Refusing to mix models"):
        rc.main()


def test_existing_results_are_not_silently_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "RESULTS", tmp_path)
    monkeypatch.setattr(rc, "ROOT", tmp_path)
    monkeypatch.setattr(rc, "describe_provider", lambda: (None, "none"))
    (tmp_path / "rules_only.jsonl").write_text(json.dumps({"meta": {"provider": None, "model": "none (rules-only)"}}) + "\n")
    monkeypatch.setattr(sys, "argv", ["x", "--arch", "rules_only"])
    with pytest.raises(SystemExit, match="already has results"):
        rc.main()
