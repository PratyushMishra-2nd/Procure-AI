from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mock_api.app import app  # noqa: E402
from src import vendor_client  # noqa: E402
from src.llm import LLMResponse, LLMUnavailable, to_response  # noqa: E402

_client = TestClient(app)


class _Resp:
    def __init__(self, r):
        self.status_code = r.status_code
        self._r = r
        self.text = r.text

    def json(self):
        return self._r.json()


@pytest.fixture(autouse=True)
def in_process_vendor_api(monkeypatch):
    """Route the vendor client to the mock API in-process (no server needed)."""

    def fake_get(url: str, timeout: float = 3.0):
        path = url.split("127.0.0.1:8001", 1)[-1]
        return _Resp(_client.get(path))

    monkeypatch.setattr(vendor_client.requests, "get", fake_get)
    monkeypatch.setenv("VENDOR_RISK_BASE_URL", "http://127.0.0.1:8001")
    yield


@pytest.fixture
def vendor_api_down(monkeypatch):
    import requests

    def boom(url: str, timeout: float = 3.0):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(vendor_client.requests, "get", boom)


class FakeLLM:
    """Scripted stand-in for the Anthropic client.

    ``script`` is a list of callables ``(agent, messages, tools) -> LLMResponse`` or
    ``Exception`` instances, consumed in order.
    """

    model = "fake-model"

    def __init__(self, script: list[Any]):
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def create(self, *, agent, system, messages, tools, output_schema, telemetry) -> LLMResponse:
        self.calls.append({"agent": agent, "system": system, "messages": [dict(m) for m in messages], "tools": tools})
        telemetry.record_llm_call(agent, input_tokens=100, output_tokens=50, ms=1.0)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step(agent, messages, tools)


def call_tools(*names: str):
    def step(agent, messages, tools):
        content = [{"type": "tool_use", "id": f"tu_{i}", "name": n, "input": {} if n not in {"search_software_catalog", "get_vendor_risk"} else ({"query": None, "category": None} if n == "search_software_catalog" else {"vendor_name": None})}
                   for i, n in enumerate(names)]
        return to_response(content, "tool_use")
    return step


def answer(payload: dict[str, Any] | Any):
    def step(agent, messages, tools):
        body = payload(agent, messages) if callable(payload) else payload
        return to_response([{"type": "text", "text": json.dumps(body)}], "end_turn")
    return step


ALL_TOOLS = ("lookup_requester", "check_budget", "search_software_catalog", "get_vendor_risk", "evaluate_policy_rules")


def decision_draft(**overrides: Any) -> dict[str, Any]:
    base = {
        "need_summary": "need",
        "overlap_assessment": "no_overlap",
        "overlap_explanation": "",
        "insufficient_fields": [],
        "clarification_questions": [],
        "prompt_injection_observed": False,
        "action": "route_for_review",
        "recommendation": "Route for review.",
        "rationale": "Because.",
        "required_approvals": [],
        "risk_flags": [],
        "cited_evidence_ids": ["E1"],
        "next_step": "Procurement sends the pack to reviewers.",
    }
    base.update(overrides)
    return base


def evidence_pack(**overrides: Any) -> dict[str, Any]:
    base = {
        "need_summary": "need",
        "overlap_assessment": "no_overlap",
        "overlap_explanation": "",
        "insufficient_fields": [],
        "clarification_questions": [],
        "prompt_injection_observed": False,
        "key_facts": [{"evidence_id": "E1", "fact": "fact"}],
        "concerns": [],
    }
    base.update(overrides)
    return base


__all__ = ["FakeLLM", "call_tools", "answer", "ALL_TOOLS", "decision_draft", "evidence_pack", "LLMUnavailable"]
