from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

from src import agents
from src.contracts import Architecture, CopilotDecision
from src.data_access import get_request
from src.llm import LLMClient, default_client


@lru_cache(maxsize=1)
def _client() -> LLMClient | None:
    return default_client()


def analyze_request(request: dict[str, Any], architecture: Architecture = "single",
                    llm: LLMClient | None | str = "default") -> CopilotDecision:
    """Analyse a request payload (used by the UI for ad-hoc requests and by the extended evals).

    ``llm="default"`` uses the configured Anthropic client; pass ``None`` for the
    deterministic-only path or a fake client in tests.
    """
    client = _client() if llm == "default" else llm
    disabled = llm is None  # caller explicitly asked for the rules-only path
    started = time.perf_counter()
    if architecture == "single":
        decision = agents.run_single(request, client, disabled=disabled)  # type: ignore[arg-type]
    elif architecture == "staged":
        decision = agents.run_staged(request, client, disabled=disabled)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Unknown architecture '{architecture}' (expected 'single' or 'staged')")
    decision.latency_ms = round((time.perf_counter() - started) * 1000, 1)
    return decision


def handle_request(request_id: str, architecture: Architecture = "single") -> CopilotDecision:
    """Assessment adapter used by the public / hidden evaluation harness."""
    return analyze_request(get_request(request_id), architecture)
