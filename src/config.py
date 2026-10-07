"""Runtime configuration.

Every knob is read from the environment (``.env`` is loaded by ``src/__init__``)
so the same code runs locally, in the eval harness and in tests.
"""
from __future__ import annotations

import os
import re
from datetime import date
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
POLICY_PATH = DATA_DIR / "procurement_policy.md"

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "low"


@lru_cache(maxsize=1)
def reference_date() -> date:
    """Policy reference date, parsed from the policy file (never the machine clock).

    The starter README requires date checks to use the snapshot date defined in
    ``data/procurement_policy.md``. Parsing it keeps the policy the single source
    of truth when a hidden data set ships a different snapshot.
    """
    text = POLICY_PATH.read_text(encoding="utf-8")
    match = re.search(r"reference date:\*{0,2}\s*(\d{4}-\d{2}-\d{2})", text, flags=re.IGNORECASE)
    if not match:
        raise RuntimeError("Policy file does not declare a data snapshot / reference date")
    return date.fromisoformat(match.group(1))


def model_name() -> str:
    return os.getenv("MODEL_NAME") or DEFAULT_MODEL


def effort() -> str:
    value = (os.getenv("COPILOT_EFFORT") or DEFAULT_EFFORT).strip().lower()
    return value if value in {"low", "medium", "high", "xhigh", "max"} else DEFAULT_EFFORT


def llm_disabled() -> bool:
    """`COPILOT_LLM=off` forces the deterministic-only path (used by tests / offline demos)."""
    return (os.getenv("COPILOT_LLM") or "").strip().lower() in {"off", "0", "false", "disabled"}


def vendor_risk_base_url() -> str:
    return (os.getenv("VENDOR_RISK_BASE_URL") or "http://127.0.0.1:8001").rstrip("/")


def max_agent_turns() -> int:
    try:
        return max(1, int(os.getenv("COPILOT_MAX_TURNS", "6")))
    except ValueError:
        return 6
