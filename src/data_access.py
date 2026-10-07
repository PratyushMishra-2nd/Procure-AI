"""Typed, read-only access to the synthetic business data.

The starter version returned pandas DataFrames, which silently turns blank CSV
cells into ``NaN`` floats (e.g. the CEO's empty ``manager_id`` or a new vendor's
empty ``security_review_date``). ``NaN`` is truthy and ``str(NaN) == "nan"``, so
"is this review date present?" checks pass when they should fail. This module
uses the stdlib ``csv`` reader and normalises blanks to ``None`` instead.
"""
from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.config import DATA_DIR

ROOT = Path(__file__).resolve().parents[1]

INT_FIELDS = {
    "annual_software_budget_usd",
    "committed_usd",
    "available_usd",
    "annual_cost_usd",
    "licensed_seats",
    "annual_amount_usd",
}


def _clean(row: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        value = (value or "").strip()
        if value == "":
            out[key] = None
        elif key in INT_FIELDS:
            out[key] = float(value) if "." in value else int(value)
        else:
            out[key] = value
    return out


@lru_cache(maxsize=None)
def _read_csv(name: str) -> tuple[dict[str, Any], ...]:
    with (DATA_DIR / name).open(encoding="utf-8", newline="") as f:
        return tuple(_clean(r) for r in csv.DictReader(f))


def load_employees() -> list[dict[str, Any]]:
    return [dict(r) for r in _read_csv("employees.csv")]


def load_budgets() -> list[dict[str, Any]]:
    return [dict(r) for r in _read_csv("department_budgets.csv")]


def load_software_catalog() -> list[dict[str, Any]]:
    return [dict(r) for r in _read_csv("software_catalog.csv")]


def load_vendors() -> list[dict[str, Any]]:
    return [dict(r) for r in _read_csv("vendors.csv")]


def load_purchase_history() -> list[dict[str, Any]]:
    return [dict(r) for r in _read_csv("purchase_history.csv")]


def load_requests() -> list[dict]:
    return json.loads((DATA_DIR / "requests.json").read_text(encoding="utf-8"))


def get_request(request_id: str) -> dict:
    for request in load_requests():
        if request["request_id"] == request_id:
            return request
    raise KeyError(f"Unknown request_id: {request_id}")


def load_policy_text() -> str:
    return (DATA_DIR / "procurement_policy.md").read_text(encoding="utf-8")


def find_employee(employee_id: str | None) -> dict[str, Any] | None:
    if not employee_id:
        return None
    return next((e for e in load_employees() if e["employee_id"] == employee_id), None)


def find_budget(department: str | None) -> dict[str, Any] | None:
    if not department:
        return None
    return next((b for b in load_budgets() if b["department"].lower() == department.lower()), None)


def find_vendor(vendor_name: str | None) -> dict[str, Any] | None:
    if not vendor_name:
        return None
    key = vendor_name.strip().lower()
    return next((v for v in load_vendors() if v["vendor_name"].lower() == key), None)


def clear_caches() -> None:
    """Drop cached CSV contents (tests swap data in place)."""
    _read_csv.cache_clear()
