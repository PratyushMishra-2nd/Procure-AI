"""Client for the external vendor-risk service.

The starter client called ``raise_for_status()``, which collapses three very
different situations into one exception: the vendor has no risk record (404),
the service is degraded (5xx) and the service is unreachable (connection error /
timeout). Policy section 10 treats an *unavailable* tool differently from a
vendor that simply has no assessment, so they are distinguished here.
"""
from __future__ import annotations

from urllib.parse import quote

import requests

from src.config import vendor_risk_base_url


class VendorRiskError(Exception):
    """Base class for vendor-risk lookups that did not return a record."""

    status: str = "error"

    def __init__(self, message: str, http_status: int | None = None):
        super().__init__(message)
        self.http_status = http_status


class VendorRiskNotFound(VendorRiskError):
    status = "not_found"


class VendorRiskUnavailable(VendorRiskError):
    status = "unavailable"


def get_vendor_risk(vendor_name: str, timeout_seconds: float = 3.0) -> dict:
    url = f"{vendor_risk_base_url()}/vendor-risk/{quote(vendor_name, safe='')}"
    try:
        response = requests.get(url, timeout=timeout_seconds)
    except requests.RequestException as exc:
        raise VendorRiskUnavailable(f"Vendor-risk service unreachable: {type(exc).__name__}") from exc

    if response.status_code == 404:
        raise VendorRiskNotFound(f"No vendor-risk record for '{vendor_name}'", http_status=404)
    if response.status_code >= 400:
        detail = _detail(response)
        raise VendorRiskUnavailable(
            f"Vendor-risk service returned HTTP {response.status_code}: {detail}",
            http_status=response.status_code,
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise VendorRiskUnavailable("Vendor-risk service returned a non-JSON body") from exc
    if not isinstance(payload, dict):
        raise VendorRiskUnavailable("Vendor-risk service returned an unexpected payload shape")
    return payload


def _detail(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(body, dict) and "detail" in body:
        return str(body["detail"])[:200]
    return str(body)[:200]
