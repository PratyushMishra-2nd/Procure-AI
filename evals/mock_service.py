"""Make sure the mock vendor-risk API is up before an eval run.

Without it every vendor lookup fails, the copilot (correctly) reports
`vendor_risk_unavailable` for every case, and the eval measures an outage
instead of the system. The starter runner did not check this.
"""
from __future__ import annotations

import atexit
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


def _healthy(base: str) -> bool:
    try:
        return requests.get(f"{base}/health", timeout=0.5).ok
    except requests.RequestException:
        return False


def ensure_mock_api() -> None:
    base = (os.getenv("VENDOR_RISK_BASE_URL") or "http://127.0.0.1:8001").rstrip("/")
    if _healthy(base):
        return
    port = base.rsplit(":", 1)[-1]
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "mock_api.app:app", "--host", "127.0.0.1", "--port", port, "--log-level", "warning"],
        cwd=ROOT,
    )
    atexit.register(lambda: proc.poll() is None and proc.terminate())
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Mock vendor-risk API exited with code {proc.returncode} (is port {port} taken?)")
        if _healthy(base):
            print(f"(started mock vendor-risk API on {base})")
            return
        time.sleep(0.25)
    raise RuntimeError(f"Mock vendor-risk API did not become healthy at {base}")
