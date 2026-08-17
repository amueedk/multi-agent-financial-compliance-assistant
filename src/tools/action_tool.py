"""
Action Tool: writes compliance reports to disk and triggers the simulated
AWS serverless webhook with the violation payload.

Both functions are called by the Actor agent only after human confirmation.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict

from ..config import OUTPUT_DIR, WEBHOOK_ENABLED, WEBHOOK_URL


def write_report(content: str, run_id: str) -> Dict[str, Any]:
    """
    Persist the final compliance report as a timestamped .txt file.

    Args:
        content: Full report text.
        run_id:  Unique run identifier used in the filename.

    Returns:
        Dict with success flag, file path, size, and timestamp.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = OUTPUT_DIR / f"compliance_report_{run_id}_{timestamp}.txt"

    try:
        filepath.write_text(content, encoding="utf-8")
        return {
            "success": True,
            "path": str(filepath),
            "filename": filepath.name,
            "size_bytes": len(content.encode("utf-8")),
            "timestamp": timestamp,
        }
    except OSError as e:
        return {"success": False, "error": str(e)}


def trigger_webhook(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Trigger the compliance update webhook (simulated AWS Lambda endpoint).

    When WEBHOOK_ENABLED=false (default), returns a simulated successful
    response so the pipeline can be tested without a live endpoint.

    When WEBHOOK_ENABLED=true, sends a real POST request to WEBHOOK_URL.

    Args:
        payload: Compliance summary dict (violations, run_id, timestamps).

    Returns:
        Dict with success flag and response details.
    """
    if not WEBHOOK_ENABLED:
        return {
            "success": True,
            "simulated": True,
            "endpoint": WEBHOOK_URL,
            "payload_keys": list(payload.keys()),
            "ledger_update_id": f"LU-{int(time.time())}",
            "message": (
                "Webhook simulated (set WEBHOOK_ENABLED=true to call live endpoint)"
            ),
            "timestamp": datetime.now().isoformat(),
        }

    try:
        import httpx

        response = httpx.post(WEBHOOK_URL, json=payload, timeout=10)
        return {
            "success": response.status_code < 400,
            "simulated": False,
            "status_code": response.status_code,
            "endpoint": WEBHOOK_URL,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as exc:
        return {
            "success": False,
            "simulated": False,
            "error": str(exc),
            "endpoint": WEBHOOK_URL,
        }
