"""
Actor Agent — Human-in-the-Loop action execution.

The Actor is the final node in the pipeline. It:
1. Composes a formatted compliance report from the pipeline state
2. Presents the pending action to the user for explicit confirmation
3. In CLI mode:  uses input() to prompt the user
4. In API mode:  blocks on a threading.Event set by POST /api/runs/{id}/confirm
5. If confirmed: writes report to disk + triggers the webhook
6. If denied:    logs cancellation, no files written

This dual-mode design ensures the same agent works correctly whether
invoked from main.py (CLI) or from the FastAPI dashboard (web UI).
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from typing import Any, Dict

from langchain_core.messages import AIMessage
from rich.console import Console
from rich.panel import Panel

from ..config import OUTPUT_DIR
from ..logger import log_step
from ..state import AgentState
from ..tools.action_tool import trigger_webhook, write_report

console = Console()

# ── API-Mode Confirmation Registry ────────────────────────────────────────────
# Keys: run_id strings
# Values: threading.Event (set when web UI submits /confirm)
_confirmation_events: Dict[str, threading.Event] = {}
_confirmation_decisions: Dict[str, bool] = {}
_confirmation_queues: Dict[str, Any] = {}   # SSE update queues keyed by run_id
_API_CONFIRM_TIMEOUT_S = 300  # 5 minutes


def register_run(run_id: str, update_queue=None) -> None:
    """
    Pre-register a run for API-mode human-in-the-loop gating.
    Must be called before the pipeline starts so the event exists
    when the actor node is reached.

    Args:
        run_id:       Unique run identifier.
        update_queue: Optional SSE queue — if provided, actor will push
                      action_required BEFORE blocking so modal fires correctly.
    """
    _confirmation_events[run_id] = threading.Event()
    _confirmation_decisions[run_id] = False
    if update_queue is not None:
        _confirmation_queues[run_id] = update_queue


def confirm_run(run_id: str, confirmed: bool) -> None:
    """
    Called by the FastAPI /confirm endpoint to resolve a pending gate.
    Sets the threading.Event, unblocking the actor node thread.
    """
    _confirmation_decisions[run_id] = confirmed
    if run_id in _confirmation_events:
        _confirmation_events[run_id].set()


def cleanup_run(run_id: str) -> None:
    """Remove run registration after completion to avoid memory leaks."""
    _confirmation_events.pop(run_id, None)
    _confirmation_decisions.pop(run_id, None)
    _confirmation_queues.pop(run_id, None)


# ── Report Composition ────────────────────────────────────────────────────────

def _compose_report(state: AgentState) -> str:
    """Format the complete compliance analysis report as a text document."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_id = state.get("run_id", "N/A")
    violations = state.get("policy_violations", [])
    cleaned = state.get("cleaned_data", {})
    violations_str = (
        "\n".join(f"  • {v}" for v in violations) if violations else "  No violations detected."
    )

    # Serialize cleaned data summary (truncated)
    data_summary = json.dumps(
        {k: v for k, v in cleaned.items() if k not in ("records", "cleaning_notes")},
        indent=2,
        default=str,
    )[:800]

    return (
        "=" * 80 + "\n"
        + "          ACME CORP — AUTOMATED COMPLIANCE ANALYSIS REPORT\n"
        + "=" * 80 + "\n"
        + f"Generated:    {now}\n"
        + f"Run ID:       {run_id}\n"
        + f"Query:        {state.get('user_query', 'N/A')}\n"
        + f"Input Type:   {cleaned.get('type', 'N/A')}\n"
        + "=" * 80 + "\n\n"
        + "EXECUTIVE SUMMARY\n"
        + "-" * 40 + "\n"
        + f"{state.get('draft_answer', 'No analysis available.')[:3000]}\n\n"
        + "=" * 80 + "\n"
        + "POLICY VIOLATIONS IDENTIFIED\n"
        + "-" * 40 + "\n"
        + f"{violations_str}\n\n"
        + "=" * 80 + "\n"
        + "VERIFICATION METADATA\n"
        + "-" * 40 + "\n"
        + f"Verified:     {state.get('is_verified', False)}\n"
        + f"Iterations:   {state.get('iteration_count', 0)}\n"
        + f"Critic Note:  {state.get('critique', 'N/A')[:400]}\n\n"
        + "=" * 80 + "\n"
        + "DATA SUMMARY\n"
        + "-" * 40 + "\n"
        + f"{data_summary}\n\n"
        + "=" * 80 + "\n"
        + "                          END OF REPORT\n"
        + "=" * 80
    )


# ── Agent Node ─────────────────────────────────────────────────────────────────

@log_step("Actor")
def actor_node(state: AgentState) -> Dict[str, Any]:
    """
    Node: Actor
    Human-in-the-loop gate before any file system or webhook action.
    Supports both CLI (input()) and API (threading.Event) confirmation modes.
    """
    run_id = state.get("run_id", "default")
    report_content = _compose_report(state)
    violations = state.get("policy_violations", [])
    violations_count = len(violations)

    # Display pending action summary
    console.print(Panel(
        f"[bold]Run ID:[/bold]          {run_id}\n"
        f"[bold]Violations Found:[/bold] {violations_count}\n"
        f"[bold]Verified:[/bold]        {state.get('is_verified', False)}\n"
        f"[bold]Report Size:[/bold]     {len(report_content):,} chars\n\n"
        + "[yellow]Top Violations:[/yellow]\n"
        + "\n".join(f"  • {v[:100]}" for v in violations[:3])
        + ("\n  …" if len(violations) > 3 else ""),
        title="[bold yellow]⚠  ACTION CONFIRMATION REQUIRED[/bold yellow]",
        border_style="yellow",
        expand=False,
    ))

    # ── Determine confirmation mode ───────────────────────────────────────────
    confirmed = False

    if run_id in _confirmation_events:
        # ── API Mode: push action_required to SSE queue FIRST, then block ────
        # This is the fix for the modal-never-appearing deadlock:
        # graph.stream() won't yield this node's chunk until event.wait()
        # unblocks, so we must push the modal trigger BEFORE waiting.
        q = _confirmation_queues.get(run_id)
        if q is not None:
            from datetime import datetime as _dt
            q.put({
                "type": "action_required",
                "run_id": run_id,
                "timestamp": _dt.now().isoformat(),
                "violations_count": violations_count,
                "violations_preview": violations[:5],
            })

        console.print(
            "[cyan]⏳ Waiting for web dashboard confirmation "
            f"(timeout: {_API_CONFIRM_TIMEOUT_S // 60} min)...[/cyan]"
        )
        event = _confirmation_events[run_id]
        timed_out = not event.wait(timeout=_API_CONFIRM_TIMEOUT_S)

        if timed_out:
            console.print("[red]✗ Confirmation timeout. Action cancelled.[/red]")
            confirmed = False
        else:
            confirmed = _confirmation_decisions.get(run_id, False)

    else:
        # ── CLI Mode: standard input() ────────────────────────────────────────
        try:
            answer = input(
                "\n[ACTION CONFIRMATION] Write final compliance report to output/? (yes/no): "
            ).strip().lower()
            confirmed = answer in ("yes", "y")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[red]Input interrupted. Action cancelled.[/red]")
            confirmed = False

    # ── Execute or Abort ──────────────────────────────────────────────────────
    result_meta: Dict[str, Any] = {}

    if confirmed:
        console.print("[bold green]✓ Confirmed. Executing actions...[/bold green]")

        # Write compliance report to disk
        write_result = write_report(report_content, run_id)
        result_meta["report"] = write_result
        if write_result.get("success"):
            console.print(f"[green]  ✓ Report saved:[/green] {write_result['path']}")
        else:
            console.print(f"[red]  ✗ Report write failed:[/red] {write_result.get('error')}")

        # Trigger compliance webhook (simulated AWS Lambda)
        webhook_payload = {
            "run_id": run_id,
            "user_query": state.get("user_query"),
            "total_violations": violations_count,
            "violations": violations[:10],  # Truncate for payload size
            "verified": state.get("is_verified", False),
            "report_path": write_result.get("path"),
            "timestamp": datetime.now().isoformat(),
        }
        webhook_result = trigger_webhook(webhook_payload)
        result_meta["webhook"] = webhook_result

        if webhook_result.get("simulated"):
            console.print(
                f"[dim]  ↳ Webhook: simulated (ledger_update_id={webhook_result.get('ledger_update_id')})[/dim]"
            )
        elif webhook_result.get("success"):
            console.print(f"[green]  ✓ Webhook triggered: {WEBHOOK_URL}[/green]")

        final_output = (
            f"✓ Compliance report saved to: {write_result.get('path', 'unknown')}\n"
            f"  Violations documented: {violations_count}\n"
            f"  Webhook: {webhook_result.get('message', 'triggered')}"
        )
    else:
        final_output = "✗ Action declined by user. No report written, no webhook triggered."
        console.print(f"[yellow]{final_output}[/yellow]")

    # Cleanup API event registry
    cleanup_run(run_id)

    return {
        "final_output": final_output,
        "action_confirmed": confirmed,
        "action_pending": False,
        "messages": [AIMessage(content=final_output)],
        "step_logs": [
            {
                "agent": "actor",
                "confirmed": confirmed,
                "violations_count": violations_count,
                "result": result_meta,
            }
        ],
    }


