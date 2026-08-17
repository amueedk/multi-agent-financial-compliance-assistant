"""
Quantitative evaluation harness for the Multi-Agent Compliance Assistant.

Runs 5 automated test cases from tests/test_cases.json and reports:
  1. Groundedness Score — keyword overlap between output and expected keywords
  2. Latency            — wall-clock seconds per pipeline run
  3. Action Gating      — verify no action executes without approval (auto-denied)

Usage:
  python evaluate.py

Output: Rich formatted ASCII table with per-test and aggregate metrics.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
TEST_CASES_PATH = Path("tests/test_cases.json")


# ── Metric Helpers ─────────────────────────────────────────────────────────────

def keyword_overlap(text: str, keywords: List[str]) -> float:
    """Fraction of expected keywords present (case-insensitive) in text."""
    if not keywords:
        return 1.0
    text_lower = text.lower()
    matched = sum(1 for kw in keywords if kw.lower() in text_lower)
    return matched / len(keywords)


def has_policy_citation(text: str) -> bool:
    """Check if text contains recognizable policy citation markers."""
    markers = [
        "section 4.2", "4.2.1", "4.2.2", "policy", "violation",
        "net-30", "$5,000", "$2,000", "eng-01", "vp of engineering",
    ]
    lower = text.lower()
    return any(m.lower() in lower for m in markers)


# ── Auto-Confirm Helper ────────────────────────────────────────────────────────

def _auto_deny_thread(run_id: str, delay: float = 1.0) -> None:
    """
    Background thread that automatically DENIES the actor confirmation gate
    after a short delay. Prevents evaluation from writing files to disk
    or blocking on user input.
    """
    time.sleep(delay)
    try:
        from src.agents.actor import confirm_run
        confirm_run(run_id, confirmed=False)
    except Exception:
        pass


# ── Pipeline Runner ────────────────────────────────────────────────────────────

def run_test_case(tc: Dict[str, Any], test_idx: int) -> Dict[str, Any]:
    """Execute a single test case through the pipeline and collect metrics."""
    from src.agents.actor import register_run
    from src.graph import run_pipeline

    run_id = f"eval_{test_idx:02d}"

    # Register and pre-set auto-deny so actor doesn't block
    register_run(run_id)
    deny_thread = threading.Thread(
        target=_auto_deny_thread, args=(run_id, 0.5), daemon=True
    )
    deny_thread.start()

    t0 = time.perf_counter()
    error: str | None = None
    final_state: Dict[str, Any] = {}

    try:
        final_state = run_pipeline(
            user_query=tc["query"],
            raw_input_data=tc.get("raw_data", ""),
            raw_input_type=tc.get("input_type", "auto"),
            run_id=run_id,
        )
    except Exception as exc:
        error = str(exc)

    elapsed = time.perf_counter() - t0

    # Combine draft + final output for scoring
    output_text = " ".join([
        final_state.get("draft_answer", ""),
        final_state.get("final_output", ""),
        " ".join(final_state.get("policy_violations", [])),
    ])

    keywords = tc.get("expected_keywords", [])
    g_score = keyword_overlap(output_text, keywords)
    citation = has_policy_citation(output_text)

    # Action gating: confirm action_confirmed is False (gate held)
    action_confirmed = final_state.get("action_confirmed", False)
    expect_no_action = tc.get("expect_no_action", True)
    gating_ok = not action_confirmed if expect_no_action else True

    return {
        "name": tc["name"],
        "run_id": run_id,
        "latency_s": elapsed,
        "groundedness": g_score,
        "citation_present": citation,
        "action_gating_ok": gating_ok,
        "violations_found": len(final_state.get("policy_violations", [])),
        "iterations": final_state.get("iteration_count", 0),
        "is_verified": final_state.get("is_verified", False),
        "error": error,
    }


# ── Results Table ──────────────────────────────────────────────────────────────

def print_results(results: List[Dict[str, Any]]) -> None:
    """Render the evaluation results as a formatted Rich table."""
    table = Table(
        title="📊  Multi-Agent Evaluation Results",
        box=box.ROUNDED,
        style="cyan",
        header_style="bold magenta",
        show_footer=True,
    )

    table.add_column("Test Case",        style="bold white",    min_width=24)
    table.add_column("Latency (s)",      justify="right",       style="yellow",       footer="AVG")
    table.add_column("Groundedness",     justify="right",       style="green",        footer="AVG")
    table.add_column("Citation",         justify="center")
    table.add_column("Action Gated",     justify="center")
    table.add_column("Violations",       justify="center",      style="dim")
    table.add_column("Verified",         justify="center",      style="dim")
    table.add_column("Status",           justify="center")

    for r in results:
        status = "[red]ERROR[/red]" if r["error"] else "[green]PASS[/green]"
        table.add_row(
            r["name"],
            f"{r['latency_s']:.1f}",
            f"{r['groundedness']:.0%}",
            "✓" if r["citation_present"] else "✗",
            "[green]✓[/green]" if r["action_gating_ok"] else "[red]✗[/red]",
            str(r["violations_found"]),
            "✓" if r["is_verified"] else "✗",
            status,
        )

    console.print(table)

    # Aggregate metrics
    n = len(results)
    avg_lat = sum(r["latency_s"] for r in results) / n
    avg_gnd = sum(r["groundedness"] for r in results) / n
    pct_cite = sum(r["citation_present"] for r in results) / n
    pct_gate = sum(r["action_gating_ok"] for r in results) / n
    errors   = sum(1 for r in results if r["error"])

    console.print(Panel(
        f"  [bold]Average Latency:[/bold]      [yellow]{avg_lat:.1f}s[/yellow]\n"
        f"  [bold]Average Groundedness:[/bold] [green]{avg_gnd:.0%}[/green]\n"
        f"  [bold]Citation Rate:[/bold]        [cyan]{pct_cite:.0%}[/cyan]\n"
        f"  [bold]Action Gating:[/bold]        {'[green]100%[/green]' if pct_gate == 1.0 else f'[red]{pct_gate:.0%}[/red]'}\n"
        f"  [bold]Errors:[/bold]               {'[green]0[/green]' if errors == 0 else f'[red]{errors}[/red]'}",
        title="[bold cyan]Aggregate Metrics[/bold cyan]",
        border_style="cyan",
    ))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    console.rule("[bold cyan]Multi-Agent Compliance Evaluation Harness[/bold cyan]")

    if not TEST_CASES_PATH.exists():
        console.print(f"[red]Test cases not found: {TEST_CASES_PATH}[/red]")
        sys.exit(1)

    with open(TEST_CASES_PATH) as f:
        test_cases = json.load(f)

    console.print(f"[bold]Running {len(test_cases)} test cases...[/bold]\n")
    results: List[Dict[str, Any]] = []

    for i, tc in enumerate(test_cases, 1):
        console.print(
            f"[bold yellow]▶ Test {i}/{len(test_cases)}:[/bold yellow] {tc['name']}"
        )
        console.print(f"  [dim]Query: {tc['query'][:90]}[/dim]")
        result = run_test_case(tc, i)
        results.append(result)

        status = "[red]✗ ERROR[/red]" if result["error"] else "[green]✓ OK[/green]"
        console.print(
            f"  {status} | "
            f"Latency: [yellow]{result['latency_s']:.1f}s[/yellow] | "
            f"Groundedness: [green]{result['groundedness']:.0%}[/green] | "
            f"Violations: {result['violations_found']}"
        )
        if result["error"]:
            console.print(f"  [red]Error: {result['error'][:120]}[/red]")
        console.print()

    print_results(results)


if __name__ == "__main__":
    main()
