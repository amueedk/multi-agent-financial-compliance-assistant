"""
Observability module: per-step latency tracking, rich console output,
and execution summary table.

Usage:
    from src.logger import log_step, print_execution_summary

    @log_step("MyAgent")
    def my_node(state: AgentState) -> dict:
        ...
"""
from __future__ import annotations

import functools
import time
from datetime import datetime
from typing import Any, Callable, Dict

from rich import box
from rich.console import Console
from rich.table import Table

console = Console()

# ── Step Logging Decorator ─────────────────────────────────────────────────────


def log_step(step_name: str):
    """
    Decorator that wraps a LangGraph node function with:
    - Rich console heading + completion log
    - Wall-clock timing stored in state["execution_stats"][step_name]

    The decorated function must accept and return a state dict.
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            start = time.perf_counter()
            console.rule(f"[bold cyan]▶  {step_name}[/bold cyan]")

            try:
                result = fn(state)
                elapsed = round(time.perf_counter() - start, 3)
                console.print(
                    f"[bold green]✓[/bold green]  {step_name} "
                    f"completed in [bold]{elapsed}s[/bold]"
                )

                # Merge timing into execution_stats without overwriting other keys
                timing_entry = {
                    step_name: {
                        "duration_s": elapsed,
                        "timestamp": datetime.now().isoformat(),
                        "status": "success",
                    }
                }

                # The state reducer (_merge_dicts) will merge these on return
                existing_stats = result.get("execution_stats", {})
                result["execution_stats"] = {**existing_stats, **timing_entry}
                return result

            except Exception as exc:
                elapsed = round(time.perf_counter() - start, 3)
                console.print(
                    f"[bold red]✗[/bold red]  {step_name} "
                    f"failed after [bold]{elapsed}s[/bold]: {exc}"
                )
                raise

        return wrapper

    return decorator


# ── Summary Table ──────────────────────────────────────────────────────────────


def print_execution_summary(stats: Dict[str, Any]) -> None:
    """
    Pretty-print a summary table of all agent step timings.

    Args:
        stats: The execution_stats dict from the final AgentState.
    """
    table = Table(
        title="Pipeline Execution Summary",
        box=box.ROUNDED,
        style="cyan",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Agent Step", style="bold white", min_width=18)
    table.add_column("Duration (s)", justify="right", style="green")
    table.add_column("Status", justify="center", style="yellow")
    table.add_column("Completed At", style="dim")

    total_time = 0.0
    for step, info in stats.items():
        if not isinstance(info, dict) or "duration_s" not in info:
            continue
        duration = info["duration_s"]
        total_time += duration
        status_icon = "✓" if info.get("status") == "success" else "✗"
        table.add_row(
            step,
            f"{duration:.3f}",
            status_icon,
            info.get("timestamp", "—"),
        )

    table.add_section()
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold green]{total_time:.3f}[/bold green]",
        "",
        "",
    )
    console.print(table)
