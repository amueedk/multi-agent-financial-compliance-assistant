"""
Interactive CLI entrypoint for the Multi-Agent Compliance Assistant.

Modes:
  query    → Plain-text compliance question (no raw data)
  csv      → Load a CSV file from data/raw_inputs/ for analysis
  invoice  → Load a messy invoice text file from data/raw_inputs/ for analysis
  api      → Start the FastAPI dashboard server (http://localhost:8000)

Usage:
  python main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()

BANNER = """
╔══════════════════════════════════════════════════════════════════════╗
║        Multi-Agent Compliance Research & Action Assistant           ║
║      LangGraph + Ollama (CPU) + FAISS RAG + FastAPI Dashboard      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

MODES = {
    "query":   "Plain-text compliance question",
    "csv":     "Analyze messy CSV ledger from data/raw_inputs/",
    "invoice": "Analyze messy invoice text from data/raw_inputs/",
    "api":     "Start FastAPI dashboard (http://localhost:8000)",
}


def _check_prerequisites() -> bool:
    """Warn if data files or Ollama are not set up."""
    from src.config import DOCUMENTS_DIR, RAW_INPUTS_DIR

    ok = True
    if not any(DOCUMENTS_DIR.glob("*.txt")):
        console.print(
            "[yellow]⚠  No policy documents found in data/documents/\n"
            "   Run:  python generate_messy_data.py[/yellow]"
        )
        ok = False
    return ok


def run_query_mode() -> tuple[str, str, str]:
    user_query = Prompt.ask("[bold]Enter your compliance question[/bold]")
    return user_query, "", "query"


def run_csv_mode() -> tuple[str, str, str]:
    from src.config import RAW_INPUTS_DIR

    csv_files = sorted(RAW_INPUTS_DIR.glob("*.csv"))
    if not csv_files:
        console.print(
            "[red]No CSV files found in data/raw_inputs/\n"
            "Run: python generate_messy_data.py[/red]"
        )
        sys.exit(1)

    console.print("[bold]Available CSV files:[/bold]")
    for i, f in enumerate(csv_files, 1):
        size = f.stat().st_size
        console.print(f"  [[cyan]{i}[/cyan]] {f.name}  [dim]({size:,} bytes)[/dim]")

    idx = max(0, min(len(csv_files) - 1, int(Prompt.ask("Select file number", default="1")) - 1))
    selected = csv_files[idx]
    raw_data = selected.read_text(encoding="utf-8")

    user_query = Prompt.ask(
        "[bold]Compliance question[/bold]",
        default="Analyze these transactions for policy violations",
    )
    console.print(f"[green]Loaded:[/green] {selected.name} ({len(raw_data):,} chars)")
    return user_query, raw_data, "csv"


def run_invoice_mode() -> tuple[str, str, str]:
    from src.config import RAW_INPUTS_DIR

    invoice_files = sorted(RAW_INPUTS_DIR.glob("*.txt"))
    if not invoice_files:
        console.print(
            "[red]No invoice text files found in data/raw_inputs/\n"
            "Run: python generate_messy_data.py[/red]"
        )
        sys.exit(1)

    console.print("[bold]Available invoice files:[/bold]")
    for i, f in enumerate(invoice_files, 1):
        console.print(f"  [[cyan]{i}[/cyan]] {f.name}")

    idx = max(0, min(len(invoice_files) - 1, int(Prompt.ask("Select file number", default="1")) - 1))
    selected = invoice_files[idx]
    raw_data = selected.read_text(encoding="utf-8")

    user_query = Prompt.ask(
        "[bold]Compliance question[/bold]",
        default="Check this invoice for compliance with our procurement policy",
    )
    console.print(f"[green]Loaded:[/green] {selected.name} ({len(raw_data):,} chars)")
    return user_query, raw_data, "invoice"


def run_api_mode() -> None:
    console.print(
        Panel(
            "[bold green]Starting FastAPI Dashboard Server[/bold green]\n\n"
            "  URL:   [cyan]http://localhost:8000[/cyan]\n"
            "  Docs:  [cyan]http://localhost:8000/api/docs[/cyan]\n\n"
            "Press [bold]Ctrl+C[/bold] to stop.",
            border_style="green",
        )
    )
    from src.api.app import start_server
    start_server()


def main() -> None:
    console.print(f"[bold cyan]{BANNER}[/bold cyan]")
    _check_prerequisites()

    console.print("[bold]Available modes:[/bold]")
    for key, desc in MODES.items():
        console.print(f"  [cyan]{key:7}[/cyan] {desc}")
    console.print()

    mode = Prompt.ask("Select mode", choices=list(MODES.keys()), default="csv")

    if mode == "api":
        run_api_mode()
        return

    # Get inputs
    if mode == "query":
        user_query, raw_data, input_type = run_query_mode()
    elif mode == "csv":
        user_query, raw_data, input_type = run_csv_mode()
    elif mode == "invoice":
        user_query, raw_data, input_type = run_invoice_mode()
    else:
        console.print("[red]Unknown mode[/red]")
        sys.exit(1)

    # Display run summary
    console.print(Panel(
        f"[bold]Query:[/bold]       {user_query}\n"
        f"[bold]Input Type:[/bold]  {input_type}\n"
        f"[bold]Data Size:[/bold]   {len(raw_data):,} chars",
        title="[bold cyan]Starting Pipeline[/bold cyan]",
        border_style="cyan",
    ))

    # Run the pipeline
    from src.graph import run_pipeline
    from src.logger import print_execution_summary

    try:
        result = run_pipeline(
            user_query=user_query,
            raw_input_data=raw_data,
            raw_input_type=input_type,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user.[/yellow]")
        return

    # Print execution summary
    console.rule("[bold cyan]Pipeline Complete[/bold cyan]")
    print_execution_summary(result.get("execution_stats", {}))

    # Violations summary
    violations = result.get("policy_violations", [])
    if violations:
        console.print(Panel(
            "\n".join(f"  • {v}" for v in violations),
            title=f"[bold red]⚠  {len(violations)} Policy Violation(s) Detected[/bold red]",
            border_style="red",
        ))
    else:
        console.print(Panel("[green]No policy violations detected.[/green]", border_style="green"))

    # Final output
    console.print(Panel(
        result.get("final_output", "No output generated."),
        title="[bold green]Final Output[/bold green]",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
