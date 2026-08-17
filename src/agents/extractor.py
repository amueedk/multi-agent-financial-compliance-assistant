"""
Data Extraction Agent — the "Python REPL" node in the pipeline.

This agent:
1. Calls extract_and_clean() — the Pandas/Regex data sandbox — to transform
   raw messy corporate data into structured JSON.
2. Asks the LLM to summarize the extraction results in natural language so
   downstream agents receive both machine-readable JSON and a human-readable
   summary.

This satisfies the "tool use" requirement: the agent equips itself with the
data_extraction_tool and executes Python code (pandas) to clean the data.
"""
from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama
from rich.console import Console
from rich.syntax import Syntax

from ..config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TEMPERATURE
from ..logger import log_step
from ..state import AgentState
from ..tools.data_extraction_tool import extract_and_clean

console = Console()

_SUMMARY_PROMPT = """\
You are a data analyst. Below is the structured JSON extracted from a raw {data_type} input.

Extracted Data:
{cleaned_json}

Write a concise 3-sentence summary for a compliance analyst covering:
1. What data was extracted and cleaned (record count or invoice details)
2. Any high-risk items found (amounts > $2,000)
3. Key vendors or transactions that may require policy review

Summary:"""


@log_step("DataExtractor")
def extractor_node(state: AgentState) -> Dict[str, Any]:
    """
    Node: DataExtractor
    Runs the Python data extraction sandbox then summarizes results via LLM.

    Input state fields used: raw_input_data, raw_input_type
    Output state fields set: cleaned_data, messages, step_logs
    """
    raw_data = state.get("raw_input_data", "").strip()
    input_type = state.get("raw_input_type", "auto")

    # ── Step 1: Run deterministic Python cleaning pipeline ────────────────────
    console.print(
        "[bold yellow]⚙  Running Python data extraction sandbox (Pandas/Regex)...[/bold yellow]"
    )

    if not raw_data:
        # No raw data — treat as plain query mode, skip extraction
        cleaned = {
            "type": "plain_query",
            "query": state.get("user_query", ""),
            "cleaning_notes": ["No raw data provided; passing through as plain query"],
        }
    else:
        cleaned = extract_and_clean(raw_data, input_type)

    # Pretty-print cleaning results
    cleaned_json_str = json.dumps(cleaned, indent=2, default=str)
    console.print(Syntax(cleaned_json_str[:1500], "json", theme="monokai", line_numbers=False))

    if cleaned.get("cleaning_notes"):
        console.print("[dim]Cleaning steps applied:[/dim]")
        for note in cleaned["cleaning_notes"]:
            console.print(f"  [green]✓[/green] {note}")

    # High-risk summary
    high_risk = cleaned.get("high_risk_transactions", [])
    if high_risk:
        console.print(
            f"[bold red]⚠  {len(high_risk)} high-risk transaction(s) detected "
            f"(|amount| > $2,000)[/bold red]"
        )

    # ── Step 2: LLM summarizes extraction for downstream agents ──────────────
    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=OLLAMA_TEMPERATURE,
    )

    summary_prompt = _SUMMARY_PROMPT.format(
        data_type=cleaned.get("type", input_type),
        cleaned_json=cleaned_json_str[:2500],
    )

    try:
        resp = llm.invoke(summary_prompt)
        summary = resp.content.strip() if hasattr(resp, "content") else str(resp)
    except Exception as e:
        summary = f"Data extraction complete. {len(high_risk)} high-risk items flagged. Error generating summary: {e}"

    console.print(f"\n[cyan]Extraction Summary:[/cyan] {summary[:200]}…")

    return {
        "cleaned_data": cleaned,
        "messages": [AIMessage(content=f"Data Extraction Summary:\n{summary}")],
        "step_logs": [
            {
                "agent": "extractor",
                "data_type": cleaned.get("type"),
                "record_count": cleaned.get("record_count", 1),
                "high_risk_count": len(high_risk),
                "cleaning_steps": len(cleaned.get("cleaning_notes", [])),
                "summary": summary[:200],
            }
        ],
    }
