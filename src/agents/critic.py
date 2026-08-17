"""
Critic Agent: Verifies the Analyst's compliance findings against retrieved
policy evidence. Returns a structured JSON verdict with specific feedback.

Routing logic (defined in graph.py):
  - is_valid=True  OR iteration_count >= MAX_ITERATIONS → proceed to Actor
  - is_valid=False AND iteration_count < MAX_ITERATIONS → retry Analyst

Max iteration guard prevents infinite CPU cycles on small local models.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from rich.console import Console

from ..config import CRITIC_TEMPERATURE, MAX_ITERATIONS, OLLAMA_BASE_URL, OLLAMA_MODEL
from ..logger import log_step
from ..state import AgentState

console = Console()

_CRITIC_PROMPT = """\
You are a strict compliance auditor. Verify whether the analyst's report is accurate.

## Retrieved Policy Evidence
{policy_context}

## Analyst's Draft Report
{draft_answer}

## Verification Checklist
1. Are all dollar amounts correct and compared to the right policy thresholds?
2. Are policy section citations accurate (check Section 4.2.1, 4.2.2)?
3. Are any obvious violations missed?
4. Is the analysis grounded in the provided policy text (no hallucinations)?
5. Are payment terms correctly assessed (Net-30 requirement)?

Respond ONLY with valid JSON in this EXACT format (no other text):
{{
  "is_valid": true,
  "confidence": 0.85,
  "feedback": "The analysis correctly identifies...",
  "missed_violations": [],
  "math_errors": []
}}

Or if issues found:
{{
  "is_valid": false,
  "confidence": 0.4,
  "feedback": "The analysis missed a violation regarding payment terms for Vendor X.",
  "missed_violations": ["Vendor X payment terms do not meet Net-30 requirement"],
  "math_errors": ["The amount for Vendor Y was compared against the wrong threshold"]
}}"""


@log_step("Critic")
def critic_node(state: AgentState) -> Dict[str, Any]:
    """
    Node: Critic
    Verifies analyst's draft against policy evidence.
    Returns is_verified=True to proceed or False to trigger retry loop.
    Max-iteration guard ensures the loop always terminates.
    """
    iteration_count = state.get("iteration_count", 1)

    # ── Max-iteration safety guard ─────────────────────────────────────────────
    if iteration_count >= MAX_ITERATIONS:
        console.print(
            f"[bold yellow]⚠  Max iterations ({MAX_ITERATIONS}) reached. "
            "Forcing verification pass-through.[/bold yellow]"
        )
        return {
            "is_verified": True,
            "critique": (
                f"Max iterations ({MAX_ITERATIONS}) reached. "
                "Proceeding with best available analysis."
            ),
            "messages": [AIMessage(content="Max iterations reached. Proceeding.")],
            "step_logs": [
                {
                    "agent": "critic",
                    "result": "max_iterations_forced",
                    "iteration": iteration_count,
                }
            ],
        }

    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=CRITIC_TEMPERATURE,
    )

    policy_context = "\n\n---\n".join(state.get("retrieved_docs", []))[:3000]
    draft_answer = state.get("draft_answer", "")[:2500]

    prompt = _CRITIC_PROMPT.format(
        policy_context=policy_context,
        draft_answer=draft_answer,
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw_response = response.content.strip()
    except Exception as e:
        console.print(f"[red]Critic LLM error: {e}. Passing through.[/red]")
        return {
            "is_verified": True,
            "critique": f"Critic error: {e}. Passing through.",
            "step_logs": [{"agent": "critic", "result": "llm_error", "error": str(e)}],
        }

    # ── Parse JSON verdict ─────────────────────────────────────────────────────
    critique_data: Dict[str, Any] = {}
    try:
        # Try extracting JSON block from response (model may add extra text)
        json_match = re.search(r"\{[\s\S]*\}", raw_response)
        if json_match:
            critique_data = json.loads(json_match.group())
        else:
            critique_data = json.loads(raw_response)
    except (json.JSONDecodeError, AttributeError):
        # Fallback: heuristic parsing
        has_true = "true" in raw_response.lower()
        has_false = "false" in raw_response.lower()
        is_valid_fallback = has_true and not has_false
        critique_data = {
            "is_valid": is_valid_fallback,
            "confidence": 0.5,
            "feedback": raw_response[:500],
            "missed_violations": [],
            "math_errors": [],
        }

    is_valid = bool(critique_data.get("is_valid", False))
    confidence = float(critique_data.get("confidence", 0.5))
    feedback = critique_data.get("feedback", "No feedback provided.")
    missed = critique_data.get("missed_violations", [])
    math_err = critique_data.get("math_errors", [])

    # Console display
    if is_valid:
        console.print(
            f"[bold green]✓ VERIFIED[/bold green] "
            f"(confidence: {confidence:.0%}, iteration: {iteration_count})"
        )
    else:
        console.print(
            f"[bold red]✗ REVISION NEEDED[/bold red] "
            f"(confidence: {confidence:.0%}, iteration: {iteration_count})"
        )
        console.print(f"[yellow]Feedback:[/yellow] {feedback[:300]}")
        if missed:
            console.print(f"[red]Missed violations:[/red] {missed}")

    return {
        "is_verified": is_valid,
        "critique": feedback,
        "messages": [AIMessage(content=f"Critic verdict: {'PASS' if is_valid else 'RETRY'}\n{feedback}")],
        "step_logs": [
            {
                "agent": "critic",
                "is_valid": is_valid,
                "confidence": confidence,
                "iteration": iteration_count,
                "missed_violations": missed,
                "math_errors": math_err,
            }
        ],
    }
