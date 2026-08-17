"""
Planner Agent: Decomposes the user query into a structured 5-step execution plan.

Design choices:
- Temperature 0.1 for highly deterministic, reproducible plans
- Provides explicit fallback plan if LLM output is malformed
- Plan is logged to step_logs for full observability
"""
from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from rich.console import Console
from rich.panel import Panel

from ..config import OLLAMA_BASE_URL, OLLAMA_MODEL, PLANNER_TEMPERATURE
from ..logger import log_step
from ..state import AgentState

console = Console()

_PLAN_PROMPT = """\
You are an AI compliance orchestrator for a corporate finance team.
Your job is to create a precise execution plan given a user's request about financial data.

User Request: {query}
Input Data Type: {input_type}

Create EXACTLY 5 steps as a numbered list. Each step should be one concise sentence.
Do not add explanations, headers, or extra text. Output ONLY the numbered list.

Example output format:
1. Extract and clean the raw {input_type} data using the data processing sandbox
2. Retrieve relevant compliance policy sections from the knowledge base
3. Analyze the cleaned data against retrieved policies for violations
4. Verify the analyst's findings for mathematical and policy accuracy
5. Generate and export the compliance report pending human confirmation
"""

_FALLBACK_PLAN = [
    "Extract and clean raw input data using the Pandas/Regex processing sandbox",
    "Retrieve relevant compliance policy sections from the FAISS knowledge base",
    "Analyze cleaned data against retrieved policies to identify violations",
    "Verify analyst findings for accuracy and completeness with the Critic",
    "Generate compliance report and execute action pending human confirmation",
]


@log_step("Planner")
def planner_node(state: AgentState) -> Dict[str, Any]:
    """
    Node: Planner
    Decomposes the user query into a 5-step execution plan.
    Uses temperature=0.1 for maximum determinism.
    """
    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=PLANNER_TEMPERATURE,
    )

    input_type = state.get("raw_input_type", "query")
    if not input_type or input_type == "auto":
        input_type = "data"

    prompt = _PLAN_PROMPT.format(
        query=state["user_query"],
        input_type=input_type,
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw_plan = response.content.strip()
    except Exception as e:
        console.print(f"[yellow]Planner LLM error: {e}. Using fallback plan.[/yellow]")
        raw_plan = "\n".join(f"{i+1}. {s}" for i, s in enumerate(_FALLBACK_PLAN))

    # Parse numbered lines into a clean list
    plan: List[str] = []
    for line in raw_plan.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading number, dot, parenthesis, dash, space
        clean = line.lstrip("0123456789.-) ").strip()
        if len(clean) > 5:
            plan.append(clean)

    # Ensure we always have a full plan
    if len(plan) < 5:
        plan = _FALLBACK_PLAN[:]

    # Log plan to console
    console.print(Panel(
        "\n".join(f"[cyan]{i+1}.[/cyan] {s}" for i, s in enumerate(plan)),
        title="[bold yellow]Execution Plan[/bold yellow]",
        border_style="yellow",
    ))

    return {
        "plan": plan,
        "messages": [AIMessage(content="Plan:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan)))],
        "step_logs": [{"agent": "planner", "plan": plan, "input_type": input_type}],
    }
