"""
Analyst Agent: Synthesizes cleaned financial data against retrieved policy
sections to produce a structured compliance report with specific violation findings.

On retry loops (triggered by Critic), the agent receives the critique feedback
and is expected to revise its analysis to address the identified issues.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from rich.console import Console

from ..config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TEMPERATURE
from ..logger import log_step
from ..state import AgentState

console = Console()

_ANALYST_PROMPT = """\
You are a senior compliance analyst at a corporate finance firm.
Your task: write a formatted compliance report using the PRE-COMPUTED VIOLATION FLAGS provided.

## Iteration
Analysis pass #{iteration} of 3 maximum.
{critique_section}

## IMPORTANT: Trust the Pre-Computed Flags
The Python extraction sandbox has already computed all math-based policy checks.
The field `python_computed_violations` lists EXACTLY which violations were found.
Do NOT do your own math. Do NOT invent violations not listed. Do NOT cite amounts that are not in the data.

## Cleaned Financial Data
{data_summary}

## Retrieved Policy Sections (for citation purposes only)
{policy_context}

## Your Task
1. Use the `python_computed_violations` list as your SINGLE SOURCE OF TRUTH for violations.
2. For each violation in that list, write one row in the findings table and one bullet in Identified Violations.
3. If `is_compliant` is true and the violations list is empty, state the invoice is fully compliant. Do NOT add violations.
4. Cite the correct policy section from the Retrieved Policy Sections for each finding.
5. DO NOT hallucinate violations, amounts, or vendor names not present in the data.

## Required Output Format

**COMPLIANCE FINDINGS:**
| Item | Amount | Policy Section | Status | Reason |
|------|--------|---------------|--------|--------|
[One row per finding from python_computed_violations; if compliant write a single Compliant row]

**IDENTIFIED VIOLATIONS:**
[List each violation as: "VIOLATION: [item] - [reason] - [policy citation]"]
[If no violations, write: "None"]

**OVERALL ASSESSMENT:**
[2-3 sentence summary]

Begin your analysis:"""

_CRITIQUE_SECTION_TEMPLATE = """\
## Previous Critique (Address These Issues)
{critique}

Please revise your analysis to address the above issues."""


@log_step("Analyst")
def analyst_node(state: AgentState) -> Dict[str, Any]:
    """
    Node: Analyst
    Compares cleaned financial data against retrieved policy chunks.
    Produces a structured draft compliance report with violations table.
    Incorporates Critic feedback on revision passes.
    """
    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=OLLAMA_TEMPERATURE,
    )

    cleaned = state.get("cleaned_data", {})
    retrieved_docs = state.get("retrieved_docs", [])
    critique = state.get("critique", "")
    iteration = state.get("iteration_count", 0)

    # Format policy context
    policy_context = "\n\n---\n".join(
        f"[Policy Chunk {i + 1}]\n{doc}" for i, doc in enumerate(retrieved_docs)
    )

    # Format data summary based on type
    data_type = cleaned.get("type", "plain_query")
    if data_type == "csv_transactions":
        data_summary = json.dumps(
            {
                "total_records": cleaned.get("record_count"),
                "total_spend_usd": cleaned.get("total_absolute_spend"),
                "high_risk_transactions": cleaned.get("high_risk_transactions", []),
                "vendor_spend_summary": cleaned.get("vendor_spend_summary", {}),
            },
            indent=2,
            default=str,
        )
    elif data_type == "invoice":
        # Put the pre-computed violation flags FIRST so the LLM sees them immediately
        invoice_summary = {
            "PYTHON_COMPUTED_VIOLATIONS": cleaned.get("python_computed_violations", []),
            "is_compliant": cleaned.get("is_compliant", True),
            "invoice_number": cleaned.get("invoice_number"),
            "vendor": cleaned.get("vendor"),
            "total_amount_usd": cleaned.get("total_amount"),
            "payment_terms": cleaned.get("payment_terms"),
            "department": cleaned.get("department"),
            "exceeds_2000_threshold": cleaned.get("exceeds_2000_threshold", False),
            "exceeds_5000_threshold": cleaned.get("exceeds_5000_threshold", False),
            "payment_terms_violation": cleaned.get("payment_terms_violation", False),
            "requires_vp_authorization": cleaned.get("requires_vp_authorization", False),
        }
        data_summary = json.dumps(invoice_summary, indent=2, default=str)
    else:
        data_summary = f"User Query: {state.get('user_query', '')}"

    # Include critique on revision passes
    critique_section = ""
    if critique and iteration > 0:
        critique_section = _CRITIQUE_SECTION_TEMPLATE.format(critique=critique[:600])

    prompt = _ANALYST_PROMPT.format(
        iteration=iteration + 1,
        critique_section=critique_section,
        data_summary=data_summary[:2000],
        policy_context=policy_context[:3000],
    )

    console.print(f"[yellow]Analyst — pass {iteration + 1}[/yellow]")

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        draft = response.content.strip()
    except Exception as e:
        draft = f"Analysis error: {e}. Raw data processed. Please review manually."

    # ── Extract violations from draft output ────────────────────────────────────
    # First, try to use Python-computed violations from the extractor (ground truth).
    # These are deterministic and not subject to LLM hallucination.
    python_violations = cleaned.get("python_computed_violations", [])
    is_compliant_flag = cleaned.get("is_compliant", None)

    if python_violations is not None and data_type == "invoice":
        # For invoices: use the Python-computed list directly.
        # Filter out the COMPLIANT marker — it's not a violation.
        violations = [
            v for v in python_violations
            if not v.upper().startswith("COMPLIANT")
        ]
    else:
        # For CSV transactions: parse the LLM's draft output.
        # Skip lines that sound compliant/within-limits to avoid false positives.
        _skip_phrases = [
            "within limits", "is compliant", "no violation", "acceptable",
            "fully compliant", "does not exceed", "not a violation",
        ]
        violations = []
        for line in draft.splitlines():
            stripped = line.strip().lstrip("\u2022*- ")
            low = stripped.lower()
            if any(skip in low for skip in _skip_phrases):
                continue
            if any(kw in low for kw in ["violation:", "violates", "exceeds", "non-compliant", "flagged"]):
                if len(stripped) > 15:
                    violations.append(stripped)

    console.print(
        f"[{'red' if violations else 'green'}]"
        f"{'⚠' if violations else '✓'} "
        f"{len(violations)} violation(s) identified[/]"
    )

    return {
        "draft_answer": draft,
        "policy_violations": violations,
        "iteration_count": iteration + 1,
        "messages": [AIMessage(content=draft)],
        "step_logs": [
            {
                "agent": "analyst",
                "iteration": iteration + 1,
                "violations_found": len(violations),
                "draft_length": len(draft),
                "source": "python_computed" if (python_violations is not None and data_type == "invoice") else "llm_extracted",
            }
        ],
    }
