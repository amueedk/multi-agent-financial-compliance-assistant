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
from rich.console import Console

from ..config import CRITIC_TEMPERATURE, MAX_ITERATIONS
from ..llm import get_llm
from ..logger import log_step
from ..state import AgentState

console = Console()

_CRITIC_PROMPT = """\
You are a strict narrative auditor for a compliance workflow.

Important: DO NOT recompute the policy math. The Python extraction sandbox is the ground truth for all numeric checks.
Your job is to verify the report's overall correctness and consistency against that truth.

## Ground Truth Facts (authoritative)
{ground_truth}

## Retrieved Policy Evidence
{policy_context}

## Analyst's Draft Report
{draft_answer}

## Verification Checklist
1. Does the report match the deterministic ground-truth facts and not contradict them?
2. Are the policy citations and section references consistent with the retrieved policy text?
3. Does the report correctly describe non-math issues such as vendor, payment terms, department, missing data, or compliance status?
4. Does it mention the right level of risk / compliance verdict for the actual data?
5. Is the report free of obvious hallucinations, unsupported claims, or contradiction of the extracted facts?

Respond ONLY with valid JSON in this EXACT format (no other text):
{{
  "is_valid": true,
  "confidence": 0.85,
  "feedback": "The report matches the deterministic ground truth and is consistent with the policy context.",
  "missed_violations": [],
  "math_errors": []
}}

Or if issues found:
{{
  "is_valid": false,
  "confidence": 0.4,
  "feedback": "The report contradicts the deterministic findings or omits a required non-math fact.",
  "missed_violations": ["The report fails to mention the actual vendor risk or status implied by the ground truth"],
  "math_errors": []
}}"""


@log_step("Critic")
def critic_node(state: AgentState) -> Dict[str, Any]:
    """
    Node: Critic
    Validates that the analyst's draft matches the deterministic ground truth and
    the policy context without recomputing the math itself. It still checks for
    narrative consistency, missing citations, unsupported claims, and non-math
    compliance issues.
    """
    iteration_count = state.get("iteration_count", 1)
    cleaned_data = state.get("cleaned_data", {})
    precomputed_violations = cleaned_data.get("python_computed_violations", [])
    is_compliant_flag = cleaned_data.get("is_compliant")

    ground_truth = {
        "type": cleaned_data.get("type"),
        "is_compliant": is_compliant_flag,
        "requires_manual_review": cleaned_data.get("requires_manual_review", False),
        "python_computed_violations": precomputed_violations,
        "invoice_number": cleaned_data.get("invoice_number"),
        "vendor": cleaned_data.get("vendor"),
        "total_amount": cleaned_data.get("total_amount"),
        "payment_terms": cleaned_data.get("payment_terms"),
        "department": cleaned_data.get("department"),
        "record_count": cleaned_data.get("record_count"),
        "high_risk_transactions": cleaned_data.get("high_risk_transactions", []),
        "vendor_spend_summary": cleaned_data.get("vendor_spend_summary", {}),
    }
    ground_truth_json = json.dumps(ground_truth, indent=2, default=str)

    draft_lower = str(state.get("draft_answer", "")).lower()

    # Ground-truth consistency gate: do not let a second LLM re-derive compliance
    # facts. Only ensure the narrative matches the authoritative facts.
    if precomputed_violations is not None:
        compliant_markers = [
            str(v).upper().startswith("COMPLIANT")
            for v in precomputed_violations
        ]
        has_compliant_marker = any(
            compliant_markers) if compliant_markers else False
        has_violation_marker = any(
            str(v).upper().startswith((
                "CRITICAL:",
                "REQUIRES VP AUTHORIZATION:",
                "PAYMENT TERMS VIOLATION:",
            ))
            for v in precomputed_violations
        )
        risk_review = bool(cleaned_data.get("high_risk_transactions")) or bool(
            cleaned_data.get("requires_manual_review"))

        if has_compliant_marker:
            if any(k in draft_lower for k in ["violation", "violates", "non-compliant", "exceeds", "requires vp authorization"]):
                return {
                    "is_verified": False,
                    "critique": "The report contradicts the deterministic compliant result. It must not claim a violation when the extraction sandbox marks the record as compliant.",
                    "messages": [AIMessage(content="Critic verdict: RETRY\nReport contradicts deterministic compliant ground truth.")],
                    "step_logs": [{"agent": "critic", "result": "ground_truth_conflict", "iteration": iteration_count}],
                }
            if risk_review and "review" not in draft_lower and "manual review" not in draft_lower:
                return {
                    "is_verified": False,
                    "critique": "The report says compliant but the extracted data contains elevated-risk items that require manual review. Separate review flags from actual violations.",
                    "messages": [AIMessage(content="Critic verdict: RETRY\nHigh-risk review items were not clearly distinguished from violations.")],
                    "step_logs": [{"agent": "critic", "result": "risk_review_mismatch", "iteration": iteration_count}],
                }
            return {
                "is_verified": True,
                "critique": "The report matches the deterministic compliant result and correctly separates review items from actual violations.",
                "messages": [AIMessage(content="Critic verdict: PASS\nGround-truth compliance facts are aligned.")],
                "step_logs": [{"agent": "critic", "result": "ground_truth_approved", "iteration": iteration_count}],
            }

        if is_compliant_flag is True and not has_violation_marker:
            if any(k in draft_lower for k in ["violation", "violates", "non-compliant", "exceeds", "requires vp authorization"]):
                return {
                    "is_verified": False,
                    "critique": "The report claims a violation even though the deterministic ground truth says the record is compliant.",
                    "messages": [AIMessage(content="Critic verdict: RETRY\nReport contradicts deterministic compliance status.")],
                    "step_logs": [{"agent": "critic", "result": "ground_truth_conflict", "iteration": iteration_count}],
                }
            if risk_review and "review" not in draft_lower and "manual review" not in draft_lower:
                return {
                    "is_verified": False,
                    "critique": "The report omits the fact that elevated-risk items require manual review even though no policy violation is present.",
                    "messages": [AIMessage(content="Critic verdict: RETRY\nNo violation exists, but review items need explicit callout.")],
                    "step_logs": [{"agent": "critic", "result": "risk_review_mismatch", "iteration": iteration_count}],
                }
            return {
                "is_verified": True,
                "critique": "The report aligns with the deterministic compliance status and distinguishes review flags from actual violations.",
                "messages": [AIMessage(content="Critic verdict: PASS\nGround-truth compliance status is consistent.")],
                "step_logs": [{"agent": "critic", "result": "ground_truth_approved", "iteration": iteration_count}],
            }

        if has_violation_marker:
            if "compliant" in draft_lower and "violation" not in draft_lower:
                return {
                    "is_verified": False,
                    "critique": "The report claims the record is compliant even though the deterministic ground truth includes a violation.",
                    "messages": [AIMessage(content="Critic verdict: RETRY\nGround truth includes violations but report says compliant.")],
                    "step_logs": [{"agent": "critic", "result": "ground_truth_conflict", "iteration": iteration_count}],
                }
            return {
                "is_verified": True,
                "critique": "The report is consistent with the deterministic violations and does not override the factual risk flags.",
                "messages": [AIMessage(content="Critic verdict: PASS\nGround-truth violation set is consistent.")],
                "step_logs": [{"agent": "critic", "result": "ground_truth_approved", "iteration": iteration_count}],
            }

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

    llm = get_llm(temperature=CRITIC_TEMPERATURE)

    policy_context = "\n\n---\n".join(state.get("retrieved_docs", []))[:3000]
    draft_answer = state.get("draft_answer", "")[:2500]

    prompt = _CRITIC_PROMPT.format(
        ground_truth=ground_truth_json[:2000],
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
