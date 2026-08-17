"""
LangGraph AgentState definition for the Multi-Agent Compliance Assistant.

All agent nodes read from and write partial updates to this shared TypedDict.
The `Annotated` fields with `operator.add` are "append-only" — each node
appends to the list rather than replacing it, enabling full audit trails.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


def _merge_dicts(a: Dict, b: Dict) -> Dict:
    """Merge two dicts; used for execution_stats accumulation."""
    return {**a, **b}


class AgentState(TypedDict):
    """Shared pipeline state passed through every LangGraph node."""

    # ── Conversation History ──────────────────────────────────────────────────
    # Appended to by each node; never overwritten.
    messages: Annotated[List[BaseMessage], operator.add]

    # ── User Input ────────────────────────────────────────────────────────────
    user_query: str
    raw_input_type: str      # "csv" | "invoice" | "query"
    raw_input_data: str      # Raw messy CSV text or invoice text

    # ── Orchestration ─────────────────────────────────────────────────────────
    plan: List[str]          # 5-step plan from Planner agent

    # ── Data Extraction Results ───────────────────────────────────────────────
    cleaned_data: Dict[str, Any]   # Structured JSON after Pandas/Regex cleaning

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieved_docs: List[str]      # Top-K policy chunks from FAISS

    # ── Analysis ──────────────────────────────────────────────────────────────
    draft_answer: str                         # Analyst's compliance report draft
    compliance_findings: List[Dict[str, Any]] # Structured per-item findings
    policy_violations: List[str]              # Human-readable violation list

    # ── Verification ──────────────────────────────────────────────────────────
    critique: str         # Critic's JSON feedback
    is_verified: bool     # True = pass; False = retry analyst
    iteration_count: int  # Current retry count (capped at MAX_ITERATIONS)

    # ── Action Gate (Human-in-the-Loop) ───────────────────────────────────────
    action_pending: bool    # True while waiting for confirmation
    action_confirmed: bool  # Set to True when user approves

    # ── Final Output ──────────────────────────────────────────────────────────
    final_output: str

    # ── Observability ─────────────────────────────────────────────────────────
    run_id: str
    execution_stats: Annotated[Dict[str, Any], _merge_dicts]  # Per-step timings
    step_logs: Annotated[List[Dict[str, Any]], operator.add]  # Full audit trail
