"""
Main LangGraph StateGraph orchestration.

Pipeline topology:

  For QUERY input_type (plain policy question):
    START → planner → retriever → responder → END
    (No data extraction, no violation checking, no human gate needed)

  For CSV / OCR input_type (transaction/invoice data):
    START → planner → extractor → retriever → analyst → critic
                                                            │
                                 ┌──(retry, iter < 3)──────┘
                                 ↓
                              analyst
                                 │
                                 └──(verified OR max_iter)──→ actor → END

The conditional edge after 'planner' decides which path to take.
The conditional edge after 'critic' implements the verification loop.
"""
from __future__ import annotations

import uuid
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph

from .agents.actor import actor_node, register_run
from .agents.analyst import analyst_node
from .agents.critic import critic_node
from .agents.extractor import extractor_node
from .agents.planner import planner_node
from .agents.retriever import retriever_node
from .config import MAX_ITERATIONS
from .state import AgentState
from .tools.data_extraction_tool import detect_input_type


from .logger import log_step

# ── Responder node (query-only fast path) ──────────────────────────────────────

@log_step("Responder")
def responder_node(state: AgentState) -> dict:
    """
    Lightweight final node for plain policy questions.
    Formats the RAG-retrieved docs into a direct answer — no violation checking,
    no human gate, no file writes.
    """
    from .config import OLLAMA_TEMPERATURE
    from .llm import get_llm

    query = state.get("user_query", "")
    docs  = state.get("retrieved_docs", [])
    context = "\n\n".join(docs[:4]) if docs else "No policy documents retrieved."

    prompt = (
        "You are a corporate policy expert. Answer the question below using ONLY "
        "the provided policy document excerpts. Be direct and complete.\n\n"
        f"POLICY DOCUMENTS:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer:"
    )

    try:
        llm = get_llm(temperature=0.1)
        response = llm.invoke(prompt)
        answer = response.content.strip()
    except Exception as e:
        answer = f"Could not generate answer: {e}\n\nPolicy context:\n{context[:1000]}"

    return {
        "draft_answer": answer,
        "final_output": answer,
        "action_pending": False,
        "action_confirmed": True,   # Auto-confirmed for plain questions
        "policy_violations": [],
        "compliance_findings": [],
        "step_logs": [{"agent": "responder", "answer_length": len(answer)}],
    }


# ── Conditional Routing ────────────────────────────────────────────────────────

def route_after_planner(state: AgentState) -> str:
    """
    After planning: if the input is a plain policy question (no data),
    skip the data extraction and violation-checking path entirely.
    Go straight to retriever → responder.
    """
    raw_type = state.get("raw_input_type", "query")
    raw_data = (state.get("raw_input_data") or "").strip()

    # If it's explicitly a query, or there's no data at all → fast path
    if raw_type == "query" or not raw_data:
        return "retriever_query"   # Fast path
    return "extractor"             # Full compliance path


def route_after_critic(state: AgentState) -> str:
    """
    Routing for the verification loop leaving the 'critic' node.
    Returns 'analyst' to retry, or 'actor' to proceed.
    """
    is_verified    = state.get("is_verified", False)
    iteration_count = state.get("iteration_count", 0)

    if is_verified or iteration_count >= MAX_ITERATIONS:
        return "actor"
    return "analyst"


# ── Graph Builder ──────────────────────────────────────────────────────────────

def build_graph():
    """
    Construct and compile the LangGraph StateGraph.
    Two distinct paths: fast query path and full compliance path.
    """
    graph = StateGraph(AgentState)

    # ── Register nodes ──────────────────────────────────────────────────────────
    graph.add_node("planner",          planner_node)
    graph.add_node("extractor",        extractor_node)
    graph.add_node("retriever",        retriever_node)   # Full path retriever
    graph.add_node("retriever_query",  retriever_node)   # Fast path retriever (same function)
    graph.add_node("analyst",          analyst_node)
    graph.add_node("critic",           critic_node)
    graph.add_node("actor",            actor_node)
    graph.add_node("responder",        responder_node)   # Fast path final node

    # ── Entry ───────────────────────────────────────────────────────────────────
    graph.add_edge(START, "planner")

    # ── Route after planner based on input type ─────────────────────────────────
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "extractor":       "extractor",       # Full compliance path
            "retriever_query": "retriever_query",  # Fast query path
        },
    )

    # ── Full compliance path ─────────────────────────────────────────────────────
    graph.add_edge("extractor", "retriever")
    graph.add_edge("retriever", "analyst")
    graph.add_edge("analyst",   "critic")
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"analyst": "analyst", "actor": "actor"},
    )
    graph.add_edge("actor", END)

    # ── Fast query path ──────────────────────────────────────────────────────────
    graph.add_edge("retriever_query", "responder")
    graph.add_edge("responder",       END)

    return graph.compile()


# ── Singleton ──────────────────────────────────────────────────────────────────

_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        try:
            from .tools.rag_tool import build_index
            build_index()
        except FileNotFoundError:
            pass
        _compiled_graph = build_graph()
    return _compiled_graph


# ── Public Runner ──────────────────────────────────────────────────────────────

def run_pipeline(
    user_query: str,
    raw_input_data: str = "",
    raw_input_type: str = "auto",
    run_id: Optional[str] = None,
) -> AgentState:
    if not run_id:
        run_id = str(uuid.uuid4())[:8]

    if raw_input_type == "auto" and raw_input_data.strip():
        raw_input_type = detect_input_type(raw_input_data)
    elif not raw_input_data.strip():
        raw_input_type = "query"

    register_run(run_id)

    initial_state: AgentState = {
        "messages":           [],
        "user_query":         user_query,
        "raw_input_type":     raw_input_type,
        "raw_input_data":     raw_input_data,
        "plan":               [],
        "cleaned_data":       {},
        "retrieved_docs":     [],
        "draft_answer":       "",
        "compliance_findings":[],
        "policy_violations":  [],
        "critique":           "",
        "is_verified":        False,
        "iteration_count":    0,
        "action_pending":     True,
        "action_confirmed":   False,
        "final_output":       "",
        "run_id":             run_id,
        "execution_stats":    {},
        "step_logs":          [],
    }

    graph = get_compiled_graph()
    return graph.invoke(initial_state)
