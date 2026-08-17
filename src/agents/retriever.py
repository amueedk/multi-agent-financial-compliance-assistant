"""
Retriever Agent: Queries the FAISS vector store using a context-enriched query
built from the cleaned structured data.

Strategy:
- For CSV transactions: focuses on high-risk vendor names + department codes
- For invoices:        focuses on vendor name + amount + payment terms
- For plain queries:   uses the query directly

This context-aware query construction significantly improves policy retrieval
accuracy compared to using the raw user query alone.
"""
from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import AIMessage
from rich.console import Console

from ..logger import log_step
from ..state import AgentState
from ..tools.rag_tool import retrieve_context

console = Console()


def _build_retrieval_query(state: AgentState) -> str:
    """
    Construct a semantically rich FAISS retrieval query from the pipeline state.
    Enriches the user query with key entities extracted from cleaned_data.
    """
    user_query = state.get("user_query", "")
    cleaned = state.get("cleaned_data", {})
    data_type = cleaned.get("type", "plain_query")

    if data_type == "csv_transactions":
        # Extract high-risk vendor names + department codes for focused retrieval
        high_risk = cleaned.get("high_risk_transactions", [])
        vendor_spend = cleaned.get("vendor_spend_summary", {})

        vendors = set()
        for r in high_risk:
            v = str(r.get("Vendor_Desc", "")).strip()
            if v and v != "UNKNOWN":
                # Normalize known cloud vendor names for better policy matching
                if "AWS" in v or "AMAZON" in v:
                    vendors.add("Amazon Web Services AWS")
                elif "CLOUDFLARE" in v:
                    vendors.add("Cloudflare")
                elif "DATADOG" in v:
                    vendors.add("Datadog")
                elif "SLACK" in v:
                    vendors.add("Slack")
                elif "GITHUB" in v:
                    vendors.add("GitHub")
                else:
                    vendors.add(v.title())

        total_spend = cleaned.get("total_absolute_spend", 0)
        vendor_str = " ".join(list(vendors)[:4])

        enriched = (
            f"{user_query} cloud vendor compliance policy "
            f"ENG-01 department spending limit authorization "
            f"{vendor_str} procurement policy section 4.2 "
            f"total spend {total_spend}"
        )

    elif data_type == "invoice":
        vendor = cleaned.get("vendor", "")
        amount = cleaned.get("total_amount", 0)
        terms = cleaned.get("payment_terms", "")
        dept = cleaned.get("department", "")

        enriched = (
            f"{user_query} {vendor} invoice ${amount} payment terms "
            f"{terms} compliance policy procurement {dept} "
            f"section 4.2 authorization limit Net-30"
        )

    else:
        # Plain query — use as-is
        enriched = (
            f"{user_query} compliance policy procurement "
            f"spending limit authorization vendor"
        )

    return enriched.strip()


@log_step("Retriever")
def retriever_node(state: AgentState) -> Dict[str, Any]:
    """
    Node: Retriever
    Builds context-enriched retrieval query from cleaned_data, then pulls
    top-K policy chunks from FAISS.
    """
    query = _build_retrieval_query(state)
    console.print(f"[cyan]Retrieval query:[/cyan] {query[:250]}")

    try:
        docs = retrieve_context(query)
    except FileNotFoundError as e:
        console.print(f"[red]FAISS index not found: {e}[/red]")
        docs = [
            "Policy not available. Run generate_messy_data.py and rebuild index."
        ]

    console.print(f"[green]✓ Retrieved {len(docs)} policy chunk(s)[/green]")
    for i, doc in enumerate(docs, 1):
        console.print(f"  [dim][{i}][/dim] {doc[:120].strip()}…")

    return {
        "retrieved_docs": docs,
        "messages": [
            AIMessage(content=f"Retrieved {len(docs)} relevant policy sections from FAISS.")
        ],
        "step_logs": [
            {
                "agent": "retriever",
                "query_length": len(query),
                "docs_retrieved": len(docs),
                "first_chunk_preview": docs[0][:120] if docs else "",
            }
        ],
    }
