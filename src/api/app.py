"""
FastAPI dashboard application for the Multi-Agent Compliance Assistant.

Key improvements vs v1:
  - Predictive node_started events: UI knows which node is ACTIVE immediately,
    not just after it finishes (critical for 60-90s LLM nodes)
  - step_logs are piped as individual log events so the live feed shows
    real progress (e.g. "Analyst pass 1", "3 high-risk transactions detected")
  - Heartbeat every 5s keeps the SSE connection alive during long CPU calls
  - Node sequence prediction based on state (handles critic retry routing)

Endpoints:
  GET  /                           → Serve dashboard HTML SPA
  GET  /api/health                 → Health check + FAISS index status
  POST /api/runs                   → Start a new pipeline run
  GET  /api/runs/{id}/stream       → SSE stream of real-time step updates
  POST /api/runs/{id}/confirm      → Human-in-the-loop confirmation gate
  GET  /api/runs/{id}              → Get run status and results
  GET  /api/runs                   → List all runs
  DELETE /api/runs/{id}            → Remove a run
"""
from __future__ import annotations

import json
import queue as thread_queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

from .models import ActionConfirmation, RunRequest
from ..agents.actor import confirm_run, register_run
from ..config import API_HOST, API_PORT, MAX_ITERATIONS
from ..tools.rag_tool import build_index, get_index_stats

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multi-Agent Compliance Assistant",
    description="AI-powered compliance checking: LangGraph + Ollama + FAISS",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=4)

# ── In-memory run store ────────────────────────────────────────────────────────
runs_db: Dict[str, Dict[str, Any]] = {}
update_queues: Dict[str, thread_queue.Queue] = {}

# ── Node ordering for predictive UI updates ────────────────────────────────────
_NODE_SEQUENCE_FULL  = ["planner", "extractor", "retriever", "analyst", "critic", "actor"]
_NODE_SEQUENCE_QUERY = ["planner", "retriever_query", "responder"]


def _predict_next_node(completed_node: str, state: Dict[str, Any]) -> str | None:
    """
    Predict the next node that will run after the given completed node.
    Mirrors the conditional routing logic in graph.py.
    """
    # Query fast-path
    if completed_node == "planner":
        raw_type = state.get("raw_input_type", "query")
        raw_data = (state.get("raw_input_data") or "").strip()
        if raw_type == "query" or not raw_data:
            return "retriever_query"
        return "extractor"

    if completed_node == "retriever_query":
        return "responder"

    if completed_node == "responder":
        return None  # END

    if completed_node == "critic":
        is_verified = state.get("is_verified", False)
        iteration   = state.get("iteration_count", 0)
        return "actor" if (is_verified or iteration >= MAX_ITERATIONS) else "analyst"

    try:
        idx = _NODE_SEQUENCE_FULL.index(completed_node)
        return _NODE_SEQUENCE_FULL[idx + 1] if idx < len(_NODE_SEQUENCE_FULL) - 1 else None
    except ValueError:
        return None


# ── Background pipeline thread ─────────────────────────────────────────────────

def _run_pipeline_thread(
    run_id: str,
    request: RunRequest,
    update_q: thread_queue.Queue,
) -> None:
    """
    Execute the LangGraph pipeline synchronously in a background thread.

    Event types pushed to update_q:
      node_started  → immediately when a node begins (predictive, before it finishes)
      node_log      → individual log lines from step_logs (real progress inside a node)
      step          → when a node finishes (includes full state snapshot)
      action_required → when Actor is waiting for human confirmation
      action_decision → when human has confirmed/rejected
      complete      → pipeline fully done
      error         → unhandled exception
    """
    try:
        from ..graph import build_graph
        from ..tools.data_extraction_tool import detect_input_type

        # Resolve input type
        raw_data = request.raw_input_data or ""
        input_type = request.raw_input_type.value
        if input_type == "auto" and raw_data.strip():
            input_type = detect_input_type(raw_data)
        elif not raw_data.strip():
            input_type = "query"

        register_run(run_id, update_queue=update_q)  # Pass queue for modal deadlock fix

        initial_state = {
            "messages": [],
            "user_query": request.user_query,
            "raw_input_type": input_type,
            "raw_input_data": raw_data,
            "plan": [],
            "cleaned_data": {},
            "retrieved_docs": [],
            "draft_answer": "",
            "compliance_findings": [],
            "policy_violations": [],
            "critique": "",
            "is_verified": False,
            "iteration_count": 0,
            "action_pending": True,
            "action_confirmed": False,
            "final_output": "",
            "run_id": run_id,
            "execution_stats": {},
            "step_logs": [],
        }

        graph = build_graph()
        final_state: Dict[str, Any] = {}

        # Signal that Planner is starting (before graph.stream even yields)
        update_q.put({
            "type": "node_started",
            "node": "planner",
            "timestamp": datetime.now().isoformat(),
            "message": "Planning your compliance analysis...",
        })

        for chunk in graph.stream(initial_state):
            for node_name, node_state in chunk.items():
                ts = datetime.now().isoformat()

                # Accumulate full state
                final_state.update(node_state)
                runs_db[run_id]["last_state"] = {
                    k: v for k, v in final_state.items()
                    if k not in ("messages", "raw_input_data")
                }

                # ── 1. Pipe step_logs as individual log lines ──────────────
                for log_entry in node_state.get("step_logs", []):
                    # Build a human-readable log line from the log dict
                    parts = []
                    for k, v in log_entry.items():
                        if k == "agent":
                            continue
                        if v is None or v == "" or v == [] or v == {}:
                            continue
                        if isinstance(v, list):
                            v = f"{len(v)} items"
                        if isinstance(v, float):
                            v = f"{v:.2f}"
                        parts.append(f"{k}={v}")

                    update_q.put({
                        "type": "node_log",
                        "node": node_name,
                        "timestamp": ts,
                        "message": " · ".join(parts) if parts else f"{node_name} processed",
                        "raw": log_entry,
                    })

                # ── 2. Push the completed step event ──────────────────────
                violations = node_state.get("policy_violations") or []
                update_q.put({
                    "type": "step",
                    "node": node_name,
                    "timestamp": ts,
                    "data": {
                        "plan": node_state.get("plan"),
                        "violations": violations,
                        "is_verified": node_state.get("is_verified"),
                        "iteration_count": node_state.get("iteration_count"),
                        "draft_preview": (node_state.get("draft_answer") or "")[:600],
                        "critique": node_state.get("critique", ""),
                        "final_output": node_state.get("final_output"),
                        "action_confirmed": node_state.get("action_confirmed"),
                    },
                })

                # ── 3. Predict the NEXT node and signal it's starting ──────
                next_node = _predict_next_node(node_name, final_state)
                if next_node:
                    node_messages = {
                        "extractor":       "Cleaning and structuring your raw data...",
                        "retriever":       "Searching policy documents for relevant rules...",
                        "retriever_query": "Searching policy documents for your answer...",
                        "analyst":         "Analyzing data against compliance policies...",
                        "critic":          "Verifying the analysis for accuracy...",
                        "actor":           "Preparing final report — awaiting your approval...",
                        "responder":       "Composing your policy answer...",
                    }
                    update_q.put({
                        "type": "node_started",
                        "node": next_node,
                        "timestamp": ts,
                        "message": node_messages.get(next_node, f"Running {next_node}..."),
                    })

        # ── Pipeline complete ──────────────────────────────────────────────
        runs_db[run_id]["status"] = "complete"
        runs_db[run_id]["completed_at"] = datetime.now().isoformat()
        last = runs_db[run_id].get("last_state", {})

        update_q.put({
            "type": "complete",
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "final_output": last.get("final_output", ""),
            "violations": last.get("policy_violations", []),
            "is_verified": last.get("is_verified", False),
            "action_confirmed": last.get("action_confirmed", False),
            "stats": last.get("execution_stats", {}),
        })

    except Exception as exc:
        import traceback
        err_detail = traceback.format_exc()
        runs_db[run_id]["status"] = "error"
        runs_db[run_id]["error"] = str(exc)
        update_q.put({
            "type": "error",
            "run_id": run_id,
            "message": str(exc),
            "detail": err_detail[-800:],
            "timestamp": datetime.now().isoformat(),
        })


# ── Startup ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Pre-load FAISS index on server startup (runs in thread pool)."""
    import asyncio
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(executor, build_index)
    except Exception:
        pass


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the rich single-page dashboard application."""
    html_path = Path(__file__).parent / "dashboard.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Dashboard not found.</h1>", status_code=404)


@app.get("/api/health")
async def health():
    active = sum(1 for r in runs_db.values() if r.get("status") == "running")
    idx_stats = get_index_stats()
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "active_runs": active,
        "total_runs": len(runs_db),
        "index_status": idx_stats.get("status", "unknown"),
    }


@app.post("/api/runs")
async def start_run(request: RunRequest):
    run_id = str(uuid.uuid4())[:8]
    update_q: thread_queue.Queue = thread_queue.Queue()
    update_queues[run_id] = update_q
    runs_db[run_id] = {
        "run_id": run_id,
        "status": "running",
        "created_at": datetime.now().isoformat(),
        "query": request.user_query,
        "input_type": request.raw_input_type.value,
        "last_state": {},
        "error": None,
    }
    executor.submit(_run_pipeline_thread, run_id, request, update_q)
    return {"run_id": run_id, "status": "started", "message": "Pipeline started"}


@app.get("/api/runs/{run_id}/stream")
async def stream_updates(run_id: str):
    """
    Server-Sent Events stream.
    Heartbeats every 5s keep the connection alive during long LLM calls on CPU.
    """
    if run_id not in update_queues:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    q = update_queues[run_id]

    async def generate():
        import asyncio
        loop = asyncio.get_event_loop()
        while True:
            try:
                item = await loop.run_in_executor(None, lambda: q.get(timeout=5))
                yield f"data: {json.dumps(item, default=str)}\n\n"
                if item.get("type") in ("complete", "error"):
                    break
            except thread_queue.Empty:
                # Heartbeat — keeps connection alive and lets UI show elapsed time
                yield f"data: {json.dumps({'type': 'heartbeat', 'ts': datetime.now().isoformat()})}\n\n"
            except Exception:
                break

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/runs/{run_id}/confirm")
async def confirm_action(run_id: str, body: ActionConfirmation):
    if run_id not in runs_db:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    confirm_run(run_id, body.confirmed)
    action = "confirmed" if body.confirmed else "rejected"

    if run_id in update_queues:
        update_queues[run_id].put({
            "type": "action_decision",
            "run_id": run_id,
            "confirmed": body.confirmed,
            "reason": body.reason,
            "timestamp": datetime.now().isoformat(),
        })

    return {"status": action, "run_id": run_id, "timestamp": datetime.now().isoformat()}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    if run_id not in runs_db:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    run = runs_db[run_id].copy()
    last = run.get("last_state", {})
    run["violations"] = last.get("policy_violations", [])
    run["final_output"] = last.get("final_output", "")
    run["is_verified"] = last.get("is_verified", False)
    run["stats"] = last.get("execution_stats", {})
    return run


@app.get("/api/runs")
async def list_runs():
    return sorted(
        [
            {
                "run_id": r["run_id"],
                "status": r["status"],
                "created_at": r["created_at"],
                "query": r["query"][:80],
                "input_type": r["input_type"],
                "violations": len(r.get("last_state", {}).get("policy_violations", [])),
                "error": r.get("error"),
            }
            for r in runs_db.values()
        ],
        key=lambda x: x["created_at"],
        reverse=True,
    )


@app.delete("/api/runs/{run_id}")
async def delete_run(run_id: str):
    if run_id not in runs_db:
        raise HTTPException(status_code=404, detail="Run not found")
    del runs_db[run_id]
    update_queues.pop(run_id, None)
    return {"deleted": run_id}


def start_server():
    uvicorn.run(
        "src.api.app:app",
        host=API_HOST,
        port=API_PORT,
        log_level="info",
        reload=False,
    )
