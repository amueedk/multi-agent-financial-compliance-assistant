"""
Pydantic request/response models for the FastAPI dashboard API.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class InputType(str, Enum):
    csv = "csv"
    invoice = "invoice"
    query = "query"
    auto = "auto"


class RunRequest(BaseModel):
    """Body for POST /api/runs — starts a new pipeline run."""

    user_query: str = Field(
        ...,
        description="The compliance question or task description",
        examples=["Check these transactions for policy violations"],
    )
    raw_input_data: str = Field(
        default="",
        description="Raw CSV ledger text or messy invoice text (empty for plain query)",
    )
    raw_input_type: InputType = Field(
        default=InputType.auto,
        description="Input data type (auto-detected if not specified)",
    )


class RunResponse(BaseModel):
    """Response for POST /api/runs."""

    run_id: str
    status: str
    message: str = "Pipeline started successfully"


class ActionConfirmation(BaseModel):
    """Body for POST /api/runs/{run_id}/confirm — human-in-the-loop gate."""

    confirmed: bool = Field(
        ..., description="True to approve action, False to cancel"
    )
    reason: Optional[str] = Field(
        default=None, description="Optional reason for confirmation/rejection"
    )


class StepLog(BaseModel):
    """A single pipeline step log entry."""

    agent: str
    timestamp: Optional[str] = None
    data: Dict[str, Any] = {}


class RunStatus(BaseModel):
    """Full run status returned by GET /api/runs/{run_id}."""

    run_id: str
    status: str  # "running" | "waiting_confirmation" | "complete" | "error"
    created_at: str
    query: str
    input_type: str
    steps: List[Dict[str, Any]] = []
    violations: List[str] = []
    final_output: Optional[str] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Response for GET /api/health."""

    status: str
    timestamp: str
    active_runs: int
    index_status: str
