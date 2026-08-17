"""
Configuration module for the Multi-Agent Compliance Assistant.

All settings are driven by environment variables (loaded from .env).
Import this module anywhere in the project to access config constants.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
load_dotenv()

# ── Base Directory ─────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).parent.parent

# ── Ollama / LLM ──────────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.3"))
PLANNER_TEMPERATURE: float = 0.1   # Very deterministic for task decomposition
CRITIC_TEMPERATURE: float = 0.1    # Strict verification

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# ── File Paths ────────────────────────────────────────────────────────────────
DATA_DIR: Path = BASE_DIR / "data"
RAW_INPUTS_DIR: Path = DATA_DIR / "raw_inputs"
DOCUMENTS_DIR: Path = DATA_DIR / "documents"
FAISS_INDEX_DIR: Path = DATA_DIR / "faiss_index"
OUTPUT_DIR: Path = BASE_DIR / "output"

# Ensure all required directories exist at import time
for _d in [OUTPUT_DIR, FAISS_INDEX_DIR, RAW_INPUTS_DIR, DOCUMENTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── RAG ───────────────────────────────────────────────────────────────────────
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "4"))
CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 64

# ── Agent Execution ───────────────────────────────────────────────────────────
MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "3"))   # Critic retry cap
LLM_TIMEOUT: int = 120                                          # seconds per LLM call

# ── FastAPI Dashboard ─────────────────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))

# ── Webhook ───────────────────────────────────────────────────────────────────
WEBHOOK_URL: str = os.getenv("WEBHOOK_URL", "http://localhost:9000/webhook")
WEBHOOK_ENABLED: bool = os.getenv("WEBHOOK_ENABLED", "false").lower() == "true"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
