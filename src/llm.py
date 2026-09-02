"""
LLM Factory — single place that decides which chat model backend to instantiate.

Every agent node calls get_llm(temperature=...) instead of constructing
ChatOllama directly. This makes the provider swappable via one env var
(LLM_PROVIDER) without touching agent code.

Providers:
  "ollama" → local model via Ollama (fully offline, default, weak on 8GB RAM)
  "gemini" → Google Gemini via API (needs GOOGLE_API_KEY, much stronger/faster)
"""
from __future__ import annotations

from .config import (
    GEMINI_MODEL,
    GOOGLE_API_KEY,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)


def get_llm(temperature: float = 0.3):
    """
    Return a LangChain chat model configured for the active LLM_PROVIDER.

    Args:
        temperature: Sampling temperature for this specific call site
                     (agents pass their own PLANNER_TEMPERATURE / CRITIC_TEMPERATURE / etc).
    """
    if LLM_PROVIDER == "gemini":
        if not GOOGLE_API_KEY:
            raise RuntimeError(
                "LLM_PROVIDER=gemini but GOOGLE_API_KEY is not set. "
                "Add it to your .env file."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=temperature,
        )

    # Default: local Ollama
    from langchain_ollama import ChatOllama

    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=temperature,
    )
