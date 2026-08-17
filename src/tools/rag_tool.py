"""
RAG Tool: FAISS-backed document retrieval with local HuggingFace embeddings.

Pipeline:
  1. Load all .txt and .md files from data/documents/
  2. Split into overlapping chunks (RecursiveCharacterTextSplitter)
  3. Embed with all-MiniLM-L6-v2 (CPU, ~80MB, no API key required)
  4. Build and persist a FAISS index to data/faiss_index/
  5. Provide retrieve_context(query, k) for agent use

The index is rebuilt automatically if missing; reloaded from disk otherwise.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENTS_DIR,
    EMBEDDING_MODEL,
    FAISS_INDEX_DIR,
    RAG_TOP_K,
)

# ── Module-level singletons (lazy-initialized) ────────────────────────────────
_embeddings: Optional[HuggingFaceEmbeddings] = None
_vector_store: Optional[FAISS] = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Return cached HuggingFace embeddings model (CPU, local)."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def build_index(force_rebuild: bool = False) -> FAISS:
    """
    Build (or reload) the FAISS vector store from documents/ directory.

    Args:
        force_rebuild: If True, discard any cached index and rebuild from scratch.

    Returns:
        A loaded FAISS vector store ready for similarity search.

    Raises:
        FileNotFoundError: If no documents are found in DOCUMENTS_DIR.
    """
    global _vector_store
    index_path = str(FAISS_INDEX_DIR)
    faiss_file = FAISS_INDEX_DIR / "index.faiss"
    embeddings = _get_embeddings()

    # Try loading persisted index from disk
    if not force_rebuild and faiss_file.exists():
        _vector_store = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return _vector_store

    # Load all .txt and .md documents
    documents = []
    for ext in ("*.txt", "*.md"):
        for fpath in DOCUMENTS_DIR.glob(ext):
            try:
                loader = TextLoader(str(fpath), encoding="utf-8")
                documents.extend(loader.load())
            except Exception:
                pass  # Skip unreadable files silently

    if not documents:
        raise FileNotFoundError(
            f"No documents found in {DOCUMENTS_DIR}. "
            "Run: python generate_messy_data.py"
        )

    # Chunk documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Build and persist FAISS index
    _vector_store = FAISS.from_documents(chunks, embeddings)
    _vector_store.save_local(index_path)
    return _vector_store


def retrieve_context(query: str, k: Optional[int] = None) -> List[str]:
    """
    Retrieve the top-k most semantically relevant document chunks.

    Args:
        query: Natural language retrieval query (constructed from cleaned data).
        k:     Number of chunks to retrieve (defaults to RAG_TOP_K from config).

    Returns:
        List of page_content strings (raw policy text chunks).
    """
    global _vector_store
    k = k or RAG_TOP_K

    if _vector_store is None:
        _vector_store = build_index()

    results = _vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in results]


def get_index_stats() -> dict:
    """Return basic stats about the loaded index (for dashboard health checks)."""
    global _vector_store
    if _vector_store is None:
        try:
            _vector_store = build_index()
        except FileNotFoundError:
            return {"status": "not_built", "doc_count": 0}

    return {
        "status": "ready",
        "index_path": str(FAISS_INDEX_DIR),
        "embedding_model": EMBEDDING_MODEL,
    }
