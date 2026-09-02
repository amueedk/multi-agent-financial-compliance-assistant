# Multi-Agent Compliance Research & Action Assistant

Local-first multi-agent system for corporate compliance checking. Built with LangGraph, Ollama, FAISS RAG, and a FastAPI dashboard. The default path is local Ollama on CPU, while Gemini API models can also be enabled for testing or stronger cloud-backed runs.

## Architecture

```
Raw Messy Input (CSV Ledger / Invoice Text / Plain Query)
         ↓
  ┌─────────────────────────────────────────────────────┐
  │ FastAPI Dashboard  ←→  CLI (main.py)                │
  └───────────────┬─────────────────────────────────────┘
                  ↓
           ┌─ Planner ─────────────────────────────────┐
           │  5-step plan (temp=0.1, deterministic)     │
           └───────────────────────────────────────────┘
                  ↓
           ┌─ DataExtractor ────────────────────────────┐
           │  Python/Pandas sandbox:                    │
           │  • Normalizes mixed date formats           │
           │  • Strips whitespace, fixes amounts        │
           │  • Extracts invoice fields via regex       │
           │  → Structured JSON                         │
           └───────────────────────────────────────────┘
                  ↓
           ┌─ Retriever ────────────────────────────────┐
           │  FAISS + all-MiniLM-L6-v2 (CPU, local)    │
           │  Context-enriched query from cleaned data  │
           │  → Top-K policy chunks                     │
           └───────────────────────────────────────────┘
                  ↓
           ┌─ Analyst ──────────────────────────────────┐
           │  Compares data vs. policy                  │
           │  → Violations table + draft report         │
           └───────────────────────────────────────────┘
                  ↓
           ┌─ Critic ───────────────────────────────────┐
           │  JSON verdict: {is_valid, feedback}        │
           │  Loops back to Analyst if invalid (max 3x) │
           └──────────────┬────────────────────────────┘
          (retry ≤3 loops) │  (verified OR max reached)
                           ↓
           ┌─ Actor ───────────────────────────────────┐
           │  Human-in-the-Loop Gate                   │
           │  CLI:  input()                            │
           │  API:  POST /api/runs/{id}/confirm        │
           │  → Writes report + optional webhook       │
           └───────────────────────────────────────────┘
```

## Quick Start

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running for the default local-first setup
- Pull the default testing model: `ollama pull qwen2.5:3b`
  *(Note: While `qwen2.5:3b` is used for lightweight CPU testing, you can use a larger model like `qwen2.5:7b` or `llama3.1:8b` in `.env`. Gemini API is also supported for testing by setting `LLM_PROVIDER=gemini` and a valid `GOOGLE_API_KEY`.)*

### 2. Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
```

Edit `.env` if you need a different local model, optional Gemini API config, port, or embedding settings. The default provider remains Ollama unless you explicitly switch to Gemini for a cloud-backed test run.

### 4. Generate Sample Data

Policy documents, the messy CSV ledger, and invoice text files are produced by a script and are **not** committed to the repo:

```bash
python generate_messy_data.py
```

This creates:

- `data/raw_inputs/messy_ledger.csv` — dirty ERP transaction CSV
- `data/raw_inputs/invoice_*.txt` — 8 messy invoice text files (OCR-style typos; not actual OCR)
- `data/documents/*.txt` — 22 corporate policy documents (RAG corpus)

The FAISS index is built automatically on first run and stored in `data/faiss_index/`.

### 5. Run

**CLI (interactive — query, CSV, invoice, or API):**

```bash
python main.py
```

**Dashboard only:**

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000

**Evaluation harness:**

```bash
python evaluate.py
```

## Hybrid Neuro-Symbolic Compliance Engine
To eliminate LLM hallucinations on math checks and date logic, this project utilizes a hybrid neuro-symbolic approach:
- **Deterministic Sandbox:** Developed a Pandas and Regex-based data-processing sandbox to extract structured data from typo-ridden invoices and pre-compute policy violations using deterministic math, eliminating LLM math hallucinations.
- **AI Synthesis:** The Analyst agent receives these pre-computed validation flags as a mathematical ground-truth and focuses purely on semantic understanding, policy citation, and human-readable report formatting.

## Data Pipeline

### Source A: Dirty CSV bank export

```csv
Txn_Date,Ref_ID,Vendor_Desc,Amt,Status,Notes,Dept_Code
2026-08-12,TX-9921,AWS* CLOUD SERVICES,-4500.00,PENDING,monthly hosting,ENG-01
08/14/26,,slack technologies inc,-1200,cleared,,
26-08-15,TX-9925,UBER   TRVL,-45.50,CLEARED,sales trip - client dinner,SLS-99
```

The extractor normalizes dates to ISO `YYYY-MM-DD`, collapses vendor whitespace, parses amounts, and fills missing `Ref_ID` / `Dept_Code`.

### Source B: Messy invoice text

These are plain-text invoices with typos and broken labels (the kind of noise you might see after OCR). The pipeline does not run OCR; it regex-parses the text.

```
INVOICE #9982-A
DAT: 08-10-2026
Vndor: Cloudflare, Inc.
Total Amnt Due: $ 8,450.00
Notes: Please remit payment within Net30 terms.
```

Regex extracts invoice number, date, vendor, amount, and payment terms.

## Project Structure

```
multi-agent-assistant/
├── data/
│   ├── raw_inputs/          # generated: messy CSV + invoice text
│   ├── documents/           # generated: 22 policy .txt files (RAG corpus)
│   ├── sample_csvs/         # sample transaction CSV files for dashboard testing
│   └── faiss_index/         # runtime: persisted FAISS index
├── output/                  # runtime: compliance reports
├── src/
│   ├── config.py            # environment-driven settings
│   ├── state.py             # AgentState TypedDict
│   ├── logger.py            # @log_step decorator + timing
│   ├── graph.py             # StateGraph + run_pipeline()
│   ├── tools/
│   │   ├── rag_tool.py
│   │   ├── action_tool.py
│   │   └── data_extraction_tool.py
│   ├── agents/
│   │   ├── planner.py
│   │   ├── extractor.py
│   │   ├── retriever.py
│   │   ├── analyst.py
│   │   ├── critic.py
│   │   └── actor.py
│   └── api/
│       ├── app.py
│       ├── models.py
│       └── dashboard.html
├── tests/test_cases.json
├── generate_messy_data.py
├── evaluate.py
├── main.py
├── requirements.txt
└── .env.example
```

## Dashboard API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Dashboard |
| `GET`  | `/api/health` | Health + FAISS status |
| `POST` | `/api/runs` | Start a pipeline run |
| `GET`  | `/api/runs/{id}/stream` | SSE real-time updates |
| `POST` | `/api/runs/{id}/confirm` | Human-in-the-loop gate |
| `GET`  | `/api/runs/{id}` | Run status + results |
| `GET`  | `/api/runs` | Run history |
| `GET`  | `/api/docs` | OpenAPI docs |

## Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Model name |
| `OLLAMA_TEMPERATURE` | `0.3` | LLM temperature |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embeddings |
| `RAG_TOP_K` | `4` | FAISS retrieval chunks |
| `MAX_ITERATIONS` | `3` | Critic retry cap |
| `API_PORT` | `8000` | Dashboard port |
| `WEBHOOK_ENABLED` | `false` | Enable webhook on confirm |

## Evaluation

`python evaluate.py` reports:

| Metric | Description |
|--------|-------------|
| **Groundedness** | Keyword overlap between output and expected terms |
| **Citation Rate** | Share of outputs that cite policy sections |
| **Latency** | Wall-clock seconds per pipeline run |
| **Action Gating** | Confirms no report is written without human approval |

Each agent node is wrapped with `@log_step()`, which records step name, duration, timestamp, and success/failure. A summary is printed after each pipeline run.

## Requirements

| Package | Purpose |
|---------|---------|
| `langchain`, `langchain-community` | LLM chains, document loaders |
| `langchain-ollama` | Ollama integration |
| `langgraph` | StateGraph orchestration |
| `sentence-transformers` | Local CPU embeddings |
| `faiss-cpu` | Vector store |
| `pandas` | Data extraction sandbox |
| `fastapi`, `uvicorn` | Dashboard API |
| `rich` | Terminal UI |
| `pydantic` | API models |
| `python-dotenv` | Config |
