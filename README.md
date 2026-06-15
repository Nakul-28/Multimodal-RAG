# Multimodal RAG Pipeline

A fully **local**, full-stack Multimodal RAG system — **Next.js** frontend, **FastAPI** backend, **LangChain**, **ChromaDB**, and **Ollama**. Upload documents (PDF, DOCX, PPTX, and more), watch them process in real-time, and chat with your documents with no cloud API dependency.

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.com) running locally

### 1. Pull Ollama models

```bash
ollama pull qwen2.5:7b              # LLM for chat answers and query rewriting
ollama pull nomic-embed-text-v2-moe # Embedding model (MoE)
ollama pull qwen2.5vl:3b            # Vision model for image/table summarization
```

> Models are configurable via `.env` — see [Configuration](#configuration).

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the backend

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### 4. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

### 5. Open the app

Navigate to **http://localhost:3000**

- **Left sidebar** — drag-and-drop files, watch real-time pipeline stages (Parsing → Chunking → Summarizing → Embedding → Done)
- **Main area** — chat with your documents; answers stream token-by-token, images display inline

> **Tip:** Run both servers at once with `start.bat` (Windows).

---

## Supported File Types

| Category | Extensions |
|----------|-----------|
| Documents | `.pdf`, `.docx`, `.doc`, `.odt`, `.rtf` |
| Presentations | `.pptx`, `.ppt` |
| Spreadsheets | `.xlsx`, `.xls`, `.csv`, `.tsv` |
| Text | `.txt`, `.md`, `.rst` |
| Web | `.html`, `.htm`, `.xml` |
| Data | `.json` |

PDFs receive enhanced processing: high-resolution parsing, table structure inference, and embedded image extraction. Other formats use Unstructured's auto-detection.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INGESTION PIPELINE                       │
└─────────────────────────────────────────────────────────────────┘

Document Upload (PDF, DOCX, PPTX, ...)
   │
   ▼
[1] PARSING — unstructured partition()                ← stage: "parsing"
   │  PDFs: hi_res strategy, infer_table_structure,
   │        extract_image_block_to_payload
   │  Others: auto-detected format parsing
   ▼
[2] CHUNKING — chunk_by_title()                      ← stage: "chunking"
   │  max 3000 chars, combine under 500, noise filtering
   ▼
[3] AI SUMMARIZATION — Qwen2.5-VL 3B (Vision)       ← stage: "summarizing"
   │  Chunks containing tables/images → LLM generates
   │  a searchable description covering key facts,
   │  concepts, and alternative search terms
   ▼
[4] EMBEDDING & INDEXING                             ← stage: "embedding"
   │  ChromaDB — nomic-embed-text-v2-moe (MoE model)
   │  BM25 index — updated with every new document
   ▼
   Persisted to ./chroma_db/

┌─────────────────────────────────────────────────────────────────┐
│                          QUERY PIPELINE                          │
└─────────────────────────────────────────────────────────────────┘

User Query + Chat History
   │
   ▼
[1] QUERY REWRITING (if history exists)
   │  LLM resolves pronouns and references into a
   │  standalone search query; preserves technical terms,
   │  equations, model names, and section numbers
   ▼
[2] HYBRID RETRIEVAL
   │  Dense: ChromaDB MMR (lambda=0.75, fetch_k = k×4)
   │  Sparse: BM25 lexical retrieval
   │  → deduplicated union of candidates
   ▼
[3] CROSS-ENCODER RERANKING
   │  sentence-transformers CrossEncoder re-scores
   │  all candidates; top-k selected
   ▼
[4] MULTIMODAL ANSWER GENERATION
   │  System prompt: 10-rule grounded answering policy
   │  Context: raw text + HTML tables + base64 images
   │  LLM (8192 token context window), temperature=0
   ▼
SSE token stream → Next.js frontend (typewriter effect)
Inline images served via /images/{id}
```

---

## Project Structure

```
├── backend/
│   ├── main.py             # FastAPI app — endpoints, WebSocket, SSE, chat history
│   ├── rag_engine.py       # Full RAG pipeline — ingest, retrieve, rerank, generate
│   ├── requirements.txt    # Python dependencies
│   ├── .env                # Model & server configuration
│   ├── chroma_db/          # Persisted ChromaDB vector store
│   ├── uploads/            # Uploaded files
│   └── __init__.py
├── evaluation/
│   ├── eval_rag_system.py  # RAGAS evaluation runner (local judge LLM)
│   ├── rag_eval_adapter.py # Bridges query_rag() to the evaluation harness
│   ├── rag_benchmark_50q.json  # 50-question benchmark with reference answers
│   ├── rag_outputs.json    # Cached RAG outputs (auto-generated)
│   └── evaluation_results.json # Final RAGAS scores (auto-generated)
├── frontend/
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx
│   │   └── globals.css
│   ├── components/
│   │   ├── ChatWindow.tsx      # Streaming chat + markdown + inline images
│   │   ├── UploadPanel.tsx     # Drag-drop upload + WebSocket status
│   │   └── PipelineStatus.tsx  # Step-by-step pipeline indicator
│   ├── lib/
│   │   ├── api.ts
│   │   └── types.ts
│   └── .env.local
└── start.bat
```

---

## Features

### Ingestion
- **Multi-format parsing** via Unstructured — PDFs use hi-res strategy with table structure inference and image extraction; other formats use auto-detection
- **AI-enhanced chunk summaries** — Qwen2.5-VL 3B generates searchable descriptions for any chunk containing tables or images, improving retrieval precision for multimodal content
- **Noise filtering** — low-value chunks (page numbers, reference stubs, copyright footers) are dropped before embedding
- **Real-time status** — WebSocket pushes stage-by-stage progress to the frontend

### Retrieval
- **Hybrid BM25 + dense retrieval** — lexical BM25 and MMR-based dense retrieval run in parallel; results are deduplicated and merged
- **Cross-encoder reranking** — a sentence-transformers CrossEncoder re-scores all candidates; top-k are passed to the LLM
- **Conversational query rewriting** — on follow-up questions, an LLM rewrites the query into a standalone search string, preserving technical terms, equations, and section references
- **MMR diversity** — `lambda_mult=0.75` balances relevance and coverage

### Generation
- **Strictly grounded answers** — 10-rule system prompt forbids the LLM from using prior knowledge, hallucinating facts, or fabricating missing information
- **Multimodal context** — raw text, HTML tables, and base64-encoded images all passed in a single multimodal prompt
- **SSE streaming** — token-by-token response via Server-Sent Events; first event delivers metadata (chunk count, image IDs)
- **Chat history** — server-side conversation memory with optimistic concurrency (version-locked appends)

### Frontend
- Token-by-token typewriter streaming
- Full markdown rendering (tables, code blocks, lists)
- Inline image display from extracted PDF figures
- Drag-and-drop multi-file upload with per-file progress

---

## Evaluation

The system was benchmarked using **RAGAS** on a hand-crafted 50-question dataset, with **Qwen2.5 7B as the local judge LLM** (no OpenAI / cloud APIs).

| Metric | Score |
|--------|-------|
| Faithfulness | **0.897** |
| Context Precision | **0.835** |
| Context Recall | **0.842** |
| Answer Relevancy | **0.826** |
| Answer Similarity | **0.827** |
| Answer Correctness | **0.760** |
| Hallucination Rate | 10.3% |
| Avg Generation Latency | 23.2s |

> All inference is fully local — generation model, vision model, embedding model, and judge LLM run entirely on-device via Ollama. Answer correctness and hallucination rate reflect the ceiling imposed by a 3B-parameter generation model rather than retrieval quality.

### Running the Evaluation

```bash
# From the evaluation/ directory
python eval_rag_system.py
```

Results are cached to `rag_outputs.json` after generation and to `evaluation_results.json` after scoring. Re-runs skip generation and go straight to scoring unless the cache is deleted.

---

## Configuration

All model and server settings live in `.env`:

```env
# ─────────────────────────────────────────────
# Ollama Models
# ─────────────────────────────────────────────

# LLM for chat answers and query rewriting
LLM_MODEL=qwen2.5:7b

# MoE embedding model for vector store
EMBEDDING_MODEL=nomic-embed-text-v2-moe:latest

# Vision model for image/table chunk summarization at ingestion time
VISION_MODEL=qwen2.5vl:3b

# HuggingFace cross-encoder for retrieval reranking
RERANKER_MODEL=BAAI/bge-reranker-base

# ─────────────────────────────────────────────
# Server
# ─────────────────────────────────────────────
HOST=127.0.0.1
PORT=8000

# ─────────────────────────────────────────────
# HuggingFace (required for reranker model download)
# ─────────────────────────────────────────────
HF_TOKEN=your_hf_token_here
```

> **Never commit your real `HF_TOKEN` to version control.** Add `.env` to `.gitignore`.

Frontend config in `frontend/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload and ingest a single document |
| `POST` | `/upload/batch` | Upload and ingest multiple documents |
| `GET` | `/status/{job_id}` | Poll ingestion status |
| `WS` | `/ws/status/{job_id}` | Live WebSocket ingestion stages |
| `POST` | `/chat` | Full Q&A response |
| `POST` | `/chat/stream` | SSE streaming Q&A |
| `POST` | `/chat/clear` | Clear server-side chat history |
| `GET` | `/documents` | List all uploaded documents |
| `GET` | `/images/{image_id}` | Retrieve an extracted image (JPEG) |
| `GET` | `/images` | List all image IDs in memory |
| `GET` | `/health` | Health check (vector count, image count, active jobs) |

### Chat Request

```bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What does the document say about X?", "k": 5}'
```

SSE events: `metadata` → repeated `token` → `[DONE]`

---

## Notes

- **Images** are held in memory for the server's lifetime; re-ingest to repopulate after a restart.
- **ChromaDB** persists to `./chroma_db/` and survives restarts.
- **BM25 index** is rebuilt in memory on each server start as documents are ingested; it does not persist across restarts.
- Edit `start.bat` file paths before using it on Windows.
