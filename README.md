# RAG Pipeline

Multimodal RAG pipeline with a **Next.js** web interface, **FastAPI** backend, **LangChain**, **ChromaDB**, and **Ollama**. Upload documents (PDF, DOCX, TXT, PPTX, and more), watch the pipeline process them in real-time, and chat with your documents.

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.com) running locally

### 1. Pull Ollama models

```bash
ollama pull deepseek-r1:8b
ollama pull nomic-embed-text-v2-moe
ollama pull qwen2.5vl:3b
```

> Models are configurable via `.env` — see [Configuration](#configuration).

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the backend

```bash
uvicorn rag:app --host 127.0.0.1 --port 8000 --reload
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
- **Main area** — chat with your documents, answers stream token-by-token, images display inline

> **Tip:** You can also run both servers at once with `start.bat` (Windows).

---

## Supported File Types

The pipeline accepts any file type supported by [Unstructured](https://docs.unstructured.io/):

| Category | Extensions |
|----------|-----------|
| Documents | `.pdf`, `.docx`, `.doc`, `.odt`, `.rtf` |
| Presentations | `.pptx`, `.ppt` |
| Spreadsheets | `.xlsx`, `.xls`, `.csv`, `.tsv` |
| Text | `.txt`, `.md`, `.rst`, `.org` |
| Web | `.html`, `.htm`, `.xml` |
| Email | `.eml`, `.msg` |
| Other | `.epub`, `.json` |

PDFs get enhanced processing with high-resolution parsing, table structure inference, and image extraction. Other formats use Unstructured's auto-detection.

---

## Project Structure

```
├── backend/
│   ├── main.py             # FastAPI entrypoint (if present)
│   ├── rag.py              # FastAPI backend — API, pipeline, WebSocket, SSE
│   ├── requirements.txt    # Python dependencies
│   ├── .env                # Model & server configuration
│   ├── chroma_db/          # Persisted ChromaDB vector store
│   ├── uploads/            # Uploaded files
│   └── __init__.py         # Marks backend as a Python package
├── frontend/               # Next.js web interface
│   ├── app/
│   │   ├── layout.tsx      # Root layout
│   │   ├── page.tsx        # Main two-panel page
│   │   └── globals.css     # Tailwind + custom styles
│   ├── components/
│   │   ├── ChatWindow.tsx  # Chat with streaming + markdown + images
│   │   ├── UploadPanel.tsx # Drag-drop upload + WebSocket status
│   │   └── PipelineStatus.tsx  # Step-by-step pipeline indicator
│   ├── lib/
│   │   ├── api.ts          # API client (upload, WebSocket, SSE, images)
│   │   └── types.ts        # TypeScript type definitions
│   └── .env.local          # NEXT_PUBLIC_API_URL=http://localhost:8000
└── start.bat               # Launch both backend + frontend (Windows)
```

---

## Features

### Web Interface
- **Real-time pipeline tracking** — WebSocket pushes stage updates as each file is processed (parsing, chunking, summarizing, embedding, done)
- **Streaming chat** — LLM responses stream token-by-token via SSE with a typewriter effect
- **Markdown rendering** — chat answers render with full markdown support (tables, code, lists)
- **Image display** — images extracted from PDFs display inline in chat responses
- **Multi-format upload** — drag-and-drop any supported file type (PDF, DOCX, TXT, PPTX, XLSX, CSV, HTML, MD, and more)
- **Batch upload** — upload multiple files at once with per-file progress tracking
- **Responsive design** — collapsible sidebar on mobile

### Backend
- **Multi-format document ingestion** — auto-detects file type via Unstructured; PDFs get hi-res extraction of text, tables (HTML), and images (base64)
- **AI-enhanced summaries** — LLM creates searchable descriptions for chunks with tables/images
- **Semantic search** — ChromaDB with configurable embedding model, cosine similarity
- **WebSocket endpoint** — `/ws/status/{job_id}` for live pipeline stage updates
- **SSE streaming** — `/chat/stream` for token-by-token chat responses
- **Configurable models** — LLM and embedding model names loaded from `.env`

---

## Configuration

All model and server settings are configured via the `.env` file in the project root:

```env
# LLM used for summaries and chat answers
LLM_MODEL=llama3.2:3b

# Embedding model for vector store
EMBEDDING_MODEL=nomic-embed-text-v2-moe:latest

# Vision model to summarise Tables and Images
VISION_MODEL=qwen2.5vl:3b

# Server
HOST=127.0.0.1
PORT=8000
```

Frontend configuration is in `frontend/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

| Variable | File | Default | Description |
|----------|------|---------|-------------|
| `LLM_MODEL` | `.env` | `llama3.2:3b` | Ollama model for summaries & chat |
| `EMBEDDING_MODEL` | `.env` | `nomic-embed-text-v2-moe:latest` | Ollama embedding model for vector store |
| `VISION_MODEL` | `.env` | `qwen2.5-vl:3b` | Ollama model for summarising tables & images|
| `HOST` | `.env` | `127.0.0.1` | Backend server host |
| `PORT` | `.env` | `8000` | Backend server port |
| `NEXT_PUBLIC_API_URL` | `frontend/.env.local` | `http://localhost:8000` | Backend URL for the frontend |

To swap models, edit `.env` and restart the backend. For example, to use a vision-capable model:

```env
LLM_MODEL=llava:13b
```

---

## API Reference

### `POST /upload` — Upload a document

Supports all [supported file types](#supported-file-types).

```bash
curl -X POST http://localhost:8000/upload -F "file=@document.pdf"
```

### `POST /upload/batch` — Batch upload

```bash
curl -X POST http://localhost:8000/upload/batch \
  -F "files=@doc1.pdf" -F "files=@report.docx" -F "files=@notes.txt"
```

### `GET /status/{job_id}` — Ingestion status

Returns `{ job_id, status, stage, message, files_processed, total_files, current_file }`.

Stages: `pending` → `parsing` → `chunking` → `summarizing` → `embedding` → `done`

### `WS /ws/status/{job_id}` — WebSocket live status

Connects and pushes JSON status updates every 0.5s until done/failed.

### `POST /chat` — Ask a question (full response)

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What does the document say about X?", "k": 3}'
```

### `POST /chat/stream` — Ask a question (SSE streaming)

Returns `text/event-stream`. Events:
1. `{"type": "metadata", "retrieved_chunks": 3, "image_ids": [...]}`
2. `{"type": "token", "content": "..."}` (repeated)
3. `[DONE]`

### `GET /images/{image_id}` — Retrieve an extracted image (JPEG)

### `GET /images` — List all image IDs

### `GET /health` — Health check

---

## Architecture

```
Document Upload (PDF, DOCX, TXT, PPTX, ...)
   │
   ▼
partition (unstructured, auto-detect)        ← stage: "parsing"
   │  PDFs: hi_res strategy, table + image extraction
   │  Others: auto-detected format parsing
   ▼
chunk_by_title (unstructured)                ← stage: "chunking"
   │  max 3000 chars per chunk
   ▼
AI Summary (LLM_MODEL)                      ← stage: "summarizing"
   │  multimodal: text + tables + images → searchable description
   ▼
ChromaDB (EMBEDDING_MODEL)                  ← stage: "embedding"
   │  cosine similarity
   ▼
/chat/stream → retrieve top-k → LLM → SSE token stream
                              └─→ image_ids for inline display
```

## Notes

- **Images** are kept in memory during the server's lifetime. Restart clears them (re-ingest to repopulate).
- **ChromaDB** is persisted to `./chroma_db/` — data survives restarts.
- To use a different model, update `LLM_MODEL` or `EMBEDDING_MODEL` in `.env` and restart.
- Make changes in the file path in the `start.bat` file before using it.
