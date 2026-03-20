"""
main.py - FastAPI application: server setup, Pydantic models, background
          ingestion task, and all API / WebSocket endpoints.

Core RAG pipeline logic lives in rag.py.
"""

import json
import asyncio
import uuid
import base64
from typing import List
from pathlib import Path

# FastAPI
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain (for streaming)
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# RAG pipeline
from rag_engine import (
    UPLOAD_DIR,
    SUPPORTED_EXTENSIONS,
    image_store,
    get_vectorstore,
    is_supported_file,
    ingest_file,
    build_answer_message_content,
    generate_final_answer,
    LLM_MODEL,
)


# ----------------------------------------------
# App & middleware
# ----------------------------------------------
app = FastAPI(
    title="RAG Pipeline API",
    description="Multimodal RAG pipeline with PDF ingestion, vector search, and chat",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job tracker  { job_id: { status, stage, message, ... } }
ingestion_jobs: dict = {}
chat_history: List = []
chat_history_version = 0
chat_history_lock = asyncio.Lock()

REWRITE_QUERY_SYSTEM_PROMPT = (
    "Given the chat history, rewrite the new question so it is standalone and searchable. "
    "Return only the rewritten query."
)


# ----------------------------------------------
# Pydantic models
# ----------------------------------------------
class ChatRequest(BaseModel):
    query: str
    k: int = 5                   # number of chunks to retrieve
    include_images: bool = True  # whether to pass images to LLM


class ChatResponse(BaseModel):
    answer: str
    retrieved_chunks: int
    image_ids: List[str]         # IDs of images used; fetch via /images/{id}


class IngestionStatus(BaseModel):
    job_id: str
    status: str                  # pending | processing | done | failed
    stage: str = "pending"       # pending | parsing | chunking | summarizing | embedding | done | failed
    message: str
    files_processed: int = 0
    total_files: int = 0
    current_file: str = ""


async def get_chat_history_snapshot() -> tuple[List, int]:
    async with chat_history_lock:
        return list(chat_history), chat_history_version


async def resolve_search_query(query: str, history: List) -> str:
    if not history:
        return query

    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    rewrite_messages = [
        SystemMessage(content=REWRITE_QUERY_SYSTEM_PROMPT),
        *history,
        HumanMessage(content=f"New question: {query}"),
    ]
    response = await asyncio.to_thread(llm.invoke, rewrite_messages)
    rewritten = response.content if isinstance(response.content, str) else str(response.content)
    return rewritten.strip() or query


async def append_chat_turn(query: str, answer: str, expected_version: int) -> bool:
    async with chat_history_lock:
        if chat_history_version != expected_version:
            return False
        chat_history.extend([
            HumanMessage(content=query),
            AIMessage(content=answer),
        ])
        return True


async def clear_server_chat_history() -> int:
    global chat_history_version
    async with chat_history_lock:
        cleared_messages = len(chat_history)
        chat_history.clear()
        chat_history_version += 1
        return cleared_messages


# ----------------------------------------------
# Background task for async ingestion
# ----------------------------------------------

def background_ingest(job_id: str, file_paths: List[str]):
    ingestion_jobs[job_id]["status"] = "processing"
    ingestion_jobs[job_id]["stage"] = "pending"
    ingestion_jobs[job_id]["total_files"] = len(file_paths)
    processed = 0
    try:
        for fp in file_paths:
            ingestion_jobs[job_id]["current_file"] = Path(fp).name
            ingest_file(fp, job_id=job_id, ingestion_jobs=ingestion_jobs)
            processed += 1
            ingestion_jobs[job_id]["files_processed"] = processed
        ingestion_jobs[job_id]["status"] = "done"
        ingestion_jobs[job_id]["stage"] = "done"
        ingestion_jobs[job_id]["message"] = f"All {processed} file(s) ingested successfully."
    except Exception as e:
        ingestion_jobs[job_id]["status"] = "failed"
        ingestion_jobs[job_id]["stage"] = "failed"
        ingestion_jobs[job_id]["message"] = str(e)


# ----------------------------------------------
# API Endpoints
# ----------------------------------------------

# ── 1. Single file upload ─────────────────────────────────────────────────────
@app.post("/upload", summary="Upload and ingest a document")
async def upload_single(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a single document. Ingestion runs in the background.
    Supports: PDF, DOCX, TXT, PPTX, XLSX, CSV, HTML, Markdown, and more.
    Poll `/status/{job_id}` for progress.
    """
    if not file.filename or not is_supported_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    job_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    contents = await file.read()
    save_path.write_bytes(contents)

    ingestion_jobs[job_id] = {
        "status": "pending",
        "stage": "pending",
        "message": "Queued for ingestion.",
        "files_processed": 0,
        "total_files": 1,
        "current_file": "",
    }
    background_tasks.add_task(background_ingest, job_id, [str(save_path)])

    return {"job_id": job_id, "filename": file.filename, "status": "pending"}


# ── 2. Batch upload ───────────────────────────────────────────────────────────
@app.post("/upload/batch", summary="Upload and ingest multiple documents")
async def upload_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
):
    """
    Upload multiple documents in one request. Returns a single job_id.
    Supports: PDF, DOCX, TXT, PPTX, XLSX, CSV, HTML, Markdown, and more.
    Poll `/status/{job_id}` for progress.
    """
    for f in files:
        if not f.filename or not is_supported_file(f.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {f.filename}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            )

    job_id = str(uuid.uuid4())
    saved_paths = []
    for f in files:
        save_path = UPLOAD_DIR / f"{job_id}_{f.filename}"
        save_path.write_bytes(await f.read())
        saved_paths.append(str(save_path))

    ingestion_jobs[job_id] = {
        "status": "pending",
        "stage": "pending",
        "message": f"Queued {len(files)} file(s) for ingestion.",
        "files_processed": 0,
        "total_files": len(files),
        "current_file": "",
    }
    background_tasks.add_task(background_ingest, job_id, saved_paths)

    return {
        "job_id": job_id,
        "total_files": len(files),
        "filenames": [f.filename for f in files],
        "status": "pending",
    }


# ── 3. Ingestion status ───────────────────────────────────────────────────────
@app.get("/status/{job_id}", response_model=IngestionStatus, summary="Check ingestion status")
async def ingestion_status(job_id: str):
    if job_id not in ingestion_jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    job = ingestion_jobs[job_id]
    return IngestionStatus(
        job_id=job_id,
        status=job["status"],
        stage=job.get("stage", "pending"),
        message=job["message"],
        files_processed=job.get("files_processed", 0),
        total_files=job.get("total_files", 0),
        current_file=job.get("current_file", ""),
    )


# ── 4. Chat / Q&A ─────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse, summary="Ask a question over ingested docs")
async def chat(request: ChatRequest):
    """
    Retrieve relevant chunks from the vector store and generate a comprehensive answer using the LLM.
    Use all available context (text, tables, and images) from the retrieved documents to answer the user's question.
    If the information is insufficient, clearly state so in the response.
    Focus on accuracy and clarity, and cite the relevant document sections if possible.
    """
    history, history_version = await get_chat_history_snapshot()
    search_query = await resolve_search_query(request.query, history)

    vs = get_vectorstore()
    retriever = vs.as_retriever(
        search_type="similarity",
        search_kwargs={"k": request.k}
    )
    chunks = await asyncio.to_thread(retriever.invoke, search_query)

    if not chunks:
        return ChatResponse(
            answer="No relevant documents found in the vector store. Please upload and ingest documents first.",
            retrieved_chunks=0,
            image_ids=[],
        )

    answer, image_ids = await asyncio.to_thread(
        generate_final_answer,
        chunks,
        request.query,
        request.include_images,
        history,
    )
    await append_chat_turn(request.query, answer, history_version)

    return ChatResponse(
        answer=answer,
        retrieved_chunks=len(chunks),
        image_ids=image_ids,
    )


# ── 4b. Clear chat history ────────────────────────────────────────────────────
@app.post("/chat/clear", summary="Clear chat history")
async def clear_chat():
    """
    Clear the in-memory server-side chat history used for conversational follow-up.
    """
    cleared_messages = await clear_server_chat_history()
    return {
        "message": "Chat history cleared.",
        "status": "success",
        "cleared_messages": cleared_messages,
    }


# ── 5. Retrieve a specific image ──────────────────────────────────────────────
@app.get("/images/{image_id}", summary="Retrieve an image extracted from an ingested document")
async def get_image(image_id: str):
    """
    Returns the raw image (JPEG) by its ID.
    Image IDs are returned in the `image_ids` field of /chat responses.
    """
    img_b64 = image_store.get(image_id)
    if img_b64 is None:
        raise HTTPException(status_code=404, detail="Image not found.")

    img_bytes = base64.b64decode(img_b64)
    return Response(content=img_bytes, media_type="image/jpeg")


# ── 6. List all available images ──────────────────────────────────────────────
@app.get("/images", summary="List all image IDs currently in memory")
async def list_images():
    return {"total": len(image_store), "image_ids": list(image_store.keys())}


# ── 6b. List uploaded documents ───────────────────────────────────────────────
@app.get("/documents", summary="List all uploaded documents")
async def list_documents():
    """
    Scans the uploads/ directory and returns a list of previously uploaded
    documents with metadata. Persists across server restarts.
    """
    docs = []
    if UPLOAD_DIR.exists():
        for f in sorted(UPLOAD_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if f.is_file():
                stat = f.stat()
                # Filename format: {job_id}_{original_name}
                name = f.name
                # Strip the job_id prefix (UUID + underscore = 37 chars)
                original_name = name[37:] if len(name) > 37 and name[36] == "_" else name
                docs.append({
                    "filename": original_name,
                    "size_bytes": stat.st_size,
                    "uploaded_at": stat.st_mtime,
                    "full_path": name,
                })
    return {"total": len(docs), "documents": docs}


# ── 7. Health check ───────────────────────────────────────────────────────────
@app.get("/health", summary="Health check")
async def health():
    vs = get_vectorstore()
    try:
        count = vs._collection.count()
    except Exception:
        count = -1
    return {
        "status": "ok",
        "vector_store_documents": count,
        "images_in_memory": len(image_store),
        "active_jobs": len(ingestion_jobs),
    }


# ── 8. WebSocket for live ingestion status ────────────────────────────────────
@app.websocket("/ws/status/{job_id}")
async def ws_ingestion_status(websocket: WebSocket, job_id: str):
    await websocket.accept()
    if job_id not in ingestion_jobs:
        await websocket.send_json({"error": "Job not found"})
        await websocket.close()
        return
    try:
        prev_snapshot = None
        while True:
            job = ingestion_jobs.get(job_id)
            if job is None:
                await websocket.send_json({"error": "Job not found"})
                break
            snapshot = {
                "status": job["status"],
                "stage": job.get("stage", "pending"),
                "message": job["message"],
                "files_processed": job.get("files_processed", 0),
                "total_files": job.get("total_files", 0),
                "current_file": job.get("current_file", ""),
            }
            if snapshot != prev_snapshot:
                await websocket.send_json(snapshot)
                prev_snapshot = snapshot
            if job["status"] in ("done", "failed"):
                break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ── 9. Streaming chat (SSE) ───────────────────────────────────────────────────
@app.post("/chat/stream", summary="Stream a chat answer via SSE")
async def chat_stream(request: ChatRequest):
    """
    Same retrieval as /chat but streams the LLM answer token-by-token
    as Server-Sent Events.  First event contains metadata.
    Last event is `[DONE]`.
    """
    history, history_version = await get_chat_history_snapshot()
    search_query = await resolve_search_query(request.query, history)

    vs = get_vectorstore()
    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": request.k, "fetch_k": 10, "lambda_mult": 0.75}
    )
    chunks = await asyncio.to_thread(retriever.invoke, search_query)

    if not chunks:
        async def _empty():
            yield f"data: {json.dumps({'type': 'metadata', 'retrieved_chunks': 0, 'image_ids': []})}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'content': 'No relevant documents found. Please upload and ingest documents first.'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_empty(), media_type="text/event-stream")

    message_content, used_image_ids = build_answer_message_content(
        chunks,
        request.query,
        request.include_images,
    )
    llm_messages: List = [SystemMessage(content=(
        "You are a helpful assistant that answers questions using the provided document context. "
        "Use earlier chat turns only to resolve references in the current question, not to invent facts. "
        "Base the answer on the retrieved text, tables, and images. If the documents do not contain enough information, say so clearly."
    ))]
    llm_messages.extend(history)
    llm_messages.append(HumanMessage(content=message_content))

    async def _stream_tokens():
        yield f"data: {json.dumps({'type': 'metadata', 'retrieved_chunks': len(chunks), 'image_ids': used_image_ids})}\n\n"
        answer_parts: List[str] = []
        try:
            llm = ChatOllama(model=LLM_MODEL, temperature=0)
            for token_chunk in llm.stream(llm_messages):
                text = token_chunk.content if isinstance(token_chunk.content, str) else str(token_chunk.content)
                if text:
                    answer_parts.append(text)
                    yield f"data: {json.dumps({'type': 'token', 'content': text})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        else:
            await append_chat_turn(request.query, "".join(answer_parts), history_version)
        yield "data: [DONE]\n\n"

    return StreamingResponse(_stream_tokens(), media_type="text/event-stream")


# ----------------------------------------------
# Entry point
# ----------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)