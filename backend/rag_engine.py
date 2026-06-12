"""
RAG Engine - Core logic for document ingestion, retrieval, and chat.
Extracted from rag.ipynb for production use with FastAPI.
"""

import os
import uuid
import base64
import json
import re
from typing import List, Optional, Dict
from pathlib import Path

# Unstructured for document parsing
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title

# LangChain components
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.retrievers import BM25Retriever

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

BASE_DIR = Path(__file__).resolve().parent

PERSIST_DIRECTORY = str(BASE_DIR / "chroma_db")
EMBEDDING_MODEL = "nomic-embed-text-v2-moe:latest"
LLM_MODEL = os.getenv("LLM_MODEL")
VISION_MODEL = os.getenv("VISION_MODEL")
MIN_IMAGE_SIZE = 5000
MIN_TEXT_CHARS = 100
RERANKER_MODEL = os.getenv("RERANKER_MODEL")

ANSWER_SYSTEM_PROMPT = """
You are a retrieval-augmented question answering assistant.

Answer the user's question ONLY using the information provided in the retrieved document context.

Rules:
1. Do not use prior knowledge or external information.
2. Use earlier chat turns only to resolve references in the current question.
3. Base the answer strictly on the retrieved text, tables, and images.
4. If the retrieved information does not contain enough information, explicitly state:
   "The retrieved documents do not contain sufficient information to answer this question."
5. Do not infer facts that are not directly supported by the retrieved information.
6. Prefer exact facts, numbers, names, equations, and terminology from the documents.
7. If information is spread across multiple retrieved passages, combine it into a single coherent answer.
8. Do not mention "retrieved context" or "documents" unless information is missing.
9. Keep answers concise but complete.
10. Never fabricate missing information.
"""

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".txt",
    ".pptx",
    ".xlsx",
    ".xls",
    ".csv",
    ".html",
    ".htm",
    ".md",
    ".xml",
    ".json",
    ".rtf",
}


def is_supported_file(filename: str) -> bool:
    """Check if the file extension is supported for ingestion."""
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


# ---------------------------------------------------------------------------
# Image store (in-memory: image_id -> base64)
# ---------------------------------------------------------------------------
image_store: Dict[str, str] = {}

# ---------------------------------------------------------------------------
# Declaring Global Variables
# ---------------------------------------------------------------------------
_vectorstore: Optional[Chroma] = None
_bm25_retriever = None
_all_documents = []

# ---------------------------------------------------------------------------
# Vector store (singleton pattern)
# ---------------------------------------------------------------------------
def get_vectorstore() -> Chroma:
    global _vectorstore, _bm25_retriever, _all_documents

    if _vectorstore is None:
        embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL)
        _vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embedding_model,
        )

        # ── NEW: Rebuild BM25 from persisted ChromaDB docs ──
        if _bm25_retriever is None:
            try:
                result = _vectorstore.get(include=["documents", "metadatas"])
                if result["documents"]:
                    _all_documents = [
                        Document(page_content=doc, metadata=meta)
                        for doc, meta in zip(result["documents"], result["metadatas"])
                    ]
                    _bm25_retriever = BM25Retriever.from_documents(_all_documents)
                    _bm25_retriever.k = 10
                    print(f"BM25 rebuilt from ChromaDB: {len(_all_documents)} docs")
                else:
                    print("ChromaDB empty — BM25 not initialized")
            except Exception as e:
                print(f"BM25 rebuild failed: {e}")
        # ─────────────────────────────────────────────────────

    return _vectorstore


# ---------------------------------------------------------------------------
# Document parsing & chunking
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Remove extraction noise before text is embedded or sent to the LLM."""
    if not text:
        return ""

    cleaned_lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            cleaned_lines.append("")
            continue
        if re.fullmatch(r"\d{1,4}", line):
            continue
        if re.fullmatch(r"(?:[A-Za-z0-9.\[\]/:-]\s+){4,}[A-Za-z0-9.\[\]/:-]?", line):
            continue
        if re.fullmatch(r"(?:arXiv|cs\.[A-Z]{2}|stat\.[A-Z]{2}|math\.[A-Z]{2}).*", line, re.IGNORECASE):
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n\s*\d+\s*\n", "\n", cleaned)
    cleaned = re.sub(r"\b\d{4}\s+\d\s+0\s+2\b", " ", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def is_useless_text_chunk(text: str, *, has_tables: bool = False, has_images: bool = False) -> bool:
    """Filter low-value text chunks while keeping meaningful tables and images."""
    if has_tables or has_images:
        return False

    normalized = text.strip()
    if len(normalized) < MIN_TEXT_CHARS:
        return True

    lower = normalized.lower()
    words = re.findall(r"[a-zA-Z]{3,}", normalized)
    if not words:
        return True
    if len(words) < 20:
        return True
    if lower.startswith(("references", "bibliography")) and len(words) < 80:
        return True
    if "copyright" in lower and len(words) < 80:
        return True

    return False


def get_chunk_section(chunk) -> str:
    """Infer the closest section heading from unstructured chunk metadata."""
    if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
        for element in chunk.metadata.orig_elements:
            if type(element).__name__ == "Title":
                section = clean_text(getattr(element, "text", ""))
                if section:
                    return section[:200]
    first_line = clean_text(getattr(chunk, "text", "")).split(". ")[0]
    return first_line[:200] if first_line else "Unknown"


def get_chunk_page(chunk) -> Optional[int]:
    """Return the first page number available on a chunk or original element."""
    metadata = getattr(chunk, "metadata", None)
    page_number = getattr(metadata, "page_number", None)
    if page_number is not None:
        return page_number
    if metadata is not None and hasattr(metadata, "orig_elements"):
        for element in metadata.orig_elements:
            element_page = getattr(getattr(element, "metadata", None), "page_number", None)
            if element_page is not None:
                return element_page
    return None


def partition_document(file_path: str):
    """Parse a document using unstructured with PDF-specific options."""
    ext = Path(file_path).suffix.lower()
    kwargs: dict = {"filename": file_path}

    # PDF-specific options for high-quality extraction
    if ext == ".pdf":
        kwargs.update(
            {
                "strategy": "hi_res",
                "infer_table_structure": True,
                "extract_image_block_types": ["Image"],
                "extract_image_block_to_payload": True,
            }
        )

    elements = partition(**kwargs)
    return elements


def create_chunks_by_title(elements):
    """Chunk elements by title with appropriate limits."""
    return chunk_by_title(
        elements,
        max_characters=3000,
        new_after_n_chars=2400,
        combine_text_under_n_chars=500,
    )


def separate_content_types(chunk):
    """Separate text, tables, and images from a chunk."""
    content_data = {
        "text": clean_text(chunk.text),
        "tables": [],
        "images": [],
        "types": ["text"],
        "section": get_chunk_section(chunk),
        "page": get_chunk_page(chunk),
    }
    if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__
            if element_type == "Table":
                content_data["types"].append("table")
                table_html = getattr(element.metadata, "text_as_html", element.text)
                content_data["tables"].append(table_html)
            elif element_type == "Image":
                if hasattr(element, "metadata") and hasattr(
                    element.metadata, "image_base64"
                ):
                    content_data["types"].append("image")
                    content_data["images"].append(element.metadata.image_base64)
    content_data["types"] = list(set(content_data["types"]))
    return content_data


# ---------------------------------------------------------------------------
# AI-enhanced summarization
# ---------------------------------------------------------------------------
def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:
    """Create a searchable summary enhanced by AI for better retrieval."""
    try:
        llm = ChatOllama(model=VISION_MODEL, temperature=0)

        prompt_text = f"""You are creating a searchable description for document content retrieval.

CONTENT TO ANALYZE:
TEXT CONTENT:
{text}

"""
        if tables:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i + 1}:\n{table}\n\n"

        prompt_text += """
                        YOUR TASK:
                        Generate a comprehensive, searchable description that covers:
                        1. Key facts, numbers, and data points from text and tables
                        2. Main topics and concepts discussed
                        3. Questions this content could answer
                        4. Visual content analysis (charts, diagrams, patterns in images)
                        5. Alternative search terms users might use

                        Make it detailed and searchable - prioritize findability over brevity.

                        SEARCHABLE DESCRIPTION:"""

        message_content: List = [{"type": "text", "text": prompt_text}]
        for image_base64 in images:
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                }
            )

        response = llm.invoke([HumanMessage(content=message_content)])
        if isinstance(response.content, str):
            return response.content
        else:
            return str(response.content)

    except Exception as e:
        summary = f"{text[:300]}..."
        if tables:
            summary += f" [Contains {len(tables)} table(s)]"
        if images:
            summary += f" [Contains {len(images)} image(s)]"
        return summary


# ---------------------------------------------------------------------------
# Chunk summarization
# ---------------------------------------------------------------------------
def summarise_chunks(chunks) -> List[Document]:
    """Process chunks into LangChain Documents with enhanced summaries."""
    langchain_documents = []
    content_data = dict()
    for chunk in chunks:
        content_data = separate_content_types(chunk)
        if is_useless_text_chunk(
            content_data["text"],
            has_tables=bool(content_data["tables"]),
            has_images=bool(content_data["images"]),
        ):
            continue
        
        search_summary = ""

        if content_data["tables"] or content_data["images"]:
            try:
                search_summary = create_ai_enhanced_summary(
                content_data["text"],
                content_data["tables"],
                content_data["images"],
                )
            except Exception:
                search_summary = ""

        table_text = "\n".join(
            str(table)
            for table in content_data["tables"]
        )

        enhanced_content = f"""
                            SECTION: {content_data['section']}
                            PAGE: {content_data['page']}

                            TEXT:
                            {content_data['text']}

                            TABLES:
                            {table_text}

                            KEYWORDS:
                            {search_summary}
                            """
        # Persist images to in-memory store and record their IDs
        img_ids = []
        for img_b64 in content_data["images"]:
            img_id = str(uuid.uuid4())
            image_store[img_id] = img_b64
            img_ids.append(img_id)

        doc = Document(
            page_content=clean_text(enhanced_content),
            metadata={
                "section": content_data["section"],
                "page": content_data["page"] or 0,
                "search_summary": search_summary,
                "original_content": json.dumps(
                    {
                        "raw_text": content_data["text"],
                        "tables_html": content_data["tables"],
                        "image_ids": img_ids,
                    }
                )
            },
        )
        langchain_documents.append(doc)

    return langchain_documents


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------
def ingest_file(
    file_path: str,
    job_id: Optional[str] = None,
    ingestion_jobs: Optional[dict] = None,
) -> List[Document]:
    """Parse, chunk, summarise, and embed a single file into the vector store."""
    global _bm25_retriever
    global _all_documents


    def _update_stage(stage: str, message: str):
        if job_id and ingestion_jobs and job_id in ingestion_jobs:
            ingestion_jobs[job_id]["stage"] = stage
            ingestion_jobs[job_id]["message"] = message

    _update_stage("parsing", f"Parsing {Path(file_path).name}...")
    elements = partition_document(file_path)

    _update_stage("chunking", f"Chunking {Path(file_path).name}...")
    chunks = create_chunks_by_title(elements)

    _update_stage("summarizing", f"Summarizing {Path(file_path).name}...")
    docs = summarise_chunks(chunks)

    _update_stage("embedding", f"Embedding & vectorizing {Path(file_path).name}...")
    
    vs = get_vectorstore()
    
    _all_documents.extend(docs)
    _bm25_retriever = BM25Retriever.from_documents(_all_documents)

    _bm25_retriever.k = 10

    print(f"BM25 initialized with {len(docs)} documents")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Number of docs: {len(docs)}")

    if docs:
        print("First document preview:")
        print(docs[0].page_content[:500])

    try:
        print("Collection count BEFORE:", vs._collection.count())
    except Exception as e:
        print("Count error before embedding:", e)

    if docs:
        vs.add_documents(docs)

    try:
        print("Collection count AFTER:", vs._collection.count())
    except Exception as e:
        print("Count error after embedding:", e)

    return docs
# --------------------------------------------------------------------------
# BM-25 Helper Function
# --------------------------------------------------------------------------
def get_bm25_retriever():
    global _bm25_retriever
    return _bm25_retriever



# ---------------------------------------------------------------------------
# Retrieval reranking
# ---------------------------------------------------------------------------
_reranker = None


def rerank_chunks(query: str, chunks, top_k: int = 5):
    """Rerank retrieved chunks with a local cross-encoder when available."""
    global _reranker
    if not chunks:
        return chunks

    try:
        if _reranker is None:
            from sentence_transformers import CrossEncoder

            _reranker = CrossEncoder(RERANKER_MODEL)
        pairs = [(query, chunk.page_content) for chunk in chunks]
        scores = _reranker.predict(pairs)
    except Exception as e:
        print("Reranker unavailable, using vector order:", e)
        return chunks[:top_k]

    ranked = sorted(zip(chunks, scores), key=lambda item: float(item[1]), reverse=True)
    return [chunk for chunk, _score in ranked[:top_k]]


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------
def build_answer_message_content(
    chunks,
    query: str,
    include_images: bool = True,
    rewritten_query: Optional[str] = None
) -> tuple:
    """Build multimodal prompt content and return it with any referenced image IDs."""
    used_image_ids: List[str] = []
    prompt_text = f"""
    Original User Question:
    {query}
    """

    if rewritten_query and rewritten_query != query:
        prompt_text += f"""

    Resolved Standalone Question:
    {rewritten_query}
    """

    prompt_text += """

    Retrieved Information:
    """
    for i, chunk in enumerate(chunks):
        section = chunk.metadata.get(
            "section",
            "Unknown"
        )
        page = chunk.metadata.get(
            "page",
            "Unknown"
        )
        prompt_text += (
                    f"\n\n===== SOURCE {i + 1} =====\n"
                    f"Section: {section}\n"
                    f"Page: {page}\n"
                    f"----------------------------------\n"
                    )
        if "original_content" in chunk.metadata:
            original_data = json.loads(
                chunk.metadata["original_content"]
            )
            raw_text = original_data.get(
                "raw_text",
                ""
            )
            if raw_text:
                prompt_text += (
                    "TEXT:\n"
                    f"{raw_text}\n\n"
                )
            tables_html = original_data.get(
                "tables_html",
                []
            )
            if tables_html:
                prompt_text += "TABLES:\n"
                for j, table in enumerate(tables_html):
                    table_text = str(table)
                    prompt_text += (
                        f"Table {j + 1}:\n"
                        f"{table_text}\n\n"
                    )
    print(
        "Prompt Length:",
        len(prompt_text)
    )
    print(
        "Approx Tokens:",
        len(prompt_text) // 4
    )
    message_content: List = [
        {
            "type": "text",
            "text": prompt_text,
        }
    ]
    if include_images:
        for chunk in chunks:
            if "original_content" in chunk.metadata:
                original_data = json.loads(
                    chunk.metadata["original_content"]
                )
                for img_id in original_data.get(
                    "image_ids",
                    []
                ):
                    img_b64 = image_store.get(
                        img_id
                    )
                    if img_b64:
                        used_image_ids.append(
                            img_id
                        )
                        message_content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url":
                                    f"data:image/jpeg;base64,{img_b64}"
                                },
                            }
                        )
    return message_content, used_image_ids


def generate_final_answer(
    chunks,
    query: str,
    include_images: bool = True,
    chat_history: Optional[List] = None,
    rewritten_query: Optional[str] = None,
) -> tuple:
    """Generate an answer from retrieved chunks. Returns (answer_text, list_of_image_ids_used)."""

    try:
        llm = ChatOllama(model=LLM_MODEL, temperature=0, num_ctx=8192)
        message_content, used_image_ids = build_answer_message_content(chunks,query,include_images,rewritten_query)
        messages: List = [SystemMessage(content=ANSWER_SYSTEM_PROMPT)]
        if chat_history:
            messages.extend(chat_history)
        messages.append(HumanMessage(content=message_content))
        print("Prompt Length:",len(str(message_content)))
        print("Approx Tokens:",len(str(message_content)) //4)
        response = llm.invoke(messages)
        answer_text = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        return answer_text, used_image_ids

    except Exception as e:
        return f"Error generating answer: {e}", []
