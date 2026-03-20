"""
RAG Engine - Core logic for document ingestion, retrieval, and chat.
Extracted from rag.ipynb for production use with FastAPI.
"""

import os
import uuid
import base64
import json
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

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

PERSIST_DIRECTORY = "chroma_db"
EMBEDDING_MODEL = "nomic-embed-text-v2-moe:latest"
LLM_MODEL = "deepseek-r1:8b"
VISION_MODEL = "qwen2.5vl:3b"
MIN_IMAGE_SIZE = 5000

ANSWER_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using the provided document context. "
    "Use earlier chat turns only to resolve references in the current question, not to invent facts. "
    "Base the answer on the retrieved text, tables, and images. If the documents do not contain enough information, say so clearly."
)

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".txt", ".pptx", ".xlsx", ".xls",
    ".csv", ".html", ".htm", ".md", ".xml", ".json", ".rtf",
}


def is_supported_file(filename: str) -> bool:
    """Check if the file extension is supported for ingestion."""
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


# ---------------------------------------------------------------------------
# Image store (in-memory: image_id -> base64)
# ---------------------------------------------------------------------------
image_store: Dict[str, str] = {}


# ---------------------------------------------------------------------------
# Vector store (singleton pattern)
# ---------------------------------------------------------------------------
_vectorstore: Optional[Chroma] = None


def get_vectorstore() -> Chroma:
    """Get or create the Chroma vector store."""
    global _vectorstore
    if _vectorstore is None:
        embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL)
        _vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embedding_model,
        )
    return _vectorstore


# ---------------------------------------------------------------------------
# Document parsing & chunking
# ---------------------------------------------------------------------------
def partition_document(file_path: str):
    """Parse a document using unstructured with PDF-specific options."""
    ext = Path(file_path).suffix.lower()
    kwargs: dict = {"filename": file_path}

    # PDF-specific options for high-quality extraction
    if ext == ".pdf":
        kwargs.update({
            "strategy": "hi_res",
            "infer_table_structure": True,
            "extract_image_block_types": ["Image"],
            "extract_image_block_to_payload": True,
        })

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
        "text": chunk.text,
        "tables": [],
        "images": [],
        "types": ["text"],
    }
    if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__
            if element_type == "Table":
                content_data["types"].append("table")
                table_html = getattr(element.metadata, "text_as_html", element.text)
                content_data["tables"].append(table_html)
            elif element_type == "Image":
                if hasattr(element, "metadata") and hasattr(element.metadata, "image_base64"):
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
                prompt_text += f"Table {i+1}:\n{table}\n\n"

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
    for chunk in chunks:
        content_data = separate_content_types(chunk)

        if content_data["tables"] or content_data["images"]:
            try:
                enhanced_content = create_ai_enhanced_summary(
                    content_data["text"],
                    content_data["tables"],
                    content_data["images"],
                )
            except Exception:
                enhanced_content = content_data["text"]
        else:
            enhanced_content = content_data["text"]

        # Persist images to in-memory store and record their IDs
        img_ids = []
        for img_b64 in content_data["images"]:
            img_id = str(uuid.uuid4())
            image_store[img_id] = img_b64
            img_ids.append(img_id)

        doc = Document(
            page_content=enhanced_content,
            metadata={
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
    vs.add_documents(docs)

    return docs


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------
def build_answer_message_content(chunks, query: str, include_images: bool = True) -> tuple:
    """Build multimodal prompt content and return it with any referenced image IDs."""
    used_image_ids: List[str] = []

    prompt_text = f"""Based on the following documents, please answer this question: {query}

CONTENT TO ANALYZE:
"""
    for i, chunk in enumerate(chunks):
        prompt_text += f"--- Document {i+1} ---\n"
        if "original_content" in chunk.metadata:
            original_data = json.loads(chunk.metadata["original_content"])
            raw_text = original_data.get("raw_text", "")
            if raw_text:
                prompt_text += f"TEXT:\n{raw_text}\n\n"
            tables_html = original_data.get("tables_html", [])
            if tables_html:
                prompt_text += "TABLES:\n"
                for j, table in enumerate(tables_html):
                    prompt_text += f"Table {j+1}:\n{table}\n\n"
        prompt_text += "\n"

    prompt_text += """
Please provide a clear, comprehensive answer using the text, tables, and images above.
If the documents don't contain sufficient information, say so.

ANSWER:"""

    message_content: List = [{"type": "text", "text": prompt_text}]

    if include_images:
        for chunk in chunks:
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                for img_id in original_data.get("image_ids", []):
                    img_b64 = image_store.get(img_id)
                    if img_b64:
                        used_image_ids.append(img_id)
                        message_content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_b64}"
                                },
                            }
                        )

    return message_content, used_image_ids


def generate_final_answer(
    chunks,
    query: str,
    include_images: bool = True,
    chat_history: Optional[List] = None,
) -> tuple:
    """Generate an answer from retrieved chunks. Returns (answer_text, list_of_image_ids_used)."""

    try:
        llm = ChatOllama(model=LLM_MODEL, temperature=0)
        message_content, used_image_ids = build_answer_message_content(chunks, query, include_images)
        messages: List = [SystemMessage(content=ANSWER_SYSTEM_PROMPT)]
        if chat_history:
            messages.extend(chat_history)
        messages.append(HumanMessage(content=message_content))

        response = llm.invoke(messages)
        answer_text = response.content if isinstance(response.content, str) else str(response.content)
        return answer_text, used_image_ids

    except Exception as e:
        return f"Error generating answer: {e}", []