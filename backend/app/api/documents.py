"""
Document upload and management API.
"""

import os
import uuid
import shutil
import structlog
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional

from app.config import config_manager
from app.dependencies import factory
from app.core.interfaces import Document

logger = structlog.get_logger()
router = APIRouter(prefix="/api/documents", tags=["Documents"])

UPLOAD_DIR = "./data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class DocumentResponse(BaseModel):
    id: str
    filename: str
    status: str
    chunk_count: int = 0
    message: str = ""


class IngestStatus(BaseModel):
    total_files: int
    processed: int
    chunks_created: int
    errors: list[str] = []


async def _process_document(file_path: str, filename: str, cfg: dict):
    """Background task for document processing pipeline."""
    try:
        components = factory.build_all(cfg)
        embedding = components["embedding"]
        vectorstore = components["vectorstore"]
        chunker = components["chunker"]

        # Parse
        from app.multimodal.parser import UniversalDocumentParser
        ocr = None
        asr = None
        mm_cfg = cfg.get("multimodal", {})
        if mm_cfg.get("ocr", {}).get("enabled", False):
            from app.core.registry import registry, ComponentRegistry
            ocr_provider = mm_cfg.get("ocr", {}).get("provider", "tesseract")
            try:
                ocr = registry.create(ComponentRegistry.OCR, ocr_provider)
            except Exception:
                pass
        if mm_cfg.get("asr", {}).get("enabled", False):
            from app.core.registry import registry, ComponentRegistry
            asr_provider = mm_cfg.get("asr", {}).get("provider", "whisper")
            try:
                asr = registry.create(ComponentRegistry.ASR, asr_provider)
            except Exception:
                pass

        parser = UniversalDocumentParser(ocr=ocr, asr=asr)
        raw_docs = await parser.parse(file_path)

        if not raw_docs:
            logger.warning("no_content_extracted", filename=filename)
            return

        # Chunk
        all_chunks = []
        for doc in raw_docs:
            chunks = await chunker.chunk(doc.content, metadata=doc.metadata)
            all_chunks.extend(chunks)

        if not all_chunks:
            logger.warning("no_chunks_created", filename=filename)
            return

        # Embed
        texts = [c.content for c in all_chunks]
        embeddings = await embedding.embed_texts(texts)
        for chunk, emb in zip(all_chunks, embeddings):
            chunk.embedding = emb

        # Store
        await vectorstore.add_documents(all_chunks)

        # Also add to BM25 sparse index if hybrid mode
        retriever = components.get("retriever")
        if hasattr(retriever, 'sparse') and hasattr(retriever.sparse, 'add_documents'):
            retriever.sparse.add_documents(all_chunks)
        elif hasattr(retriever, 'add_documents'):
            retriever.add_documents(all_chunks)

        logger.info(
            "document_processed",
            filename=filename,
            chunks=len(all_chunks),
        )
    except Exception as e:
        logger.error("document_processing_failed", filename=filename, error=str(e))


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session_id: str = "default",
):
    """Upload and process a document."""
    doc_id = str(uuid.uuid4())
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "txt"

    # Validate file type
    supported = ["pdf", "txt", "md", "docx", "png", "jpg", "jpeg", "mp3", "wav"]
    if ext not in supported:
        raise HTTPException(400, f"Unsupported file type: {ext}. Supported: {supported}")

    # Save file
    file_path = os.path.join(UPLOAD_DIR, f"{doc_id}.{ext}")
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Process in background
    cfg = config_manager.get_session_config(session_id)
    background_tasks.add_task(_process_document, file_path, file.filename, cfg)

    return DocumentResponse(
        id=doc_id,
        filename=file.filename,
        status="processing",
        message="Document uploaded and processing started.",
    )


@router.get("/stats")
async def get_stats(session_id: str = "default"):
    """Get vector store statistics."""
    cfg = config_manager.get_session_config(session_id)
    components = factory.build_all(cfg)
    stats = await components["vectorstore"].get_collection_stats()
    return stats