"""Document ingestion endpoints."""

import logging
from io import BytesIO

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.api.main import get_mongodb_client, get_voyage_client
from src.clients import MongoDBClient, VoyageClient
from src.ingestion import IngestionPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["ingest"])


class IngestTextRequest(BaseModel):
    """Request for ingesting raw text."""

    text: str
    title: str = "Untitled"
    source: str = "direct_input"


class IngestResponse(BaseModel):
    """Ingest response."""

    status: str
    document_id: str | None = None
    title: str | None = None
    source: str | None = None
    chunks_created: int = 0
    elapsed_seconds: float | None = None
    message: str | None = None


def get_ingestion_pipeline(
    mongodb: MongoDBClient = Depends(get_mongodb_client),
    voyage: VoyageClient = Depends(get_voyage_client),
) -> IngestionPipeline:
    """Get ingestion pipeline with dependencies."""
    return IngestionPipeline(mongodb_client=mongodb, voyage_client=voyage)


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> IngestResponse:
    """Upload and ingest a file.

    Supported formats: PDF, Markdown (.md), Text (.txt)

    Args:
        file: File to upload and process

    Returns:
        IngestResponse with document_id and chunk count
    """
    logger.info(f"Received file upload: {file.filename}")

    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    suffix = file.filename.split(".")[-1].lower() if "." in file.filename else ""
    supported = {"pdf", "md", "markdown", "mdown", "txt", "text"}

    if suffix not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: .{suffix}. Supported: {', '.join(supported)}",
        )

    try:
        # Read file content
        content = await file.read()
        file_obj = BytesIO(content)
        file_obj.name = file.filename

        # Process through pipeline
        result = await pipeline.ingest_file(
            source=file_obj,
            filename=file.filename,
        )

        return IngestResponse(
            status=result.get("status", "success"),
            document_id=result.get("document_id"),
            title=result.get("title"),
            source=result.get("source"),
            chunks_created=result.get("chunks_created", 0),
            elapsed_seconds=result.get("elapsed_seconds"),
            message=result.get("message"),
        )

    except Exception as e:
        logger.error(f"File ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(
    request: IngestTextRequest,
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> IngestResponse:
    """Ingest raw text directly.

    Args:
        request: Text content with optional title and source

    Returns:
        IngestResponse with document_id and chunk count
    """
    logger.info(f"Received text ingestion: {request.title}")

    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text content is required")

    try:
        result = await pipeline.ingest_text(
            text=request.text,
            title=request.title,
            source=request.source,
        )

        return IngestResponse(
            status=result.get("status", "success"),
            document_id=result.get("document_id"),
            title=result.get("title"),
            source=request.source,
            chunks_created=result.get("chunks_created", 0),
            elapsed_seconds=result.get("elapsed_seconds"),
        )

    except Exception as e:
        logger.error(f"Text ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.get("/documents/{document_id}")
async def get_document(
    document_id: str,
    mongodb: MongoDBClient = Depends(get_mongodb_client),
) -> dict:
    """Get document metadata by ID.

    Args:
        document_id: Document identifier

    Returns:
        Document metadata
    """
    doc = await mongodb.db["documents"].find_one({"document_id": document_id})

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Convert ObjectId to string for JSON serialization
    doc["_id"] = str(doc["_id"])
    return doc


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    mongodb: MongoDBClient = Depends(get_mongodb_client),
) -> dict:
    """Delete a document and its chunks.

    Args:
        document_id: Document identifier

    Returns:
        Deletion status
    """
    # Delete chunks first
    chunks_result = await mongodb.db["chunks"].delete_many({"document_id": document_id})

    # Delete document
    doc_result = await mongodb.db["documents"].delete_one({"document_id": document_id})

    if doc_result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "status": "deleted",
        "document_id": document_id,
        "chunks_deleted": chunks_result.deleted_count,
    }
