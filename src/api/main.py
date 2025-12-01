"""FastAPI application setup."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_settings
from src.clients import MongoDBClient, VoyageClient, GeminiClient
from src.utils import setup_logging

logger = logging.getLogger(__name__)

# Global client instances
_mongodb_client: MongoDBClient | None = None
_voyage_client: VoyageClient | None = None
_gemini_client: GeminiClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _mongodb_client, _voyage_client, _gemini_client

    # Setup logging
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info("Starting SOTA RAG API")

    # Initialize clients
    _mongodb_client = MongoDBClient(settings)
    _voyage_client = VoyageClient(settings)
    _gemini_client = GeminiClient(settings)
    logger.info("Clients initialized")

    # Create indexes
    try:
        await _mongodb_client.create_indexes()
        logger.info("MongoDB indexes created")
    except Exception as e:
        logger.warning(f"Index creation warning: {e}")

    yield

    # Cleanup
    if _mongodb_client:
        await _mongodb_client.close()
    if _voyage_client:
        await _voyage_client.close()
    if _gemini_client:
        await _gemini_client.close()
    logger.info("Clients closed")


# Create FastAPI app
app = FastAPI(
    title="SOTA RAG API",
    description="State-of-the-Art RAG reference implementation for MongoDB",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency injection
def get_mongodb_client() -> MongoDBClient:
    """Get MongoDB client instance."""
    if _mongodb_client is None:
        raise RuntimeError("MongoDB client not initialized")
    return _mongodb_client


def get_voyage_client() -> VoyageClient:
    """Get Voyage client instance."""
    if _voyage_client is None:
        raise RuntimeError("Voyage client not initialized")
    return _voyage_client


def get_gemini_client() -> GeminiClient:
    """Get Gemini client instance."""
    if _gemini_client is None:
        raise RuntimeError("Gemini client not initialized")
    return _gemini_client


# Import routes after app creation to avoid circular imports
from src.api.routes import health, query, ingest, generate

app.include_router(health.router)
app.include_router(query.router)
app.include_router(ingest.router)
app.include_router(generate.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "SOTA RAG API",
        "version": "0.1.0",
        "status": "running",
    }
