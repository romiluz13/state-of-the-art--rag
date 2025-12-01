"""API routes."""

from .health import router as health_router
from .ingest import router as ingest_router
from .query import router as query_router
from .generate import router as generate_router

__all__ = [
    "health_router",
    "ingest_router",
    "query_router",
    "generate_router",
]
