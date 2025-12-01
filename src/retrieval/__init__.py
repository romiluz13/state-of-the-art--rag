"""Retrieval module for SOTA RAG system.

Implements 6 retrieval strategies:
1. VectorSearcher - $vectorSearch with binary quantization
2. TextSearcher - $search for BM25
3. HybridSearcher - $rankFusion combining vector + BM25
4. GraphRAGRetriever - $graphLookup for knowledge graphs
5. RAPTORRetriever - Hierarchical multi-level retrieval
6. ColPaliRetriever - Multimodal visual document search
"""

from .base import BaseRetriever, RetrievalResult, RetrievalConfig
from .vector import VectorSearcher
from .text import TextSearcher
from .hybrid import HybridSearcher
from .graphrag import GraphRAGRetriever
from .raptor import RAPTORRetriever
from .colpali import ColPaliRetriever, ColPaliRetrievalResult
from .reranker import Reranker
from .pipeline import RetrievalPipeline, RetrievalStrategy

__all__ = [
    "BaseRetriever",
    "RetrievalResult",
    "RetrievalConfig",
    "VectorSearcher",
    "TextSearcher",
    "HybridSearcher",
    "GraphRAGRetriever",
    "RAPTORRetriever",
    "ColPaliRetriever",
    "ColPaliRetrievalResult",
    "Reranker",
    "RetrievalPipeline",
    "RetrievalStrategy",
]
