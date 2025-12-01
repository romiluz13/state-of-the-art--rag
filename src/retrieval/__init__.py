"""Retrieval module for SOTA RAG system.

December 2025 Update: 9 retrieval strategies for true SOTA status.

Implements retrieval strategies:
1. VectorSearcher - $vectorSearch with binary quantization
2. TextSearcher - $search for BM25
3. HybridSearcher - $rankFusion combining vector + BM25
4. GraphRAGRetriever - $graphLookup + hybrid (Dec 2025: RRF + rerank)
5. LeanRAGRetriever - December 2025 SOTA (replaces RAPTOR)
6. ColPaliRetriever - Multimodal visual document search (ColQwen2-v1.0)
7. MCTSRetriever - December 2025: Multi-hop reasoning with MCTS (+20%)
8. RAPTORRetriever - Backward compatibility alias for LeanRAGRetriever
9. Multi-Query - Query decomposition (via pipeline)
"""

from .base import BaseRetriever, RetrievalResult, RetrievalConfig
from .vector import VectorSearcher
from .text import TextSearcher
from .hybrid import HybridSearcher
from .graphrag import GraphRAGRetriever
from .leanrag import LeanRAGRetriever, RAPTORRetriever  # Dec 2025: LeanRAG is SOTA
from .colpali import ColPaliRetriever, ColPaliRetrievalResult
from .mcts import MCTSRetriever, MCTSConfig  # Dec 2025: MCTS-RAG for multi-hop
from .reranker import Reranker
from .pipeline import RetrievalPipeline, RetrievalStrategy

__all__ = [
    # Base classes
    "BaseRetriever",
    "RetrievalResult",
    "RetrievalConfig",
    # Core retrievers
    "VectorSearcher",
    "TextSearcher",
    "HybridSearcher",
    # Advanced retrievers (December 2025 SOTA)
    "GraphRAGRetriever",
    "LeanRAGRetriever",  # December 2025 SOTA (was RAPTOR)
    "RAPTORRetriever",   # Backward compatibility alias
    "ColPaliRetriever",
    "ColPaliRetrievalResult",
    "MCTSRetriever",     # December 2025: Multi-hop reasoning
    "MCTSConfig",
    # Pipeline components
    "Reranker",
    "RetrievalPipeline",
    "RetrievalStrategy",
]
