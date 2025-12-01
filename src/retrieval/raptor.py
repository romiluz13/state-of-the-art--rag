"""RAPTOR hierarchical retrieval - December 2025: Now uses LeanRAG.

This module provides backward compatibility for RAPTOR imports.
The actual implementation is in leanrag.py which provides:
- Bottom-up retrieval (entities → aggregation → summaries)
- 46% redundancy reduction
- 97.3% win rate vs naive retrieval

December 2025: RAPTOR (ICLR 2024) replaced by LeanRAG (AAAI 2026).
"""

# December 2025: Import from LeanRAG (the new SOTA implementation)
from .leanrag import (
    LeanRAGRetriever,
    RAPTORRetriever,  # Backward compatibility alias
)

# Re-export for backward compatibility
__all__ = [
    "RAPTORRetriever",  # Alias → LeanRAGRetriever
    "LeanRAGRetriever",  # December 2025 SOTA
]
