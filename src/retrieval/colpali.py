"""ColPali retriever for multimodal document search."""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.clients.mongodb import MongoDBClient
from src.clients.colpali import ColPaliClient, ColPaliConfig, MockColPaliClient
from src.ingestion.embeddings.colpali import ColPaliPageEmbedding
from .base import BaseRetriever, RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class ColPaliRetrievalResult(RetrievalResult):
    """Retrieval result with ColPali-specific fields."""

    page_num: int = 0
    image_size: tuple[int, int] = (0, 0)
    has_images: bool = False


class ColPaliRetriever(BaseRetriever):
    """Retriever using ColPali multimodal embeddings.

    Searches document pages by visual content using late interaction.
    No OCR needed - searches images directly.
    """

    def __init__(
        self,
        mongodb_client: MongoDBClient,
        colpali_client: ColPaliClient | None = None,
        collection_name: str = "documents",
        use_mock: bool = False,
    ):
        """Initialize ColPali retriever.

        Args:
            mongodb_client: MongoDB client for page embedding storage
            colpali_client: ColPali client for query embedding
            collection_name: Collection storing page embeddings
            use_mock: Use mock client for testing
        """
        self.mongodb = mongodb_client
        self.collection_name = collection_name

        if colpali_client:
            self.colpali = colpali_client
        elif use_mock:
            self.colpali = MockColPaliClient()
        else:
            self.colpali = ColPaliClient()

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        document_ids: list[str] | None = None,
        **kwargs,
    ) -> list[ColPaliRetrievalResult]:
        """Retrieve pages matching query visually.

        Args:
            query: Search query
            top_k: Number of results
            document_ids: Filter to specific documents
            **kwargs: Additional options

        Returns:
            List of ColPaliRetrievalResult
        """
        # Generate query embedding
        query_embedding = self.colpali.embed_query(query)

        # Load page embeddings from MongoDB
        page_embeddings = await self._load_page_embeddings(document_ids)

        if not page_embeddings:
            logger.warning("No page embeddings found in database")
            return []

        # Compute MaxSim scores
        scored_pages = []
        for page_emb in page_embeddings:
            score = self.colpali.compute_similarity(
                query_embedding, page_emb.embedding
            )
            scored_pages.append((page_emb, score))

        # Sort by score descending
        scored_pages.sort(key=lambda x: x[1], reverse=True)

        # Take top_k results
        results = []
        for page_emb, score in scored_pages[:top_k]:
            results.append(
                ColPaliRetrievalResult(
                    chunk_id=f"{page_emb.document_id}_page_{page_emb.page_num}",
                    document_id=page_emb.document_id,
                    content=f"Page {page_emb.page_num + 1} of document",
                    score=score,
                    metadata={
                        "type": "page",
                        "has_text": page_emb.has_text,
                        "has_images": page_emb.has_images,
                        "image_width": page_emb.image_size[0],
                        "image_height": page_emb.image_size[1],
                    },
                    page_num=page_emb.page_num,
                    image_size=page_emb.image_size,
                    has_images=page_emb.has_images,
                )
            )

        logger.info(
            f"ColPali retrieved {len(results)} pages for query: {query[:50]}..."
        )

        return results

    async def _load_page_embeddings(
        self,
        document_ids: list[str] | None = None,
    ) -> list[ColPaliPageEmbedding]:
        """Load page embeddings from MongoDB.

        Args:
            document_ids: Filter to specific documents

        Returns:
            List of ColPaliPageEmbedding objects
        """
        collection = self.mongodb.db[self.collection_name]

        # Build query
        query: dict[str, Any] = {"pages.colpali_embedding": {"$exists": True}}
        if document_ids:
            query["document_id"] = {"$in": document_ids}

        # Fetch documents with page embeddings
        cursor = collection.find(query)
        documents = await cursor.to_list(length=1000)

        embeddings = []
        for doc in documents:
            document_id = doc.get("document_id", str(doc["_id"]))

            for page_data in doc.get("pages", []):
                if "colpali_embedding" not in page_data:
                    continue

                page_emb = ColPaliPageEmbedding.from_list(
                    document_id=document_id,
                    page_num=page_data.get("page_num", 0),
                    embedding_list=page_data["colpali_embedding"],
                    image_size=(
                        page_data.get("image_width", 0),
                        page_data.get("image_height", 0),
                    ),
                    has_text=page_data.get("has_text", True),
                    has_images=page_data.get("has_images", False),
                )
                embeddings.append(page_emb)

        logger.info(f"Loaded {len(embeddings)} page embeddings from database")
        return embeddings

    async def retrieve_from_embeddings(
        self,
        query: str,
        page_embeddings: list[ColPaliPageEmbedding],
        top_k: int = 10,
    ) -> list[ColPaliRetrievalResult]:
        """Retrieve from pre-loaded embeddings.

        Args:
            query: Search query
            page_embeddings: Pre-loaded page embeddings
            top_k: Number of results

        Returns:
            List of ColPaliRetrievalResult
        """
        query_embedding = self.colpali.embed_query(query)

        scored_pages = []
        for page_emb in page_embeddings:
            score = self.colpali.compute_similarity(
                query_embedding, page_emb.embedding
            )
            scored_pages.append((page_emb, score))

        scored_pages.sort(key=lambda x: x[1], reverse=True)

        results = []
        for page_emb, score in scored_pages[:top_k]:
            results.append(
                ColPaliRetrievalResult(
                    chunk_id=f"{page_emb.document_id}_page_{page_emb.page_num}",
                    document_id=page_emb.document_id,
                    content=f"Page {page_emb.page_num + 1} of document",
                    score=score,
                    metadata={
                        "type": "page",
                        "has_text": page_emb.has_text,
                        "has_images": page_emb.has_images,
                    },
                    page_num=page_emb.page_num,
                    image_size=page_emb.image_size,
                    has_images=page_emb.has_images,
                )
            )

        return results

    async def hybrid_retrieve(
        self,
        query: str,
        text_results: list[RetrievalResult],
        top_k: int = 10,
        visual_weight: float = 0.4,
    ) -> list[RetrievalResult]:
        """Combine visual and text retrieval results.

        Args:
            query: Search query
            text_results: Results from text/vector retrieval
            top_k: Number of combined results
            visual_weight: Weight for visual scores (0-1)

        Returns:
            Combined and reranked results
        """
        # Get visual results
        visual_results = await self.retrieve(query, top_k=top_k * 2)

        # Build score map by document
        doc_scores: dict[str, dict[str, float]] = {}

        # Add text scores
        for result in text_results:
            doc_id = result.document_id
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"text": 0, "visual": 0, "result": result}
            doc_scores[doc_id]["text"] = max(
                doc_scores[doc_id]["text"], result.score
            )

        # Add visual scores
        for result in visual_results:
            doc_id = result.document_id
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"text": 0, "visual": 0, "result": result}
            doc_scores[doc_id]["visual"] = max(
                doc_scores[doc_id]["visual"], result.score
            )

        # Combine scores
        combined = []
        text_weight = 1 - visual_weight

        for doc_id, scores in doc_scores.items():
            combined_score = (
                text_weight * scores["text"] +
                visual_weight * scores["visual"]
            )
            result = scores["result"]
            result.score = combined_score
            combined.append(result)

        # Sort by combined score
        combined.sort(key=lambda x: x.score, reverse=True)

        return combined[:top_k]
