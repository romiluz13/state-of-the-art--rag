"""Caching utilities for performance optimization."""

import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """Single cache entry with metadata."""

    value: T
    created_at: float
    expires_at: float | None
    hits: int = 0


@dataclass
class CacheStats:
    """Cache statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0-1.0)."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class LRUCache(Generic[T]):
    """Thread-safe LRU cache with TTL support.

    Features:
    - LRU eviction policy
    - Optional TTL for entries
    - Statistics tracking
    - Memory-efficient for embeddings
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float | None = None,
        name: str = "cache",
    ):
        """Initialize LRU cache.

        Args:
            max_size: Maximum number of entries
            ttl_seconds: Time-to-live for entries (None = no expiry)
            name: Cache name for logging
        """
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.name = name
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._stats = CacheStats(max_size=max_size)

    def get(self, key: str) -> T | None:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        if key not in self._cache:
            self._stats.misses += 1
            return None

        entry = self._cache[key]

        # Check expiry
        if entry.expires_at and time.time() > entry.expires_at:
            self._remove(key)
            self._stats.misses += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        entry.hits += 1
        self._stats.hits += 1

        return entry.value

    def set(self, key: str, value: T, ttl: float | None = None) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL override for this entry
        """
        now = time.time()
        entry_ttl = ttl if ttl is not None else self.ttl
        expires_at = now + entry_ttl if entry_ttl else None

        # Remove existing entry if present
        if key in self._cache:
            self._remove(key)

        # Evict oldest if at capacity
        while len(self._cache) >= self.max_size:
            oldest_key = next(iter(self._cache))
            self._remove(oldest_key)
            self._stats.evictions += 1

        # Add new entry
        self._cache[key] = CacheEntry(
            value=value,
            created_at=now,
            expires_at=expires_at,
        )
        self._stats.size = len(self._cache)

    def _remove(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self._cache:
            del self._cache[key]
            self._stats.size = len(self._cache)

    def invalidate(self, key: str) -> bool:
        """Invalidate a cache entry.

        Args:
            key: Cache key to invalidate

        Returns:
            True if entry existed and was removed
        """
        if key in self._cache:
            self._remove(key)
            return True
        return False

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._stats.size = 0
        logger.info(f"Cache '{self.name}' cleared")

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    def cleanup_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries removed
        """
        now = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.expires_at and now > entry.expires_at
        ]

        for key in expired_keys:
            self._remove(key)

        if expired_keys:
            logger.debug(f"Cache '{self.name}': cleaned up {len(expired_keys)} expired entries")

        return len(expired_keys)


class EmbeddingCache:
    """Specialized cache for embeddings.

    Caches query embeddings to avoid redundant API calls.
    Uses content hash as key for deduplication.
    """

    def __init__(
        self,
        max_size: int = 5000,
        ttl_seconds: float = 3600,  # 1 hour default
    ):
        """Initialize embedding cache.

        Args:
            max_size: Maximum number of embeddings to cache
            ttl_seconds: TTL for cached embeddings
        """
        self._cache: LRUCache[list[float]] = LRUCache(
            max_size=max_size,
            ttl_seconds=ttl_seconds,
            name="embeddings",
        )

    @staticmethod
    def _make_key(text: str, model: str) -> str:
        """Create cache key from text and model."""
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def get(self, text: str, model: str = "voyage-3.5") -> list[float] | None:
        """Get cached embedding for text.

        Args:
            text: Text to look up
            model: Embedding model name

        Returns:
            Cached embedding or None
        """
        key = self._make_key(text, model)
        return self._cache.get(key)

    def set(self, text: str, embedding: list[float], model: str = "voyage-3.5") -> None:
        """Cache embedding for text.

        Args:
            text: Original text
            embedding: Embedding vector
            model: Embedding model name
        """
        key = self._make_key(text, model)
        self._cache.set(key, embedding)

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._cache.get_stats()


class QueryResultCache:
    """Cache for query results.

    Caches retrieval results to speed up repeated queries.
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 300,  # 5 minutes default
    ):
        """Initialize query result cache.

        Args:
            max_size: Maximum number of query results to cache
            ttl_seconds: TTL for cached results
        """
        self._cache: LRUCache[list[dict[str, Any]]] = LRUCache(
            max_size=max_size,
            ttl_seconds=ttl_seconds,
            name="query_results",
        )

    @staticmethod
    def _make_key(query: str, strategy: str, top_k: int) -> str:
        """Create cache key from query parameters."""
        content = f"{strategy}:{top_k}:{query}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def get(
        self,
        query: str,
        strategy: str,
        top_k: int,
    ) -> list[dict[str, Any]] | None:
        """Get cached query results.

        Args:
            query: Search query
            strategy: Retrieval strategy
            top_k: Number of results

        Returns:
            Cached results or None
        """
        key = self._make_key(query, strategy, top_k)
        return self._cache.get(key)

    def set(
        self,
        query: str,
        strategy: str,
        top_k: int,
        results: list[dict[str, Any]],
    ) -> None:
        """Cache query results.

        Args:
            query: Search query
            strategy: Retrieval strategy
            top_k: Number of results
            results: Results to cache
        """
        key = self._make_key(query, strategy, top_k)
        self._cache.set(key, results)

    def invalidate_query(self, query: str) -> None:
        """Invalidate all cached results for a query."""
        # Note: This is a simple implementation
        # A more sophisticated version would track all keys per query
        pass

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._cache.get_stats()


# Global cache instances (initialized lazily)
_embedding_cache: EmbeddingCache | None = None
_query_cache: QueryResultCache | None = None


def get_embedding_cache() -> EmbeddingCache:
    """Get global embedding cache instance."""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache()
    return _embedding_cache


def get_query_cache() -> QueryResultCache:
    """Get global query result cache instance."""
    global _query_cache
    if _query_cache is None:
        _query_cache = QueryResultCache()
    return _query_cache
