"""Tests for caching utilities."""

import time
import pytest

from src.utils.cache import (
    LRUCache,
    CacheStats,
    EmbeddingCache,
    QueryResultCache,
)


class TestLRUCache:
    """Tests for LRUCache."""

    def test_basic_get_set(self):
        """Test basic get and set operations."""
        cache: LRUCache[str] = LRUCache(max_size=10)
        cache.set("key1", "value1")

        assert cache.get("key1") == "value1"
        assert cache.get("nonexistent") is None

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache: LRUCache[int] = LRUCache(max_size=3)

        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)

        # Access "a" to make it recently used (moves to end)
        assert cache.get("a") == 1

        # Add new item, should evict "b" (least recently used)
        cache.set("d", 4)

        # "b" should be evicted (was least recently used)
        assert cache.get("b") is None
        assert cache.get("a") == 1  # Still there
        assert cache.get("c") == 3  # Still there
        assert cache.get("d") == 4  # New item

    def test_ttl_expiry(self):
        """Test TTL-based expiry."""
        cache: LRUCache[str] = LRUCache(max_size=10, ttl_seconds=0.1)
        cache.set("key", "value")

        # Should be present immediately
        assert cache.get("key") == "value"

        # Wait for expiry
        time.sleep(0.15)

        # Should be expired
        assert cache.get("key") is None

    def test_ttl_override(self):
        """Test TTL override for specific entry."""
        cache: LRUCache[str] = LRUCache(max_size=10, ttl_seconds=1.0)

        # Set with shorter TTL
        cache.set("short", "value", ttl=0.1)
        cache.set("long", "value")

        time.sleep(0.15)

        assert cache.get("short") is None  # Expired
        assert cache.get("long") == "value"  # Still valid

    def test_invalidate(self):
        """Test manual invalidation."""
        cache: LRUCache[str] = LRUCache(max_size=10)
        cache.set("key", "value")

        assert cache.invalidate("key") is True
        assert cache.get("key") is None
        assert cache.invalidate("nonexistent") is False

    def test_clear(self):
        """Test clearing the cache."""
        cache: LRUCache[str] = LRUCache(max_size=10)
        cache.set("a", "1")
        cache.set("b", "2")

        cache.clear()

        assert cache.get("a") is None
        assert cache.get("b") is None
        assert cache.get_stats().size == 0

    def test_stats_tracking(self):
        """Test statistics tracking."""
        cache: LRUCache[str] = LRUCache(max_size=10)

        cache.set("key", "value")

        # Miss
        cache.get("nonexistent")

        # Hits
        cache.get("key")
        cache.get("key")

        stats = cache.get_stats()
        assert stats.hits == 2
        assert stats.misses == 1
        assert stats.hit_rate == pytest.approx(2/3, rel=0.01)

    def test_cleanup_expired(self):
        """Test cleanup of expired entries."""
        cache: LRUCache[str] = LRUCache(max_size=10, ttl_seconds=0.1)

        cache.set("a", "1")
        cache.set("b", "2")
        cache.set("c", "3")

        time.sleep(0.15)

        removed = cache.cleanup_expired()
        assert removed == 3
        assert cache.get_stats().size == 0


class TestEmbeddingCache:
    """Tests for EmbeddingCache."""

    def test_cache_embedding(self):
        """Test caching an embedding."""
        cache = EmbeddingCache(max_size=100)

        embedding = [0.1, 0.2, 0.3]
        cache.set("test query", embedding)

        result = cache.get("test query")
        assert result == embedding

    def test_model_specific_caching(self):
        """Test that different models have different cache entries."""
        cache = EmbeddingCache(max_size=100)

        embedding1 = [0.1, 0.2]
        embedding2 = [0.3, 0.4]

        cache.set("query", embedding1, model="voyage-3")
        cache.set("query", embedding2, model="voyage-3.5")

        assert cache.get("query", model="voyage-3") == embedding1
        assert cache.get("query", model="voyage-3.5") == embedding2

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = EmbeddingCache(max_size=100)

        assert cache.get("nonexistent") is None


class TestQueryResultCache:
    """Tests for QueryResultCache."""

    def test_cache_results(self):
        """Test caching query results."""
        cache = QueryResultCache(max_size=100)

        results = [
            {"chunk_id": "1", "content": "test", "score": 0.9},
            {"chunk_id": "2", "content": "test2", "score": 0.8},
        ]

        cache.set("test query", "hybrid", 10, results)

        cached = cache.get("test query", "hybrid", 10)
        assert cached == results

    def test_different_strategies(self):
        """Test different strategies have different cache entries."""
        cache = QueryResultCache(max_size=100)

        results1 = [{"chunk_id": "1"}]
        results2 = [{"chunk_id": "2"}]

        cache.set("query", "hybrid", 10, results1)
        cache.set("query", "graphrag", 10, results2)

        assert cache.get("query", "hybrid", 10) == results1
        assert cache.get("query", "graphrag", 10) == results2

    def test_different_top_k(self):
        """Test different top_k values have different cache entries."""
        cache = QueryResultCache(max_size=100)

        results5 = [{"n": i} for i in range(5)]
        results10 = [{"n": i} for i in range(10)]

        cache.set("query", "hybrid", 5, results5)
        cache.set("query", "hybrid", 10, results10)

        assert len(cache.get("query", "hybrid", 5)) == 5
        assert len(cache.get("query", "hybrid", 10)) == 10


class TestCacheStats:
    """Tests for CacheStats."""

    def test_hit_rate_no_access(self):
        """Test hit rate with no accesses."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = CacheStats(hits=7, misses=3)
        assert stats.hit_rate == 0.7

    def test_all_hits(self):
        """Test 100% hit rate."""
        stats = CacheStats(hits=100, misses=0)
        assert stats.hit_rate == 1.0

    def test_all_misses(self):
        """Test 0% hit rate."""
        stats = CacheStats(hits=0, misses=50)
        assert stats.hit_rate == 0.0
