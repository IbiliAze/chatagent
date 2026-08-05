"""In-process LRU cache keyed on a hash of the normalised query."""

import hashlib
from collections import OrderedDict
from threading import Lock
from time import time

from app.cache.cache import Cache, CacheEntry, CacheStats
from core.config.settings import get_settings

type CacheStore = OrderedDict[str, CacheEntry]


class HashCache(Cache):
    """Cache responses with hash matching.

    Bounded by both age and size: entries expire after `ttl_seconds` and the
    least recently used entry is dropped once `max_entries` is exceeded. Without
    both, every unique query would pin its full response for the process lifetime.
    """

    def __init__(
        self,
        ttl_seconds: int | None = None,
        max_entries: int | None = None,
    ) -> None:
        settings = get_settings()
        self.ttl_seconds = (
            ttl_seconds if ttl_seconds is not None else settings.cache_ttl_seconds
        )
        self.max_entries = (
            max_entries if max_entries is not None else settings.cache_max_entries
        )
        self.cache: CacheStore = OrderedDict()
        # get() mutates (LRU reorder, hit count), so it is not safe to run
        # concurrently from the threadpool that serves sync endpoints.
        self._lock = Lock()

    @staticmethod
    def _hash_query(query: str) -> str:
        """Create a hash of normalised query"""
        normalised = query.lower().strip()
        return hashlib.md5(normalised.encode(), usedforsecurity=False).hexdigest()

    async def get(
        self, query: str, thread_id: str, return_full: bool = False
    ) -> str | CacheEntry | None:
        """Get cached response if the same query exists and has not expired"""
        query_hash = self._hash_query(query)

        with self._lock:
            entry = self.cache.get(query_hash)

            if entry is None:
                return None

            if entry['thread_id'] != thread_id:
                return None

            if self._is_expired(entry, int(time()), self.ttl_seconds):
                del self.cache[query_hash]
                return None

            entry['hits'] += 1
            self.cache.move_to_end(query_hash)
            return entry if return_full else entry['response']

    async def set(self, query: str, thread_id: str, response: str) -> None:
        """Cache a response, evicting the least recently used entry if full."""
        query_hash = self._hash_query(query)

        with self._lock:
            self.cache[query_hash] = {
                'query': query,
                'thread_id': thread_id,
                'response': response,
                'timestamp': int(time()),
                'hits': 0,
            }
            self.cache.move_to_end(query_hash)

            # Evict the least frequently used
            while len(self.cache) > self.max_entries:
                self.cache.popitem(last=False)

    def purge_expired(self) -> int:
        """Drop every expired entry, returning how many were removed."""
        now = int(time())

        with self._lock:
            expired = [
                key
                for key, entry in self.cache.items()
                if self._is_expired(entry, now, self.ttl_seconds)
            ]
            for key in expired:
                del self.cache[key]

        return len(expired)

    async def get_stats(self) -> CacheStats:
        """Count only live entries, since expired ones are removed lazily on read"""
        now = int(time())

        with self._lock:
            live = sum(
                1
                for entry in self.cache.values()
                if not self._is_expired(entry, now, self.ttl_seconds)
            )

        return CacheStats(cached_queries=live)

    async def clear(self):
        """Clear the cache"""
        self.cache.clear()

    def __len__(self):
        """Get number of items in the cache."""
        return len(self.cache)
