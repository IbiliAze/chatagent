"""Abstract cache interface shared by the hash and semantic cache implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypedDict


@dataclass(frozen=True)
class CacheStats:
    """Summary stats reported by a cache implementation."""

    cached_queries: int


class CacheEntry(TypedDict):
    """A single cached query/response pair."""

    query: str
    thread_id: str
    response: str
    timestamp: int
    hits: int


class Cache(ABC):
    """Interface implemented by HashCache and SemanticCache."""

    @abstractmethod
    async def set(self, query: str, thread_id: str, response: str) -> None:
        """Cache a response for the given query and thread."""
        ...

    @abstractmethod
    async def get(
        self, query: str, thread_id: str, return_full: bool = False
    ) -> str | CacheEntry | None:
        """Look up a cached response for the given query and thread."""
        ...

    @abstractmethod
    async def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        ...

    @abstractmethod
    def purge_expired(self) -> int:
        """Drop every expired entry, returning how many were removed."""
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Delete all entries from the cache."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Get number of items in the cache."""
        ...

    def _is_expired(self, entry: CacheEntry, now: float, ttl_seconds: float) -> bool:
        """Check whether a cache entry's timestamp has passed the given TTL."""
        return now - entry['timestamp'] >= ttl_seconds
