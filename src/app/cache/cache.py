from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypedDict


@dataclass(frozen=True)
class CacheStats:
  cached_queries: int


class CacheEntry(TypedDict):
  query: str
  thread_id: str
  response: str
  timestamp: int
  hits: int


class Cache(ABC):
  @abstractmethod
  async def set(self, query: str, thread_id: str, response: str) -> None: ...

  @abstractmethod
  async def get(
    self, query: str, thread_id: str, return_full: bool = False
  ) -> str | CacheEntry | None: ...

  @abstractmethod
  async def get_stats(self) -> CacheStats: ...

  @abstractmethod
  def purge_expired(self) -> int: ...

  @abstractmethod
  async def clear(self) -> None: ...

  @abstractmethod
  def __len__(self) -> int: ...

  def _is_expired(self, entry: CacheEntry, now: float, ttl_seconds: float) -> bool:
    return now - entry['timestamp'] >= ttl_seconds
