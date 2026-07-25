from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class CacheStats:
  cached_queries: int


class Cache(ABC):
  @abstractmethod
  def set(self, query: str, response: str) -> None: ...

  @abstractmethod
  def get(self, query: str) -> str | None: ...

  @abstractmethod
  def get_stats(self) -> CacheStats: ...
