import hashlib
from dataclasses import dataclass
from typing import TypedDict

from langchain_openai import ChatOpenAI


class CacheEntry(TypedDict):
  query: str
  response: str


@dataclass(frozen=True)
class CacheStats:
  cached_queries: int


type Cache = dict[str, CacheEntry]


class SemanticCache:
  """Cache responses with semantic similarity matching"""

  def __init__(self, similarity_threshold: float = 0.9) -> None:
    self.cache: Cache = {}
    self.similarity_threshold = similarity_threshold
    self.embedder = ChatOpenAI(model='gpt-4o-mini')

  @staticmethod
  def _embed_query(query: str) -> str | None:
    raise NotImplementedError

  @staticmethod
  def _hash_query(query: str) -> str:
    """Create a hash of normalised query"""
    normalised = query.lower().strip()
    return hashlib.md5(normalised.encode()).hexdigest()

  def get(self, query: str) -> str | None:
    """Get cached response if similar query exists"""
    query_hash = self._hash_query(query)

    if query_hash in self.cache:
      return self.cache[query_hash]['response']

    return None

  def set(self, query: str, response: str) -> None:
    """Cache a response"""
    query_hash = self._hash_query(query)
    self.cache[query_hash] = {'query': query, 'response': response}

  def get_stats(self) -> CacheStats:
    return CacheStats(cached_queries=len(self.cache))
