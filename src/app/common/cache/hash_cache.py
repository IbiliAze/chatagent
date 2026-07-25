import hashlib
from typing import TypedDict

from app.common.cache.cache import Cache, CacheStats


class CacheEntry(TypedDict):
  query: str
  response: str


type CacheStore = dict[str, CacheEntry]


class HashCache(Cache):
  """Cache responses with hash matching"""

  def __init__(self, similarity_threshold: float = 0.9) -> None:
    self.cache: CacheStore = {}
    self.similarity_threshold = similarity_threshold

  @staticmethod
  def _hash_query(query: str) -> str:
    """Create a hash of normalised query"""
    normalised = query.lower().strip()
    return hashlib.md5(normalised.encode()).hexdigest()

  def get(self, query: str):
    """Get cached response if similar query exists"""
    query_hash = self._hash_query(query)

    if query_hash in self.cache:
      return self.cache[query_hash]['response']

    return None

  def set(self, query: str, response: str):
    """Cache a response"""
    query_hash = self._hash_query(query)
    self.cache[query_hash] = {'query': query, 'response': response}

  def get_stats(self):
    return CacheStats(cached_queries=len(self.cache))
