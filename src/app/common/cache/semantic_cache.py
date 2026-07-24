from langchain_community.vectorstores import OpenSearchVectorSearch

from app.common.cache.cache import Cache, CacheStats
from core.config.settings import get_settings


class SemanticCache(Cache):
  """Cache responses with semantic similarity matching"""

  def __init__(
    self, vectorstore: OpenSearchVectorSearch, similarity_threshold: float = 0.9
  ) -> None:
    self.settings = get_settings()
    self.vectorstore = vectorstore
    self.similarity_threshold = similarity_threshold

  def set(self, query: str, response: str) -> None:
    """Cache a response"""
    self.vectorstore.add_texts([query], [{'response': response}])  # type: ignore[reportUnknownMemberType]

  def get(self, query: str) -> str | None:
    """Get cached response if a similar query exists"""
    results = self.vectorstore.similarity_search_with_relevance_scores(query, k=1)
    if not results:
      return None
    doc, score = results[0]
    if score < self.similarity_threshold:
      return None
    return doc.metadata['response']

  def get_stats(self) -> CacheStats:
    if not self.vectorstore.client.indices.exists(
      index=self.settings.opensearch_cache_index
    ):
      return CacheStats(cached_queries=0)
    response = self.vectorstore.client.count(index=self.settings.opensearch_cache_index)
    return CacheStats(cached_queries=response['count'])
