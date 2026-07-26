import asyncio
from collections import Counter, deque
from hashlib import sha256
from time import time
from typing import Any, cast

from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy.exceptions import (
  NotFoundError,
  OpenSearchException,
  TransportError,
)

from app.common.cache.cache import Cache, CacheEntry, CacheStats
from core.config.settings import get_settings
from core.logging.logger import logger


class SemanticCache(Cache):
  """Cache responses with semantic similarity matching"""

  def __init__(
    self,
    vectorstore: OpenSearchVectorSearch,
    similarity_threshold: float = 0.95,
    ttl_seconds: int | None = None,
    max_entries: int | None = None,
  ) -> None:
    self.settings = get_settings()
    self.ttl_seconds = (
      ttl_seconds if ttl_seconds is not None else self.settings.cache_ttl_seconds
    )
    self.max_entries = (
      max_entries if max_entries is not None else self.settings.cache_max_entries
    )
    self.vectorstore = vectorstore
    self.similarity_threshold = similarity_threshold
    self.increment_entries_queue: deque[str] = deque(maxlen=1000)

  async def get(self, query: str, thread_id: str) -> str | None:
    """Find the one most semantically similar query
    WHERE thread_id equals the current thread
    AND timestamp is newer than the expiry cutoff"""

    expires_after = int(time()) - self.ttl_seconds

    try:
      results = await self.vectorstore.asimilarity_search_with_relevance_scores(
        query=query,
        k=1,
        efficient_filter={
          'bool': {
            'filter': [
              {
                'term': {
                  'metadata.thread_id': thread_id,
                }
              },
              {
                'range': {
                  'metadata.timestamp': {
                    'gte': expires_after,
                  }
                }
              },
            ]
          }
        },
      )
    except NotFoundError:
      return None

    if not results:
      return None

    doc, score = results[0]
    if score < self.similarity_threshold:
      return None

    if doc.id is None:
      return None

    entry = cast(CacheEntry, doc.metadata)

    required_keys = {
      'hits',
      'query',
      'response',
      'thread_id',
      'timestamp',
    }

    if not required_keys.issubset(entry):
      return None

    self.increment_entries_queue.append(doc.id)

    return entry['response']

  async def set(
    self,
    query: str,
    thread_id: str,
    response: str,
  ) -> None:
    """Cache a response, evicting least-frequently-used entries if full."""

    document_id = self._get_cache_id(query, thread_id)

    entry: CacheEntry = {
      'hits': 0,
      'query': query,
      'response': response,
      'thread_id': thread_id,
      'timestamp': int(time()),
    }

    await self.vectorstore.aadd_texts(  # type: ignore[reportUnknownMemberType]
      texts=[query], metadatas=[cast(dict[str, Any], entry)], ids=[document_id]
    )

  async def get_stats(self) -> CacheStats:
    if not await self.vectorstore.async_client.indices.exists(
      index=self.settings.opensearch_cache_index
    ):
      return CacheStats(cached_queries=0)
    response = await self.vectorstore.async_client.count(
      index=self.settings.opensearch_cache_index
    )
    return CacheStats(cached_queries=response['count'])

  def purge_expired(self) -> int:
    """Drop every expired entry, returning how many were removed."""
    cutoff = int(time()) - self.ttl_seconds

    response = self.vectorstore.client.delete_by_query(
      index=self.settings.opensearch_cache_index,
      body={
        'query': {'range': {'metadata.timestamp': {'lt': cutoff}}},
      },
      params={
        'conflicts': 'proceed',
        'refresh': True,
      },
    )

    evicted_count = self._evict_entries()

    return int(response.get('deleted', 0)) + evicted_count

  def _evict_entries(self) -> int:
    """Evict overflowed entries."""
    response = self.vectorstore.client.count(index=self.settings.opensearch_cache_index)
    current_count = response.get('count', 0)
    overflowed = current_count - self.max_entries

    if overflowed <= 0:
      return 0

    index = self.settings.opensearch_cache_index

    response = self.vectorstore.client.search(
      index=index,
      body={
        'size': overflowed,
        '_source': False,
        'query': {
          'match_all': {},
        },
        'sort': [
          {
            'metadata.hits': {
              'order': 'asc',
            }
          },
          {
            'metadata.timestamp': {
              'order': 'asc',
            }
          },
        ],
      },
    )

    document_ids = [hit['_id'] for hit in response['hits']['hits']]

    if not document_ids:
      return 0

    self.vectorstore.delete(ids=document_ids)
    return len(document_ids)

  def _get_cache_id(self, query: str, thread_id: str) -> str:
    """Get cache entry ID by normalising the query."""
    normalized_query = ' '.join(query.casefold().split())
    value = f'{thread_id}:{normalized_query}'
    return sha256(value.encode()).hexdigest()

  def _increment_hits(self) -> None:
    """Flush pending cache-hit increments to OpenSearch."""
    entries = self.increment_entries_queue.copy()

    if not entries:
      return

    operations: list[dict[str, object]] = []

    # Combine duplicate IDs into a single update.
    for document_id, increment_by in Counter(entries).items():
      operations.extend(
        [
          {
            'update': {
              '_id': document_id,
              'retry_on_conflict': 3,
            }
          },
          {
            'script': {
              'lang': 'painless',
              'source': """
                if (ctx._source.metadata == null) {
                  ctx._source.metadata = new HashMap();
                }

                if (ctx._source.metadata.hits == null) {
                  ctx._source.metadata.hits = params.increment_by;
                } else {
                  ctx._source.metadata.hits += params.increment_by;
                }
              """,
              'params': {
                'increment_by': increment_by,
              },
            }
          },
        ]
      )

    try:
      response = self.vectorstore.client.bulk(
        index=self.settings.opensearch_cache_index,
        body=operations,
        params={'refresh': True},
      )
    except TransportError:
      logger.exception('Cache entry increment transport error')
      return
    except OpenSearchException:
      logger.exception('Cache entry increment OpenSearch exception')
      return
    except Exception:
      logger.exception('Cache entry increment failed')
      return

    for item in response.get('items', []):
      update_result = item.get('update', {})

      if error := update_result.get('error'):
        logger.warning(
          'Cache entry increment failed',
          extra={
            'document_id': update_result.get('_id'),
            'status': update_result.get('status'),
            'error': error,
          },
        )

      else:
        try:
          doc_id = update_result.get('_id')
          if doc_id is None:
            continue

          else:
            while doc_id in self.increment_entries_queue:
              self.increment_entries_queue.remove(doc_id)

        except Exception:
          logger.error('Failed to remove from increment queue')

    return

  async def maintenance(
    self,
    interval_seconds: int = 60,
  ) -> None:
    """Periodically flush pending hit increments and purge expired entries"""
    try:
      while True:
        await asyncio.sleep(interval_seconds)
        await asyncio.to_thread(self._increment_hits)
        await asyncio.to_thread(self.purge_expired)
    except asyncio.CancelledError:
      # Final flush during application shutdown.
      await asyncio.to_thread(self._increment_hits)
      await asyncio.to_thread(self.purge_expired)
      raise
