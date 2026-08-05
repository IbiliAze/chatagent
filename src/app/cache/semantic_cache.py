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

from app.cache.cache import Cache, CacheEntry, CacheStats
from core.config.settings import get_settings
from core.logging.logger import logger


class SemanticCache(Cache):
    """Cache responses with semantic similarity matching"""

    def __init__(
        self,
        vectorstore: OpenSearchVectorSearch,
        score_threshold: float | None = None,
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
        self.score_threshold = (
            score_threshold
            if score_threshold is not None
            else self.settings.cache_score_threshold
        )
        self.vectorstore = vectorstore
        self.increment_entries_queue: deque[str] = deque(maxlen=1000)

    async def get(
        self, query: str, thread_id: str, return_full: bool = False
    ) -> str | CacheEntry | None:
        """Find the one most semantically similar query
        WHERE thread_id equals the current thread
        AND timestamp is newer than the expiry cutoff"""

        expires_after = int(time()) - self.ttl_seconds

        try:
            results = await self.vectorstore.asimilarity_search_with_relevance_scores(
                query=self._normalise_query(query),
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

        # Raw OpenSearch kNN score, not a cosine similarity: the score formula
        # depends on the index engine and space_type, so comparing it directly
        # avoids hardcoding a conversion that silently breaks if either changes.
        # Monotonic in cosine either way, so it is still a valid cutoff.
        if score < self.score_threshold:
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

        return entry if return_full else entry['response']

    async def set(
        self,
        query: str,
        thread_id: str,
        response: str,
    ) -> None:
        """Cache a response, evicting least-frequently-used entries if full."""

        normalised_query = self._normalise_query(query=query)

        document_id = self._get_cache_id(normalised_query, thread_id)

        entry: CacheEntry = {
            'hits': 0,
            'query': normalised_query,
            'response': response,
            'thread_id': thread_id,
            'timestamp': int(time()),
        }

        await self.vectorstore.aadd_texts(  # type: ignore[reportUnknownMemberType]
            texts=[normalised_query],
            metadatas=[cast(dict[str, Any], entry)],
            ids=[document_id],
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
                'refresh': 'true',
            },
        )

        evicted_count = self._evict_entries()

        return int(response.get('deleted', 0)) + evicted_count

    def _evict_entries(self) -> int:
        """Evict overflowed entries, down to a target below max_entries.

        Evicting to a target rather than exactly to max_entries means a cache
        sitting at capacity does not re-trigger eviction on every pass, and lets
        concurrent passes from other workers overlap without over-deleting.
        """
        index = self.settings.opensearch_cache_index

        response = self.vectorstore.client.count(index=index)
        current_count = response.get('count', 0)

        if current_count <= self.max_entries:
            return 0

        target = int(self.max_entries * self.settings.cache_eviction_target_ratio)
        overflowed = current_count - target

        if overflowed <= 0:
            return 0

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

    def _normalise_query(self, query: str) -> str:
        """Normalise query."""
        return ' '.join(query.casefold().split())

    def _get_cache_id(self, query: str, thread_id: str) -> str:
        """Get cache entry ID by normalising the query."""
        normalised_query = self._normalise_query(query=query)
        value = f'{thread_id}:{normalised_query}'
        return sha256(value.encode()).hexdigest()

    def _drain_queue(self) -> list[str]:
        """Take every queued document id, leaving the queue empty.

        Draining up front means hits recorded while the flush is in flight simply
        land in the next batch. Reconciling against a copy afterwards would delete
        those late arrivals too, silently losing them.
        """
        entries: list[str] = []

        while self.increment_entries_queue:
            entries.append(self.increment_entries_queue.popleft())

        return entries

    def _increment_hits(self) -> None:
        """Flush pending cache-hit increments to OpenSearch."""
        entries = self._drain_queue()

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
                params={'refresh': 'true'},
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

        if not response.get('errors'):
            return

        # Hit counts are advisory, so a failed increment is logged and dropped
        # rather than requeued.
        for item in response.get('items', []):
            update_result = item.get('update', {})

            if error := update_result.get('error'):
                logger.warning(
                    'Cache entry increment failed',
                    extra={
                        'extra_data': {
                            'document_id': update_result.get('_id'),
                            'status': update_result.get('status'),
                            'error': error,
                        }
                    },
                )

    async def run_maintenance_once(self) -> None:
        """Flush queued hit increments, then purge and evict.

        Ordering matters: eviction sorts on metadata.hits, so the increments have
        to be flushed first or it ranks entries on stale counts.
        """
        await asyncio.to_thread(self._increment_hits)
        await asyncio.to_thread(self.purge_expired)

    async def maintenance(self, interval_seconds: int | None = None) -> None:
        """Run maintenance on a fixed interval until cancelled.

        Deliberately uncoordinated across workers: eviction leaves headroom below
        max_entries, so concurrent passes overlap harmlessly rather than thrashing
        at the boundary. The caller owns the final flush on shutdown.
        """
        interval = (
            interval_seconds
            if interval_seconds is not None
            else self.settings.cache_maintenance_interval_seconds
        )

        while True:
            await asyncio.sleep(interval)

            try:
                await self.run_maintenance_once()
            except Exception:
                # Never let one bad pass kill the loop.
                logger.exception('Cache maintenance pass failed')

    async def clear(self):
        """Delete all entries from the cache index."""
        await self.vectorstore.async_client.delete_by_query(
            index=self.settings.opensearch_cache_index,
            body={
                'query': {
                    'match_all': {},
                },
            },
            params={
                'conflicts': 'proceed',
                'refresh': 'true',
            },
        )

    def __len__(self):
        """Get number of items in the cache."""
        response = self.vectorstore.client.count(
            index=self.settings.opensearch_cache_index
        )
        return cast(int, response['count'])
