"""Integration tests run against both HashCache and SemanticCache against a real OpenSearch."""

from collections.abc import AsyncIterator
from typing import cast

import pytest
from dotenv import load_dotenv
from langchain_community.vectorstores import OpenSearchVectorSearch

from app.cache.cache import Cache
from app.cache.hash_cache import HashCache
from app.cache.semantic_cache import SemanticCache
from core.store.vectorstore.opensearch import OpenSearch

load_dotenv()

pytestmark = pytest.mark.integration


@pytest.fixture
def vector_store() -> OpenSearchVectorSearch:
    """Provision a real OpenSearch cache index and return its vector store."""
    opensearch = OpenSearch()
    opensearch.provision_indexes(embedding_dimension=1536)
    return opensearch.cache_vectorstore


@pytest.fixture
def semantic_cache(vector_store: OpenSearchVectorSearch) -> SemanticCache:
    """Build a SemanticCache backed by the provisioned OpenSearch index."""
    return SemanticCache(
        vectorstore=vector_store,  # pyright: ignore[reportArgumentType]
        score_threshold=0.95,
        ttl_seconds=300,
        max_entries=3,
    )


@pytest.fixture
def hash_cache() -> HashCache:
    """Build a fresh in-process HashCache."""
    return HashCache(ttl_seconds=300, max_entries=3)


@pytest.fixture(params=['semantic_cache', 'hash_cache'])
async def cache(request: pytest.FixtureRequest) -> AsyncIterator[Cache]:
    """Run every test in this module against both cache implementations.

    parametrize takes plain values, so listing the fixture functions there would
    hand the test the function objects rather than the caches they build.
    getfixturevalue is what resolves a fixture by name at run time.

    A fresh HashCache is built per test, but SemanticCache shares one OpenSearch
    index across the whole run, so it has to be emptied on the way in.
    """
    cache = cast(Cache, request.getfixturevalue(request.param))
    await cache.clear()
    yield cache


class TestLookup:
    """Basic set/get behaviour shared by both cache implementations."""

    thread_id: str = 'thrd-1'

    async def test_returns_cached_response(self, cache: Cache) -> None:
        """A cached response is returned for the same query and thread."""
        await cache.set('what is my balance?', self.thread_id, 'your balance is 10')

        assert (
            await cache.get('what is my balance?', self.thread_id)
            == 'your balance is 10'
        )

    async def test_normalises_case_and_whitespace(self, cache: Cache) -> None:
        """Lookups ignore case and surrounding whitespace differences in the query."""
        await cache.set('What Is My Balance?', self.thread_id, 'your balance is 10')

        assert (
            await cache.get(
                '  what is my balance?  ',
                self.thread_id,
            )
            == 'your balance is 10'
        )

    async def test_returns_none_when_missing(self, cache: Cache) -> None:
        """A query that was never cached returns None."""
        assert (
            await cache.get(
                'never asked',
                self.thread_id,
            )
            is None
        )

    async def test_overwrites_existing_response(self, cache: Cache) -> None:
        """Setting the same query again replaces the previous response."""
        await cache.set('query', self.thread_id, 'first')
        await cache.set('query', self.thread_id, 'second')

        assert (
            await cache.get(
                'query',
                self.thread_id,
            )
            == 'second'
        )
        assert (await cache.get_stats()).cached_queries == 1


class TestExpiry:
    """TTL-based expiry behaviour, exercised against HashCache directly."""

    thread_id: str = 'thrd-1'

    async def test_expired_entry_is_a_miss(self) -> None:
        """A read after the TTL has elapsed returns None."""
        cache = HashCache(ttl_seconds=0, max_entries=10)
        await cache.set('query', self.thread_id, 'response')

        assert await cache.get('query', self.thread_id) is None

    async def test_expired_entry_is_dropped_on_read(self) -> None:
        """Reading an expired entry also removes it from the cache."""
        cache = HashCache(ttl_seconds=0, max_entries=10)
        await cache.set('query', self.thread_id, 'response')
        await cache.get('query', self.thread_id)

        assert len(cache.cache) == 0

    async def test_purge_expired_removes_entries_without_a_read(self) -> None:
        """purge_expired removes expired entries even if they were never read."""
        cache = HashCache(ttl_seconds=0, max_entries=10)
        await cache.set('one', self.thread_id, 'a')
        await cache.set('two', self.thread_id, 'b')

        assert cache.purge_expired() == 2
        assert len(cache.cache) == 0

    async def test_stats_exclude_expired_entries(self) -> None:
        """get_stats does not count entries that have expired."""
        cache = HashCache(ttl_seconds=0, max_entries=10)
        await cache.set('query', self.thread_id, 'response')

        assert (await cache.get_stats()).cached_queries == 0


class TestHashEviction:
    """HashCache trims to exactly max_entries on every write."""

    thread_id: str = 'thrd-1'

    async def test_never_exceeds_max_entries(self, hash_cache: HashCache) -> None:
        """Writing past max_entries trims the cache back down to max_entries."""
        for i in range(10):
            await hash_cache.set(f'query {i}', self.thread_id, f'response {i}')

        assert len(hash_cache) == 3

    async def test_evicts_least_recently_used(self, hash_cache: HashCache) -> None:
        """A recent read promotes an entry, sparing it from LRU eviction."""
        await hash_cache.set('a', self.thread_id, 'response a')
        await hash_cache.set('b', self.thread_id, 'response b')
        await hash_cache.set('c', self.thread_id, 'response c')

        # 'a' becomes the most recently used, so 'b' is next out.
        await hash_cache.get('a', self.thread_id)
        await hash_cache.set('d', self.thread_id, 'response d')

        assert await hash_cache.get('b', self.thread_id) is None
        assert await hash_cache.get('a', self.thread_id) == 'response a'
        assert await hash_cache.get('c', self.thread_id) == 'response c'
        assert await hash_cache.get('d', self.thread_id) == 'response d'


class TestSemanticEviction:
    """SemanticCache evicts on the maintenance pass, not on write.

    It ranks on hits rather than recency, and evicts down to a target below
    max_entries so concurrent passes across workers do not thrash at the
    boundary. Neither behaviour matches HashCache, so these cannot be shared.
    """

    thread_id: str = 'thrd-1'

    async def test_evicts_down_to_target_below_max_entries(
        self, semantic_cache: SemanticCache
    ) -> None:
        """Maintenance evicts down to the configured target ratio, not exactly max_entries."""
        await semantic_cache.clear()
        for i in range(10):
            await semantic_cache.set(f'query {i}', self.thread_id, f'response {i}')

        await semantic_cache.run_maintenance_once()

        target = int(
            semantic_cache.max_entries
            * semantic_cache.settings.cache_eviction_target_ratio
        )
        assert len(semantic_cache) == target

    async def test_evicts_least_frequently_used(
        self, semantic_cache: SemanticCache
    ) -> None:
        """A recently-hit entry survives eviction over one with fewer hits."""
        await semantic_cache.clear()
        await semantic_cache.set('a', self.thread_id, 'response a')
        await semantic_cache.set('b', self.thread_id, 'response b')
        await semantic_cache.set('c', self.thread_id, 'response c')

        # The queued hit on 'a' has to reach the index before eviction ranks on
        # it, which is the ordering run_maintenance_once exists to guarantee.
        await semantic_cache.get('a', self.thread_id)
        await semantic_cache.set('d', self.thread_id, 'response d')
        await semantic_cache.run_maintenance_once()

        assert await semantic_cache.get('a', self.thread_id) == 'response a'


class TestStats:
    """get_stats reporting across both cache implementations."""

    thread_id: str = 'thrd-1'

    async def test_counts_cached_queries(self, cache: Cache) -> None:
        """cached_queries reflects the number of distinct entries set."""
        await cache.set('one', self.thread_id, 'a')
        await cache.set('two', self.thread_id, 'b')

        assert (await cache.get_stats()).cached_queries == 2

    async def test_hash_hits_are_counted_per_entry(self, hash_cache: HashCache) -> None:
        """Each read of a HashCache entry increments its hit count immediately."""
        await hash_cache.set('query', self.thread_id, 'response')
        await hash_cache.get('query', self.thread_id)
        final = await hash_cache.get('query', self.thread_id, return_full=True)

        assert isinstance(final, dict)
        assert final['hits'] == 2

    async def test_semantic_hits_are_counted_per_entry(
        self, semantic_cache: SemanticCache
    ) -> None:
        """Semantic hits are queued, so they only land on the maintenance pass.

        That also means the read that observes them has to come after the flush,
        unlike HashCache where the incrementing read returns the new count itself.
        """
        await semantic_cache.clear()
        await semantic_cache.set('query', self.thread_id, 'response')
        await semantic_cache.get('query', self.thread_id)
        await semantic_cache.get('query', self.thread_id)

        await semantic_cache.run_maintenance_once()
        final = await semantic_cache.get('query', self.thread_id, return_full=True)

        assert isinstance(final, dict)
        assert final['hits'] == 2
