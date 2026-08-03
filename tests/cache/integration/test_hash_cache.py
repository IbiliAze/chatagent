import pytest

from app.cache.hash_cache import HashCache


@pytest.fixture
def cache() -> HashCache:
  return HashCache(ttl_seconds=300, max_entries=3)


class TestLookup:
  thread_id: str = 'thrd-1'

  async def test_returns_cached_response(self, cache: HashCache) -> None:
    await cache.set('what is my balance?', self.thread_id, 'your balance is 10')

    assert (
      await cache.get('what is my balance?', self.thread_id) == 'your balance is 10'
    )

  async def test_normalises_case_and_whitespace(self, cache: HashCache) -> None:
    await cache.set('What Is My Balance?', self.thread_id, 'your balance is 10')

    assert (
      await cache.get(
        '  what is my balance?  ',
        self.thread_id,
      )
      == 'your balance is 10'
    )

  async def test_returns_none_when_missing(self, cache: HashCache) -> None:
    assert (
      await cache.get(
        'never asked',
        self.thread_id,
      )
      is None
    )

  async def test_overwrites_existing_response(self, cache: HashCache) -> None:
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
  thread_id: str = 'thrd-1'

  async def test_expired_entry_is_a_miss(self) -> None:
    cache = HashCache(ttl_seconds=0, max_entries=10)
    await cache.set('query', self.thread_id, 'response')

    assert await cache.get('query', self.thread_id) is None

  async def test_expired_entry_is_dropped_on_read(self) -> None:
    cache = HashCache(ttl_seconds=0, max_entries=10)
    await cache.set('query', self.thread_id, 'response')
    await cache.get('query', self.thread_id)

    assert len(cache.cache) == 0

  async def test_purge_expired_removes_entries_without_a_read(self) -> None:
    cache = HashCache(ttl_seconds=0, max_entries=10)
    await cache.set('one', self.thread_id, 'a')
    await cache.set('two', self.thread_id, 'b')

    assert cache.purge_expired() == 2
    assert len(cache.cache) == 0

  async def test_stats_exclude_expired_entries(self) -> None:
    cache = HashCache(ttl_seconds=0, max_entries=10)
    await cache.set('query', self.thread_id, 'response')

    assert (await cache.get_stats()).cached_queries == 0


class TestEviction:
  thread_id: str = 'thrd-1'

  async def test_never_exceeds_max_entries(self, cache: HashCache) -> None:
    for i in range(10):
      await cache.set(f'query {i}', self.thread_id, f'response {i}')

    assert len(cache.cache) == 3

  async def test_evicts_least_recently_used(self, cache: HashCache) -> None:
    await cache.set('a', self.thread_id, 'response a')
    await cache.set('b', self.thread_id, 'response b')
    await cache.set('c', self.thread_id, 'response c')

    # 'a' becomes the most recently used, so 'b' is next out.
    await cache.get('a', self.thread_id)
    await cache.set('d', self.thread_id, 'response d')

    assert await cache.get('b', self.thread_id) is None
    assert await cache.get('a', self.thread_id) == 'response a'
    assert await cache.get('c', self.thread_id) == 'response c'
    assert await cache.get('d', self.thread_id) == 'response d'


class TestStats:
  thread_id: str = 'thrd-1'

  async def test_counts_cached_queries(self, cache: HashCache) -> None:
    await cache.set('one', self.thread_id, 'a')
    await cache.set('two', self.thread_id, 'b')

    assert (await cache.get_stats()).cached_queries == 2

  async def test_hits_are_counted_per_entry(self, cache: HashCache) -> None:
    await cache.set('query', self.thread_id, 'response')
    await cache.get('query', self.thread_id)
    await cache.get('query', self.thread_id)

    key = cache._hash_query('query')  # pyright: ignore[reportPrivateUsage]

    assert cache.cache[key]['hits'] == 2
