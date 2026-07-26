import pytest

from app.common.cache.hash_cache import HashCache


@pytest.fixture
def cache() -> HashCache:
  return HashCache(ttl_seconds=300, max_entries=3)


class TestLookup:
  def test_returns_cached_response(self, cache: HashCache) -> None:
    cache.set('what is my balance?', 'your balance is 10')

    assert cache.get('what is my balance?') == 'your balance is 10'

  def test_normalises_case_and_whitespace(self, cache: HashCache) -> None:
    cache.set('What Is My Balance?', 'your balance is 10')

    assert cache.get('  what is my balance?  ') == 'your balance is 10'

  def test_returns_none_when_missing(self, cache: HashCache) -> None:
    assert cache.get('never asked') is None

  def test_overwrites_existing_response(self, cache: HashCache) -> None:
    cache.set('query', 'first')
    cache.set('query', 'second')

    assert cache.get('query') == 'second'
    assert cache.get_stats().cached_queries == 1


class TestExpiry:
  def test_expired_entry_is_a_miss(self) -> None:
    cache = HashCache(ttl_seconds=0, max_entries=10)
    cache.set('query', 'response')

    assert cache.get('query') is None

  def test_expired_entry_is_dropped_on_read(self) -> None:
    cache = HashCache(ttl_seconds=0, max_entries=10)
    cache.set('query', 'response')
    cache.get('query')

    assert len(cache.cache) == 0

  def test_purge_expired_removes_entries_without_a_read(self) -> None:
    cache = HashCache(ttl_seconds=0, max_entries=10)
    cache.set('one', 'a')
    cache.set('two', 'b')

    assert cache.purge_expired() == 2
    assert len(cache.cache) == 0

  def test_stats_exclude_expired_entries(self) -> None:
    cache = HashCache(ttl_seconds=0, max_entries=10)
    cache.set('query', 'response')

    assert cache.get_stats().cached_queries == 0


class TestEviction:
  def test_never_exceeds_max_entries(self, cache: HashCache) -> None:
    for i in range(10):
      cache.set(f'query {i}', f'response {i}')

    assert len(cache.cache) == 3

  def test_evicts_least_recently_used(self, cache: HashCache) -> None:
    cache.set('a', 'response a')
    cache.set('b', 'response b')
    cache.set('c', 'response c')

    # 'a' becomes the most recently used, so 'b' is next out.
    cache.get('a')
    cache.set('d', 'response d')

    assert cache.get('b') is None
    assert cache.get('a') == 'response a'
    assert cache.get('c') == 'response c'
    assert cache.get('d') == 'response d'


class TestStats:
  def test_counts_cached_queries(self, cache: HashCache) -> None:
    cache.set('one', 'a')
    cache.set('two', 'b')

    assert cache.get_stats().cached_queries == 2

  def test_hits_are_counted_per_entry(self, cache: HashCache) -> None:
    cache.set('query', 'response')
    cache.get('query')
    cache.get('query')

    key = HashCache._hash_query('query')  # pyright: ignore[reportPrivateUsage]

    assert cache.cache[key]['hits'] == 2
