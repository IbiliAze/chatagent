import pytest
from dotenv import load_dotenv
from langchain_community.vectorstores import OpenSearchVectorSearch

from app.cache.cache import Cache
from app.cache.hash_cache import HashCache
from app.cache.semantic_cache import SemanticCache
from core.store.vectorstore.opensearch import OpenSearch

load_dotenv()


@pytest.fixture
def vector_store() -> OpenSearchVectorSearch:
  opensearch = OpenSearch()
  opensearch.provision_indexes(embedding_dimension=1536)
  return opensearch.cache_vectorstore


@pytest.fixture
def semantic_cache(vector_store: OpenSearchVectorSearch) -> SemanticCache:
  return SemanticCache(
    vectorstore=vector_store,  # pyright: ignore[reportArgumentType]
    score_threshold=0.95,
    ttl_seconds=300,
    max_entries=3,
  )


@pytest.fixture
def hash_cache() -> HashCache:
  return HashCache(ttl_seconds=300, max_entries=3)


class TestLookup:
  thread_id: str = 'thrd-1'

  @pytest.mark.parametrize(
    'cache',
    [
      semantic_cache,
      hash_cache,
    ],
  )
  async def test_returns_cached_response(self, cache: Cache) -> None:
    await cache.set('what is my balance?', self.thread_id, 'your balance is 10')

    assert (
      await cache.get('what is my balance?', self.thread_id) == 'your balance is 10'
    )

  @pytest.mark.parametrize(
    'cache',
    [
      semantic_cache,
      hash_cache,
    ],
  )
  async def test_normalises_case_and_whitespace(self, cache: Cache) -> None:
    await cache.set('What Is My Balance?', self.thread_id, 'your balance is 10')

    assert (
      await cache.get(
        '  what is my balance?  ',
        self.thread_id,
      )
      == 'your balance is 10'
    )

  @pytest.mark.parametrize(
    'cache',
    [
      semantic_cache,
      hash_cache,
    ],
  )
  async def test_returns_none_when_missing(self, cache: Cache) -> None:
    assert (
      await cache.get(
        'never asked',
        self.thread_id,
      )
      is None
    )

  @pytest.mark.parametrize(
    'cache',
    [
      semantic_cache,
      hash_cache,
    ],
  )
  async def test_overwrites_existing_response(self, cache: Cache) -> None:
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
