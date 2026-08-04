from collections.abc import Iterator
from datetime import datetime
from typing import Any, cast

import pytest
from dotenv import load_dotenv
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.documents import Document
from opensearchpy.exceptions import NotFoundError

from app.rag.rag import Rag
from core.config.settings import get_settings
from core.store.vectorstore.opensearch import OpenSearch

load_dotenv()

pytestmark = pytest.mark.integration

KNOWLEDGE = 'Eight Mile provides software solutions for agentic apps'
DISTRACTORS = [
  'The mitochondrion is the powerhouse of the cell',
  'Sourdough bread needs a starter culture and a long proof',
  'The Peace of Westphalia was signed in 1648',
  'Titanium alloys are common in aerospace fasteners',
  'Migratory swallows navigate using the magnetic field of the earth',
]


@pytest.fixture(scope='session')
def vector_store() -> OpenSearchVectorSearch:
  opensearch = OpenSearch()

  # Drop the index so every run exercises provision_indexes against a clean cluster.
  # Mapping drift is invisible once an index exists: _create_index_if_missing returns
  # early, and langchain silently auto-creates its own default mapping on first write
  # when provisioning never ran.
  try:
    opensearch.client.indices.delete(index=get_settings().opensearch_documents_index)
  except NotFoundError:
    pass

  # Derived rather than hardcoded, so changing embedding_model cannot leave the
  # knn_vector dimension mismatched against the vectors we actually write.
  dimension = len(opensearch.embedding_model.embed_query('dimension probe'))
  opensearch.provision_indexes(embedding_dimension=dimension)

  return opensearch.document_vectorstore


@pytest.fixture
def rag(vector_store: OpenSearchVectorSearch) -> Rag:
  return Rag(vector_store)


@pytest.fixture(autouse=True)
def _empty_index(rag: Rag) -> Iterator[None]:
  rag.clear()
  yield
  rag.clear()


class TestAddition:
  def test_add_documents_indexes_chunks(self, rag: Rag) -> None:
    assert rag.get_document_count() == 0

    number_of_chunks = rag.add_documents(
      documents=[Document(page_content='My test document')],
      source='Test document source',
    )

    assert number_of_chunks == 1
    assert rag.get_document_count() == 1

  def test_add_texts_indexes_chunks(self, rag: Rag) -> None:
    number_of_chunks = rag.add_texts(texts=['My small text'], source='Test text source')

    assert number_of_chunks == 1
    assert rag.get_document_count() == 1

  def test_long_document_is_split_across_several_chunks(self, rag: Rag) -> None:
    long_text = '. '.join(f'Sentence number {i} about chunking' for i in range(200))

    number_of_chunks = rag.add_texts(texts=[long_text], source='Long text source')

    assert number_of_chunks > 1
    assert rag.get_document_count() == number_of_chunks

  def test_indexed_chunks_carry_source_and_timestamp(self, rag: Rag) -> None:
    # Regression test for the strict mapping: metadata fields the index does not
    # declare are rejected outright, so a successful write is not proof of a
    # round trip.
    rag.add_texts(texts=['Anything'], source='metadata-source')

    response = cast(
      dict[str, Any],
      rag.vectorstore.client.search(
        index=get_settings().opensearch_documents_index,
        body={'query': {'match_all': {}}},
        params={'size': '1'},
      ),
    )
    metadata = cast(dict[str, str], response['hits']['hits'][0]['_source']['metadata'])

    assert metadata['source'] == 'metadata-source'
    assert datetime.fromisoformat(metadata['indexed_at'])

  def test_clear_removes_every_document(self, rag: Rag) -> None:
    rag.add_texts(texts=['one', 'two'], source='Test text source')
    assert rag.get_document_count() == 2

    rag.clear()

    assert rag.get_document_count() == 0


class TestQueryResponse:
  def test_ask_ranks_the_relevant_document_first(self, rag: Rag) -> None:
    rag.add_texts(texts=DISTRACTORS, source='distractor')
    rag.add_texts(texts=[KNOWLEDGE], source='eight-mile')

    response = rag.ask('Who are Eight Mile?')

    assert response.startswith('[Source 1: eight-mile]')
    assert KNOWLEDGE in response

  def test_ask_returns_at_most_four_documents(self, rag: Rag) -> None:
    rag.add_texts(texts=[*DISTRACTORS, KNOWLEDGE], source='corpus')

    response = rag.ask('Who are Eight Mile?')

    assert response.count('[Source ') == 4

  def test_ask_on_an_empty_index_returns_placeholder(self, rag: Rag) -> None:
    assert rag.ask('Who are Eight Mile?') == 'No relevant documents found.'
