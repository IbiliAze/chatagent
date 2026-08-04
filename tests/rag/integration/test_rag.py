import pytest
from dotenv import load_dotenv
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.documents import Document

from app.rag.rag import Rag
from core.store.vectorstore.opensearch import OpenSearch

load_dotenv()

pytestmark = pytest.mark.integration


@pytest.fixture
def vector_store() -> OpenSearchVectorSearch:
  opensearch = OpenSearch()
  opensearch.provision_indexes(embedding_dimension=1536)
  return opensearch.document_vectorstore


@pytest.fixture
def rag(vector_store: OpenSearchVectorSearch) -> Rag:
  return Rag(vector_store)


class TestAddition:
  def test_add_documents_returns(self, rag: Rag) -> None:
    rag.clear()
    number_of_chunks = rag.add_documents(
      documents=[Document(page_content='My test document')],
      source='Test document source',
    )

    assert number_of_chunks is not None
    assert type(number_of_chunks) is int
    assert number_of_chunks == 1

  def test_add_texts_returns(self, rag: Rag) -> None:
    rag.clear()
    number_of_chunks = rag.add_texts(texts=['My small text'], source='Test text source')

    assert number_of_chunks is not None
    assert type(number_of_chunks) is int
    assert number_of_chunks == 1


class TestQueryResponse:
  def test_ask_returns_relevant_info(self, rag: Rag):
    knowledge = 'Eight mile providers software solutions for agentic apps'

    rag.clear()
    rag.add_documents(
      documents=[Document(page_content=knowledge)],
      source='Test document source',
    )

    response = rag.ask('Who are eight mile?')

    assert response is not None
    assert type(response) is str
    assert knowledge in response
