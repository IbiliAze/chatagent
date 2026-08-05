"""Unit tests for Rag against a mocked vector store."""

from collections.abc import Iterator
from datetime import datetime
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from opensearchpy.exceptions import OpenSearchException

from app.rag.rag import Rag
from core.config.settings import get_settings


@pytest.fixture(autouse=True)
def _settings_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Resolve settings without a .env file so these run on a bare CI box."""
    required = {
        'OPENAI_API_KEY': 'test',
        'ANTHROPIC_API_KEY': 'test',
        'MCP_URL': 'http://localhost:8000/mcp',
        'MCP_REMOTE_NAME': 'test',
        'APP_ENV': 'test',
        'LOG_LEVEL': 'INFO',
        'RATE_LIMIT': '100/minute',
        'CACHE_TTL_SECONDS': '60',
    }
    for key, value in required.items():
        monkeypatch.setenv(key, value)

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def vectorstore() -> MagicMock:
    """A mocked vector store standing in for OpenSearchVectorSearch."""
    return MagicMock()


@pytest.fixture
def rag(vectorstore: MagicMock) -> Rag:
    """Build a Rag instance backed by the mocked vector store."""
    return Rag(vectorstore)


def indexed_chunks(vectorstore: MagicMock) -> list[Document]:
    """The chunks handed to the vector store by the last add_documents call."""
    return vectorstore.add_documents.call_args.args[0]


class TestAddDocuments:
    """add_documents' chunking and metadata stamping."""

    def test_returns_chunk_count(self, rag: Rag) -> None:
        """Returns the number of chunks the document was split into."""
        count = rag.add_documents([Document(page_content='short')], source='src')

        assert count == 1

    def test_passes_chunks_to_vector_store(
        self, rag: Rag, vectorstore: MagicMock
    ) -> None:
        """The chunked documents are handed to the vector store's add_documents."""
        rag.add_documents([Document(page_content='short')], source='src')

        vectorstore.add_documents.assert_called_once()
        assert [c.page_content for c in indexed_chunks(vectorstore)] == ['short']

    def test_stamps_source_on_every_chunk(
        self, rag: Rag, vectorstore: MagicMock
    ) -> None:
        """Every chunk's metadata carries the given source."""
        rag.add_documents(
            [Document(page_content='a'), Document(page_content='b')], source='my-source'
        )

        assert all(
            c.metadata['source'] == 'my-source' for c in indexed_chunks(vectorstore)
        )

    def test_stamps_indexed_at_on_every_chunk(
        self, rag: Rag, vectorstore: MagicMock
    ) -> None:
        """Every chunk's metadata carries a valid indexed_at timestamp."""
        rag.add_documents([Document(page_content='a')], source='src')

        stamped = indexed_chunks(vectorstore)[0].metadata['indexed_at']

        assert datetime.fromisoformat(stamped)

    def test_blank_source_is_not_stamped(
        self, rag: Rag, vectorstore: MagicMock
    ) -> None:
        """An empty source string is not written into chunk metadata."""
        rag.add_documents([Document(page_content='a')], source='')

        assert 'source' not in indexed_chunks(vectorstore)[0].metadata

    def test_splits_documents_longer_than_chunk_size(
        self, rag: Rag, vectorstore: MagicMock
    ) -> None:
        """A document longer than the chunk size is split into multiple chunks."""
        long_text = '. '.join(f'Sentence number {i} about chunking' for i in range(200))

        count = rag.add_documents([Document(page_content=long_text)], source='src')

        assert count > 1
        assert len(indexed_chunks(vectorstore)) == count

    def test_split_chunks_inherit_source(
        self, rag: Rag, vectorstore: MagicMock
    ) -> None:
        """Every chunk produced by splitting a long document keeps the source."""
        long_text = '. '.join(f'Sentence number {i} about chunking' for i in range(200))

        rag.add_documents([Document(page_content=long_text)], source='my-source')

        assert all(
            c.metadata['source'] == 'my-source' for c in indexed_chunks(vectorstore)
        )


class TestAddTexts:
    """add_texts wraps raw strings as Documents before indexing."""

    def test_wraps_texts_as_documents(self, rag: Rag, vectorstore: MagicMock) -> None:
        """Each input string becomes its own indexed chunk."""
        count = rag.add_texts(['first', 'second'], source='src')

        assert count == 2
        assert [c.page_content for c in indexed_chunks(vectorstore)] == [
            'first',
            'second',
        ]

    def test_stamps_source(self, rag: Rag, vectorstore: MagicMock) -> None:
        """The given source is stamped onto the wrapped chunk's metadata."""
        rag.add_texts(['first'], source='my-source')

        assert indexed_chunks(vectorstore)[0].metadata['source'] == 'my-source'


class TestGetDocumentCount:
    """get_document_count against a missing or existing index."""

    def test_returns_zero_when_index_is_missing(
        self, rag: Rag, vectorstore: MagicMock
    ) -> None:
        """Returns 0 without querying the count when the index doesn't exist."""
        vectorstore.client.indices.exists.return_value = False

        assert rag.get_document_count() == 0
        vectorstore.client.count.assert_not_called()

    def test_returns_count_from_client(self, rag: Rag, vectorstore: MagicMock) -> None:
        """Returns the count reported by the vector store's client."""
        vectorstore.client.indices.exists.return_value = True
        vectorstore.client.count.return_value = {'count': 7}

        assert rag.get_document_count() == 7


class TestAsk:
    """ask's retrieval, formatting, and error handling."""

    def test_retrieves_four_most_similar_documents(
        self, rag: Rag, vectorstore: MagicMock
    ) -> None:
        """Builds a similarity retriever with k=4 and invokes it with the query."""
        vectorstore.as_retriever.return_value.invoke.return_value = []

        rag.ask('a query')

        vectorstore.as_retriever.assert_called_once_with(
            search_type='similarity', search_kwargs={'k': 4}
        )
        vectorstore.as_retriever.return_value.invoke.assert_called_once_with('a query')

    def test_formats_retrieved_documents(
        self, rag: Rag, vectorstore: MagicMock
    ) -> None:
        """Retrieved documents are formatted with their source label."""
        vectorstore.as_retriever.return_value.invoke.return_value = [
            Document(page_content='body', metadata={'source': 'src'})
        ]

        assert rag.ask('a query') == '[Source 1: src]\nbody'

    def test_returns_placeholder_when_nothing_retrieved(
        self, rag: Rag, vectorstore: MagicMock
    ) -> None:
        """Returns a placeholder message when retrieval finds no documents."""
        vectorstore.as_retriever.return_value.invoke.return_value = []

        assert rag.ask('a query') == 'No relevant documents found.'

    def test_returns_fallback_when_retrieval_fails(
        self, rag: Rag, vectorstore: MagicMock
    ) -> None:
        """An OpenSearchException during retrieval is turned into a friendly message."""
        vectorstore.as_retriever.return_value.invoke.side_effect = OpenSearchException(
            'cluster down'
        )

        assert rag.ask('a query') == 'Document search is temporarily unavailable.'

    def test_does_not_swallow_unexpected_errors(
        self, rag: Rag, vectorstore: MagicMock
    ) -> None:
        """Errors other than OpenSearchException propagate to the caller."""
        vectorstore.as_retriever.return_value.invoke.side_effect = ValueError('bug')

        with pytest.raises(ValueError, match='bug'):
            rag.ask('a query')


class TestFormatDocsForContext:
    """Formatting of retrieved documents into the prompt-ready context string."""

    def test_numbers_sources_from_one(self, rag: Rag, vectorstore: MagicMock) -> None:
        """Sources are numbered starting at 1, in retrieval order."""
        documents = [
            Document(page_content='first', metadata={'source': 'a'}),
            Document(page_content='second', metadata={'source': 'b'}),
        ]
        vectorstore.as_retriever.return_value.invoke.return_value = documents

        formatted = rag.ask('a query')

        assert formatted.startswith('[Source 1: a]')
        assert '[Source 2: b]' in formatted

    def test_joins_documents_with_a_separator(
        self, rag: Rag, vectorstore: MagicMock
    ) -> None:
        """Multiple formatted documents are joined by the '---' separator."""
        documents = [
            Document(page_content='first', metadata={'source': 'a'}),
            Document(page_content='second', metadata={'source': 'b'}),
        ]
        vectorstore.as_retriever.return_value.invoke.return_value = documents

        formatted = rag.ask('a query')

        assert formatted == '[Source 1: a]\nfirst\n\n---\n\n[Source 2: b]\nsecond'

    def test_falls_back_to_unknown_source(
        self, rag: Rag, vectorstore: MagicMock
    ) -> None:
        """A document with no source metadata is labelled 'unknown'."""
        vectorstore.as_retriever.return_value.invoke.return_value = [
            Document(page_content='body')
        ]

        formatted = rag.ask('a query')

        assert formatted == '[Source 1: unknown]\nbody'

    def test_empty_documents_returns_placeholder(
        self, rag: Rag, vectorstore: MagicMock
    ) -> None:
        """An empty document list formats to the placeholder message."""
        vectorstore.as_retriever.return_value.invoke.return_value = []

        assert rag.ask('a query') == 'No relevant documents found.'
