"""OpenSearch client and index provisioning for documents and the semantic cache."""

from typing import Any

from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch as OpenSearchClient

from core.config.settings import get_settings
from core.models.models import Models

type IndexMapping = dict[str, Any]


def match_all_delete_kwargs() -> dict[str, Any]:
    """Body/params for a delete_by_query that clears an entire index."""
    return {
        'body': {'query': {'match_all': {}}},
        'params': {'conflicts': 'proceed', 'refresh': 'true'},
    }


class OpenSearch:
    """OpenSearch client plus the document and cache vector stores built on it."""

    VECTOR_FIELD = 'vector_field'
    TEXT_FIELD = 'text'
    METADATA_FIELD = 'metadata'

    def __init__(self) -> None:
        settings = get_settings()
        models = Models()

        auth = (
            (settings.opensearch_user, settings.opensearch_password)
            if settings.opensearch_user
            else None
        )

        self.client = OpenSearchClient(
            hosts=[settings.opensearch_url],
            http_auth=auth,
        )

        self.embedding_model = models.embedding_llm

        self.document_index = settings.opensearch_documents_index
        self.cache_index = settings.opensearch_cache_index

        self.document_vectorstore = OpenSearchVectorSearch(
            opensearch_url=settings.opensearch_url,
            index_name=self.document_index,
            embedding_function=self.embedding_model,
            http_auth=auth,
        )

        self.cache_vectorstore = OpenSearchVectorSearch(
            opensearch_url=settings.opensearch_url,
            index_name=self.cache_index,
            embedding_function=self.embedding_model,
            http_auth=auth,
        )

    def provision_indexes(self, embedding_dimension: int) -> None:
        """Create all required OpenSearch indexes when missing."""
        self._create_index_if_missing(
            index_name=self.document_index,
            mapping=self._document_index_mapping(embedding_dimension),
        )

        self._create_index_if_missing(
            index_name=self.cache_index,
            mapping=self._cache_index_mapping(embedding_dimension),
        )

    def _create_index_if_missing(
        self,
        index_name: str,
        mapping: IndexMapping,
    ) -> None:
        if self.client.indices.exists(index=index_name):
            return

        self.client.indices.create(
            index=index_name,
            body=mapping,
        )

    @classmethod
    def _document_index_mapping(
        cls,
        embedding_dimension: int,
    ) -> IndexMapping:
        return {
            'settings': {
                'index': {
                    'knn': True,
                    'number_of_shards': 1,
                    'number_of_replicas': 0,
                },
            },
            'mappings': {
                'dynamic': 'strict',
                'properties': {
                    cls.VECTOR_FIELD: cls._vector_mapping(embedding_dimension),
                    cls.TEXT_FIELD: {
                        'type': 'text',
                    },
                    cls.METADATA_FIELD: {
                        'type': 'object',
                        'properties': {
                            'source': {
                                'type': 'keyword',
                            },
                            'indexed_at': {
                                'type': 'date',
                            },
                            'document_id': {
                                'type': 'keyword',
                            },
                            'page': {
                                'type': 'integer',
                            },
                            'chunk_index': {
                                'type': 'integer',
                            },
                            'title': {
                                'type': 'text',
                                'fields': {
                                    'keyword': {
                                        'type': 'keyword',
                                    },
                                },
                            },
                        },
                    },
                },
            },
        }

    @classmethod
    def _cache_index_mapping(
        cls,
        embedding_dimension: int,
    ) -> IndexMapping:
        return {
            'settings': {
                'index': {
                    'knn': True,
                    'number_of_shards': 1,
                    'number_of_replicas': 0,
                },
            },
            'mappings': {
                'dynamic': 'strict',
                'properties': {
                    cls.VECTOR_FIELD: cls._vector_mapping(embedding_dimension),
                    cls.TEXT_FIELD: {
                        'type': 'text',
                    },
                    cls.METADATA_FIELD: {
                        'type': 'object',
                        'properties': {
                            'query': {
                                'type': 'text',
                            },
                            'response': {
                                'type': 'text',
                                'index': False,
                            },
                            'thread_id': {
                                'type': 'keyword',
                            },
                            'timestamp': {
                                'type': 'long',
                            },
                            'hits': {
                                'type': 'integer',
                            },
                        },
                    },
                },
            },
        }

    @classmethod
    def _vector_mapping(
        cls,
        embedding_dimension: int,
    ) -> IndexMapping:
        return {
            'type': 'knn_vector',
            'dimension': embedding_dimension,
            'method': {
                'name': 'hnsw',
                'engine': 'lucene',
                'space_type': 'cosinesimil',
                'parameters': {
                    'ef_construction': 128,
                    'm': 16,
                },
            },
        }
