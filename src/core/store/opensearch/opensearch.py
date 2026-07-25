from langchain_community.vectorstores import OpenSearchVectorSearch

from core.config.settings import get_settings
from core.models.models import Models


class OpenSearch:
  def __init__(self) -> None:
    settings = get_settings()
    models = Models()

    self.document_vectorstore = OpenSearchVectorSearch(
      opensearch_url=settings.opensearch_url,
      index_name=settings.opensearch_documents_index,
      embedding_function=models.embedding_llm,
      http_auth=(
        (settings.opensearch_user, settings.opensearch_password)
        if settings.opensearch_user
        else None
      ),
    )

    self.cache_vectorstore = OpenSearchVectorSearch(
      opensearch_url=settings.opensearch_url,
      index_name=settings.opensearch_cache_index,
      embedding_function=models.embedding_llm,
      http_auth=(
        (settings.opensearch_user, settings.opensearch_password)
        if settings.opensearch_user
        else None
      ),
    )
