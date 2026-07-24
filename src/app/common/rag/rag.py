from datetime import datetime

from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config.settings import get_settings
from core.logging.logger import logger


class Rag:
  def __init__(self, vectorstore: OpenSearchVectorSearch) -> None:
    self.settings = get_settings()
    self.vectorstore = vectorstore
    self.splitter = RecursiveCharacterTextSplitter(
      separators=['\n\n', '\n', '. ', ' ', ''],
      chunk_size=1000,
      chunk_overlap=200,
    )

  def add_documents(self, documents: list[Document], source: str) -> int:
    """Add documents to vector store."""
    if source:
      for doc in documents:
        doc.metadata['source'] = source

    chunks = self.splitter.split_documents(documents)

    now = datetime.now().isoformat()
    for chunk in chunks:
      chunk.metadata['indexed_at'] = now

    self.vectorstore.add_documents(chunks)

    logger.info(
      f'Added {len(chunks)} chunks from {len(documents)} documents to vector store.'
    )

    return len(chunks)

  def add_texts(self, texts: list[str], source: str) -> int:
    """Add texts as documents to vector store."""
    documents = [Document(page_content=t, metadata={'source': source}) for t in texts]
    return self.add_documents(documents, source)

  def get_document_count(self) -> int:
    """Get the count of indexed chunks in vector store."""
    if not self.vectorstore.client.indices.exists(
      index=self.settings.opensearch_documents_index
    ):
      return 0
    response = self.vectorstore.client.count(
      index=self.settings.opensearch_documents_index
    )
    return response['count']

  def ask(self, query: str) -> list[Document]:
    retriever = self._build_retriever()
    return retriever.invoke(query)

  def _build_retriever(self) -> VectorStoreRetriever:
    """Build a similarity retriever"""
    return self.vectorstore.as_retriever(
      search_type='similarity', search_kwargs={'k': 4}
    )

  def _format_docs_for_context(self, documents: list[Document]) -> str:
    """Format retrieved documents into a string for the prompt."""
    if not documents or len(documents) == 0:
      return 'No relevant documents found.'

    formatted: list[str] = []
    for i, document in enumerate(documents):
      source = document.metadata.get('source', 'unknown')
      formatted.append(f'[Source {i + 1}: {source}]\n{document.page_content}')
    return '\n\n---\n\n'.join(formatted)
