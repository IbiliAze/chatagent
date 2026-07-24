from langchain_core.documents import Document
from langchain_core.tools import tool

from app.common.rag.rag import Rag


class ResearcherTools:
  def __init__(self, rag: Rag) -> None:
    self.rag = rag
    self.get_relevant_documents = tool(self._get_relevant_documents)

  def _get_relevant_documents(self, query: str) -> list[Document]:
    """Search for information relevant to the query."""
    response = self.rag.ask(query)
    return response
