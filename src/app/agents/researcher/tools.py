from langchain_core.tools import tool

from app.common.rag.rag import Rag


class ResearcherTools:
  def __init__(self, rag: Rag) -> None:
    self.rag = rag
    self.get_relevant_documents = tool(self._get_relevant_documents)

  def _get_relevant_documents(self, query: str) -> str:
    """Search for information relevant to the query."""
    return self.rag.ask(query)
