from langchain_core.tools import tool


@tool
def search(query: str) -> str:
  """Search for information relevant to the query."""
  raise NotImplementedError
