from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages  # pyright: ignore[reportMissingTypeStubs]


class ResearcherState(TypedDict):
  """State passed between nodes in the researcher graph."""

  messages: Annotated[list[AnyMessage], add_messages]
  error: str | None
  retry_count: int
  model_used: str
