from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages  # pyright: ignore[reportMissingTypeStubs]


class ResearcherState(TypedDict):
  """State passed between nodes in the researcher graph."""

  messages: Annotated[list[BaseMessage], add_messages]
  error: str | None
  retry_count: int
  model_used: str


class ResearchUpdate(TypedDict):
  messages: list[BaseMessage]
