from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages  # pyright: ignore[reportMissingTypeStubs]

from app.agents.researcher.schemas import ResearcherAgent


class ResearcherState(TypedDict):
  """State passed between nodes in the researcher graph."""

  messages: Annotated[list[BaseMessage], add_messages]
  current_agent: ResearcherAgent
  handoff_reason: str
  context_summary: str
  error: str | None
  retry_count: int
  model_used: str


class ResearchUpdate(TypedDict):
  messages: list[BaseMessage]


class TriageUpdate(TypedDict):
  current_agent: ResearcherAgent
  handoff_reason: str
  context_summary: str
  messages: list[BaseMessage]


class SpecialistUpdate(TypedDict):
  messages: Annotated[list[BaseMessage], add_messages]
  current_agent: ResearcherAgent
