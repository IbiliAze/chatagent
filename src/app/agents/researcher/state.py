from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages  # pyright: ignore[reportMissingTypeStubs]

from app.agents.researcher.schemas import ResearcherAgent
from core.config.types import AvailableModels


class ResearcherState(TypedDict):
  messages: Annotated[list[BaseMessage], add_messages]
  current_agent: ResearcherAgent | None
  handoff_reason: str
  context_summary: str
  error: NotRequired[str | None]
  retry_count: NotRequired[int]
  model_used: NotRequired[AvailableModels]


class ResearchUpdate(TypedDict):
  messages: list[BaseMessage]
  model_used: AvailableModels


class TriageUpdate(TypedDict):
  current_agent: ResearcherAgent
  handoff_reason: str
  context_summary: str
  messages: list[BaseMessage]


class SpecialistUpdate(TypedDict):
  messages: Annotated[list[BaseMessage], add_messages]
  current_agent: ResearcherAgent
  model_used: AvailableModels
