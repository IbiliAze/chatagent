"""TypedDicts and dataclasses for the researcher graph's state and node outputs."""

from dataclasses import dataclass
from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages  # pyright: ignore[reportMissingTypeStubs]

from app.agents.researcher.schemas import ResearcherAgent
from core.config.types import AvailableModels


class ResearcherState(TypedDict):
    """Full state threaded through the researcher graph."""

    messages: Annotated[list[BaseMessage], add_messages]
    current_agent: ResearcherAgent | None
    handoff_reason: str
    context_summary: str
    error: NotRequired[str | None]
    retry_count: NotRequired[int]
    model_used: NotRequired[AvailableModels]


class ResearchUpdate(TypedDict):
    """Partial state update carrying a research result."""

    messages: list[BaseMessage]
    model_used: AvailableModels


class TriageUpdate(TypedDict):
    """Partial state update produced by the triage node."""

    current_agent: ResearcherAgent
    handoff_reason: str
    context_summary: str
    model_used: NotRequired[AvailableModels]
    messages: list[BaseMessage]


class SpecialistUpdate(TypedDict):
    """Partial state update produced by a specialist node."""

    messages: Annotated[list[BaseMessage], add_messages]
    current_agent: ResearcherAgent
    model_used: AvailableModels


@dataclass
class ResearcherResponse:
    """One streamed (node, message) pair yielded to the caller."""

    message: BaseMessage
    model_used: str
    current_agent: str
    handoff_reason: str | None
