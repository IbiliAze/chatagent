from langchain_openai import ChatOpenAI
from langgraph.graph import (  # pyright: ignore[reportMissingTypeStubs]
  END,
  START,
  StateGraph,
)
from langgraph.graph.state import (
  CompiledStateGraph,  # pyright: ignore[reportMissingTypeStubs]
)

from app.agents.researcher.nodes import research
from app.agents.researcher.state import ResearcherState
from core.config.settings import get_settings


class ResearcherAgent:
  def __init__(self) -> None:
    self.settings = get_settings()

    self.primary_llm = ChatOpenAI(
      model=self.settings.primary_model,
      temperature=0,
      max_retries=0,
      timeout=30,
    )

    self.primary_llm = ChatOpenAI(
      model=self.settings.fallback_model,
      temperature=0,
      max_retries=0,
      timeout=30,
    )

    self.max_retries = self.settings.max_retries
    self.graph = self._build_graph()

  def _build_graph(self) -> CompiledStateGraph[ResearcherState]:
    """Construct and compile the researcher graph."""
    graph = StateGraph(ResearcherState)
    graph.add_node('research', research)
    graph.add_edge(START, 'research')
    graph.add_edge('research', END)
    return graph.compile()

  def process_message(self):
    """Process a message with"""
