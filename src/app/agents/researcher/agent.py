from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import (  # pyright: ignore[reportMissingTypeStubs]
  END,
  START,
  StateGraph,
)
from langgraph.graph.state import (  # pyright: ignore[reportMissingTypeStubs]
  CompiledStateGraph,
)

from app.agents.researcher.nodes import ResearcherNodes
from app.agents.researcher.state import ResearcherState
from core.config.settings import get_settings


class ResearcherAgent:
  def __init__(
    self,
    nodes: ResearcherNodes,
    saver: MemorySaver | SqliteSaver,
  ) -> None:
    self.settings = get_settings()
    self.max_retries = self.settings.max_retries
    self.graph = self._build_graph(nodes, saver)

  @staticmethod
  def _build_graph(
    nodes: ResearcherNodes,
    saver: MemorySaver | SqliteSaver,
  ) -> CompiledStateGraph[ResearcherState]:
    """Construct and compile the researcher graph."""
    graph = StateGraph(ResearcherState)
    graph.add_node('research', nodes.research)  # pyright: ignore[reportUnknownMemberType]
    graph.add_edge(START, 'research')
    graph.add_edge('research', END)
    return graph.compile(checkpointer=saver)  # pyright: ignore[reportUnknownMemberType]

  def process_message(
    self, input: ResearcherState, config: RunnableConfig
  ) -> dict[str, Any] | Any:
    """Process a message."""
    result = self.graph.invoke(input, config=config)  # pyright: ignore[reportUnknownMemberType]
    return result

  def get_graph_png(self):
    """Get graph as PNG image."""
    png_bytes = self.graph.get_graph().draw_mermaid_png()
    with open('researcher_graph.png', 'wb') as f:
      f.write(png_bytes)

  def get_current_state(self, config: RunnableConfig):
    """Get current LangGraph state."""
    return self.graph.get_state(config)

  def get_state_history(self, config: RunnableConfig):
    """Get current LangGraph state."""
    return self.graph.get_state_history(config)
