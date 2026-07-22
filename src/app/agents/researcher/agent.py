from typing import Any

from app.common.models.models import Models
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
  def __init__(self, models: Models) -> None:
    self.settings = get_settings()
    self.max_retries = self.settings.max_retries

    nodes = ResearcherNodes(models)
    self.graph = self._build_graph(nodes)

  @staticmethod
  def _build_graph(nodes: ResearcherNodes) -> CompiledStateGraph[ResearcherState]:
    """Construct and compile the researcher graph."""
    graph = StateGraph(ResearcherState)
    graph.add_node('research', nodes.research)  # pyright: ignore[reportUnknownMemberType]
    graph.add_edge(START, 'research')
    graph.add_edge('research', END)
    return graph.compile()  # pyright: ignore[reportUnknownMemberType]

  def process_message(self, input: ResearcherState) -> dict[str, Any] | Any:
    """Process a message with"""
    result = self.graph.invoke(input)  # pyright: ignore[reportUnknownMemberType]
    return result

  def get_graph_png(self):
    """Get graph as PNG image."""
    png_bytes = self.graph.get_graph().draw_mermaid_png()
    with open('researcher_graph.png', 'wb') as f:
      f.write(png_bytes)
