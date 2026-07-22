from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.researcher.nodes import research
from app.agents.researcher.state import ResearcherState


def build_graph() -> CompiledStateGraph[ResearcherState]:
  """Construct and compile the researcher graph."""
  graph = StateGraph(ResearcherState)
  graph.add_node('research', research)
  graph.add_edge(START, 'research')
  graph.add_edge('research', END)
  return graph.compile()
