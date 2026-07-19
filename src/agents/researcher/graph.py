from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.researcher.nodes import research
from agents.researcher.state import ResearcherState


def build_graph() -> CompiledStateGraph[ResearcherState]:
  """Construct and compile the researcher graph."""
  graph = StateGraph(ResearcherState)
  graph.add_node('research', research)
  graph.add_edge(START, 'research')
  graph.add_edge('research', END)
  return graph.compile()
