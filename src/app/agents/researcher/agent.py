"""The researcher graph: triage plus sales/billing/support specialists."""

from collections.abc import Iterator
from typing import cast

from langchain_core.messages import HumanMessage
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
from app.agents.researcher.routes import ResearcherRoutes
from app.agents.researcher.state import ResearcherResponse, ResearcherState
from core.config.settings import get_settings


class ResearcherAgent:
    """Builds and runs the triage/specialist LangGraph state machine."""

    def __init__(
        self,
        nodes: ResearcherNodes,
        routes: ResearcherRoutes,
        saver: MemorySaver | SqliteSaver,
    ) -> None:
        self.settings = get_settings()
        self.max_retries = self.settings.max_retries
        self.graph = self._build_graph(nodes, routes, saver)

    @staticmethod
    def _build_graph(
        nodes: ResearcherNodes,
        routes: ResearcherRoutes,
        saver: MemorySaver | SqliteSaver,
    ) -> CompiledStateGraph[ResearcherState]:
        """Construct and compile the researcher graph."""
        graph = StateGraph(ResearcherState)

        graph.add_node('triage', nodes.triage_agent)  # pyright: ignore[reportUnknownMemberType]
        graph.add_node('sales', nodes.sales_agent)  # pyright: ignore[reportUnknownMemberType]
        graph.add_node('billing', nodes.billing_agent)  # pyright: ignore[reportUnknownMemberType]
        graph.add_node('support', nodes.support_agent)  # pyright: ignore[reportUnknownMemberType]
        graph.add_node('tools', nodes.tool_node)  # pyright: ignore[reportUnknownMemberType]

        graph.add_edge(START, 'triage')
        graph.add_conditional_edges(
            'triage',
            routes.route_from_triage,
            {'sales': 'sales', 'billing': 'billing', 'support': 'support', 'end': END},
        )
        graph.add_conditional_edges(
            'billing', routes.should_continue, {'tools': 'tools', 'end': END}
        )
        graph.add_conditional_edges(
            'support', routes.should_continue, {'tools': 'tools', 'end': END}
        )
        graph.add_conditional_edges(
            'sales', routes.should_continue, {'tools': 'tools', 'end': END}
        )
        graph.add_conditional_edges(
            'tools',
            routes.route_from_tools,
            {'sales': 'sales', 'billing': 'billing', 'support': 'support'},
        )
        return graph.compile(checkpointer=saver)  # pyright: ignore[reportUnknownMemberType]

    def process_message(
        self, state: ResearcherState, config: RunnableConfig
    ) -> ResearcherState:
        """Process a message.

        The cast is unavoidable: CompiledStateGraph is generic over its output type
        but Pregel.invoke is annotated `dict[str, Any] | Any`, so the parameter never
        reaches the call site. Sound only because every key the graph may leave
        unwritten is NotRequired on ResearcherState.
        """
        return cast(
            ResearcherState,
            self.graph.invoke(state, config=config),  # pyright: ignore[reportUnknownMemberType]
        )

    def stream_messages(
        self, state: ResearcherState, config: RunnableConfig
    ) -> Iterator[ResearcherResponse]:
        """Yield (node, message) pairs as each node produces them.

        stream_mode='updates' emits only what a node just wrote, so each message is
        yielded once. 'values' would re-emit the whole accumulated history on every
        step and would need diffing.
        """
        stream = cast(
            Iterator[dict[str, ResearcherState]],
            self.graph.stream(  # pyright: ignore[reportUnknownMemberType]
                state, config=config, stream_mode='updates'
            ),
        )
        for chunk in stream:
            for node, update in chunk.items():
                model_used = update.get('model_used', 'UNKNOWN MODEL')
                handoff_reason = update.get('handoff_reason')
                for message in update.get('messages', []):
                    yield ResearcherResponse(
                        message=message,
                        model_used=model_used,
                        current_agent=node,
                        handoff_reason=handoff_reason,
                    )

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

    @staticmethod
    def build_config(thread_id: str) -> RunnableConfig:
        """Builds a runnable config."""
        return {'configurable': {'thread_id': thread_id}}

    @staticmethod
    def build_message(text: str) -> ResearcherState:
        """Builds a processable message."""

        return {
            'messages': [HumanMessage(text)],
            'error': '',
            'retry_count': 0,
            'context_summary': '',
            'current_agent': None,
            'handoff_reason': '',
        }
