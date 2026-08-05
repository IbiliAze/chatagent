"""Conditional-edge routing functions for the researcher graph."""

from typing import Literal, cast

from langchain_core.messages import AIMessage

from app.agents.researcher.schemas import ResearcherAgent
from app.agents.researcher.state import ResearcherState


class ResearcherRoutes:
    """Routing decisions for the researcher graph's conditional edges."""

    def route_from_triage(
        self, state: ResearcherState
    ) -> Literal['billing', 'sales', 'support', 'end']:
        """Route to the specialist triage selected, or end if none."""
        agent = state['current_agent']
        routable_agents: list[ResearcherAgent] = ['billing', 'sales', 'support']
        if agent in routable_agents:
            return agent

        return 'end'

    def route_from_tools(
        self, state: ResearcherState
    ) -> Literal['sales', 'billing', 'support']:
        """Route back to whichever specialist invoked the tool call."""
        return cast(Literal['sales', 'billing', 'support'], state['current_agent'])

    def should_continue(self, state: ResearcherState) -> Literal['tools', 'end']:
        """Check if should continue to tools or end."""
        last_message = state['messages'][-1]

        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return 'tools'
        return 'end'
