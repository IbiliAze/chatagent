from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt.tool_node import ToolNode
from langsmith import traceable  # pyright: ignore[reportUnknownVariableType]

from app.agents.researcher.prompts import (
    BILLING_AGENT_PROMPT,
    SALES_AGENT_PROMPT,
    SUPPORT_AGENT_PROMPT,
    TRIAGE_AGENT_PROMPT,
)
from app.agents.researcher.schemas import HandoffDecision
from app.agents.researcher.state import (
    ResearcherState,
    SpecialistUpdate,
    TriageUpdate,
)
from core.config.types import AvailableModels
from core.models.models import Models

load_dotenv()


class ResearcherNodes:
    def __init__(self, models: Models, tools: list[BaseTool]) -> None:
        self.llm = models.llm
        self.llm_with_tools = models.with_tools(tools)
        self.triage_llm = models.with_schema(HandoffDecision)
        self.tool_node = ToolNode(tools)

    @traceable(name='triage_agent')
    def triage_agent(self, state: ResearcherState) -> TriageUpdate:
        """Initial triage to route the customer query to."""
        messages: list[BaseMessage] = [
            SystemMessage(content=TRIAGE_AGENT_PROMPT),
            *state['messages'],
        ]
        decision = self.triage_llm.invoke(messages)

        if decision.handoff_to == 'end':
            response = self.llm.invoke(
                [
                    SystemMessage(
                        content='Provide a brief, helpful message to customer.'
                    ),
                    *state['messages'],
                ]
            )

            model_used: AvailableModels = response.response_metadata.get(
                'model_name', 'UNKNOWN MODEL'
            )

            return {
                'messages': [AIMessage(content=response.content)],
                'current_agent': 'end',
                'context_summary': '',
                'model_used': model_used,
                'handoff_reason': '',
            }

        return {
            'context_summary': decision.context,
            'handoff_reason': decision.reason,
            'current_agent': decision.handoff_to,
            'messages': [
                AIMessage(f'Transferring to {decision.handoff_to}. {decision.reason}')
            ],
        }

    @traceable(name='sales_agent')
    def sales_agent(self, state: ResearcherState) -> SpecialistUpdate:
        """Sales specialist."""

        response = self.llm_with_tools.invoke(
            [
                SystemMessage(
                    content=SALES_AGENT_PROMPT(state.get('context_summary', 'None'))
                ),
                *state['messages'],
            ]
        )

        response.content = response.content
        model_used: AvailableModels = response.response_metadata.get(
            'model_name', 'UNKNOWN MODEL'
        )

        return {
            'current_agent': 'sales',
            'model_used': model_used,
            'messages': [response],
        }

    @traceable(name='support_agent')
    def support_agent(self, state: ResearcherState) -> SpecialistUpdate:
        """Support specialist."""

        response = self.llm_with_tools.invoke(
            [
                SystemMessage(
                    content=SUPPORT_AGENT_PROMPT(state.get('context_summary', 'None'))
                ),
                *state['messages'],
            ]
        )

        model_used: AvailableModels = response.response_metadata.get(
            'model_name', 'UNKNOWN MODEL'
        )

        return {
            'current_agent': 'support',
            'model_used': model_used,
            'messages': [response],
        }

    @traceable(name='billing_agent')
    def billing_agent(self, state: ResearcherState) -> SpecialistUpdate:
        """Billing specialist."""

        response = self.llm_with_tools.invoke(
            [
                SystemMessage(
                    content=BILLING_AGENT_PROMPT(state.get('context_summary', 'None'))
                ),
                *state['messages'],
            ]
        )

        model_used: AvailableModels = response.response_metadata.get(
            'model_name', 'UNKNOWN MODEL'
        )

        return {
            'current_agent': 'billing',
            'model_used': model_used,
            'messages': [response],
        }
