from typing import cast

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt.tool_node import ToolNode

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
from core.models.models import Models

load_dotenv()


class ResearcherNodes:
  def __init__(self, models: Models, tools: list[BaseTool]) -> None:
    self.primary_llm = models.primary_llm
    self.llm_with_tools = models.primary_llm.bind_tools(tools)  # pyright: ignore[reportUnknownMemberType]
    self.triage_llm = models.primary_llm.with_structured_output(HandoffDecision)  # pyright: ignore[reportUnknownMemberType]
    self.tool_node = ToolNode(tools)

  def triage_agent(self, state: ResearcherState) -> TriageUpdate:
    """Initial triage to route the customer query to."""
    messages: list[BaseMessage] = [
      SystemMessage(content=TRIAGE_AGENT_PROMPT),
      *state['messages'],
    ]
    decision = cast(
      HandoffDecision,
      self.triage_llm.invoke(messages),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    )

    if decision.handoff_to == 'end':
      response = self.primary_llm.invoke(
        [
          SystemMessage(content='Provide a brief, helpful message to customer.'),
          *state['messages'],
        ]
      )
      return {
        'messages': [AIMessage(content=f'[ TRIAGE ] {response.content}')],
        'current_agent': 'end',
        'context_summary': '',
        'handoff_reason': '',
      }

    return {
      'context_summary': decision.context,
      'handoff_reason': decision.reason,
      'current_agent': decision.handoff_to,
      'messages': [
        AIMessage(
          f'[ TRIAGE ] Transferring to {decision.handoff_to}. {decision.reason}'
        )
      ],
    }

  def sales_agent(self, state: ResearcherState) -> SpecialistUpdate:
    """Sales specialist."""

    response = self.llm_with_tools.invoke(
      [
        SystemMessage(content=SALES_AGENT_PROMPT(state.get('context_summary', 'None'))),
        *state['messages'],
      ]
    )

    response.content = f'[ SALES ] {response.content}'
    return {
      'current_agent': 'sales',
      'messages': [response],
    }

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

    response.content = f'[ SUPPORT ] {response.content}'
    return {
      'current_agent': 'support',
      'messages': [response],
    }

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

    response.content = f'[ BILLING ] {response.content}'
    return {
      'current_agent': 'billing',
      'messages': [response],
    }
