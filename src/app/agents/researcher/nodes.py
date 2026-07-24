from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt.tool_node import ToolNode

from app.agents.researcher.prompts import SYSTEM_PROMPT
from app.agents.researcher.state import ResearcherState, ResearchUpdate
from app.common.models.models import Models

load_dotenv()


class ResearcherNodes:
  def __init__(self, models: Models, tools: list[BaseTool]) -> None:
    self.primary_llm = models.primary_llm.bind_tools(tools)  # pyright: ignore[reportUnknownMemberType]
    self.fallback_llm = models.fallback_llm.bind_tools(tools)  # pyright: ignore[reportUnknownMemberType]
    self.tool_node = ToolNode(tools)

  def research(self, state: ResearcherState) -> ResearchUpdate:
    """Call the LLM with the conversation so far."""
    response = self.primary_llm.invoke(
      [SystemMessage(content=SYSTEM_PROMPT), *state['messages']]
    )
    return {'messages': [response]}

  def should_continue(self, state: ResearcherState) -> Literal['tools', 'end']:
    """Check if should continue to tools or end."""
    last_message = state['messages'][-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
      return 'tools'
    return 'end'
