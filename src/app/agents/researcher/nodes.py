from dotenv import load_dotenv
from langchain_core.messages import SystemMessage

from app.agents.researcher.prompts import SYSTEM_PROMPT
from app.agents.researcher.state import ResearcherState, ResearchUpdate
from app.common.models.models import Models

load_dotenv()


class ResearcherNodes:
  def __init__(self, models: Models) -> None:
    self.primary_llm = models.primary_llm
    self.fallback_llm = models.fallback_llm

  def research(self, state: ResearcherState) -> ResearchUpdate:
    """Call the LLM with the conversation so far."""
    response = self.primary_llm.invoke(
      [SystemMessage(content=SYSTEM_PROMPT), *state['messages']]
    )
    return {'messages': [response]}
