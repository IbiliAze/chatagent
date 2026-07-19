from langchain_core.messages import AnyMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.researcher.prompts import SYSTEM_PROMPT
from agents.researcher.state import ResearcherState


def research(state: ResearcherState) -> dict[str, list[AnyMessage]]:
  """Call the LLM with the conversation so far."""
  llm = ChatOpenAI(model='gpt-4o-mini')
  response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), *state['messages']])
  return {'messages': [response]}
