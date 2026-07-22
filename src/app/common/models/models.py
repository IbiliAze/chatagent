from langchain_openai import ChatOpenAI

from core.config.settings import get_settings


class Models:
  def __init__(self) -> None:
    settings = get_settings()

    self.primary_llm = ChatOpenAI(
      model=settings.primary_model,
      temperature=0,
      max_retries=0,
      timeout=30,
    )

    self.fallback_llm = ChatOpenAI(
      model=settings.fallback_model,
      temperature=0,
      max_retries=0,
      timeout=30,
    )
