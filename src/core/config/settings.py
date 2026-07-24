from functools import lru_cache

from pydantic_settings import BaseSettings

from core.config.types import AvailableModels


class Settings(BaseSettings):
  # LLM Configuration
  openai_api_key: str
  primary_model: AvailableModels = 'gpt-4o'
  fallback_model: AvailableModels = 'gpt-4o-mini'
  embedding_model: AvailableModels = 'text-embedding-3-small'

  # LangSmith
  langchain_tracing_v2: bool = True
  langchain_api_key: str = ''
  langchain_project: str = ''

  # OpenSearch
  opensearch_url: str = 'http://localhost:9200'
  opensearch_documents_index: str = 'chatagent_documents'
  opensearch_cache_index: str = 'chatagent_cache'
  opensearch_user: str = ''
  opensearch_password: str = ''

  # Application
  app_env: str
  log_level: str
  rate_limit: str
  cache_ttl_seconds: int
  max_retries: int = 0
  token_budget: int = 4000

  model_config = {'env_file': '.env', 'extra': 'ignore'}

  @property
  def is_production(self) -> bool:
    return self.app_env == 'production'


@lru_cache
def get_settings() -> Settings:
  return Settings()  # type: ignore[call-arg]
