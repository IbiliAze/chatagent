from functools import lru_cache

from pydantic_settings import BaseSettings

from core.config.types import AvailableModels


class Settings(BaseSettings):
    # LLM Configuration
    openai_api_key: str
    anthropic_api_key: str
    primary_model: AvailableModels = 'gpt-4o-mini'
    fallback_model: AvailableModels = 'claude-sonnet-4-5-20250929'
    fallback_model_2: AvailableModels = 'gpt-4o'
    embedding_model: AvailableModels = 'text-embedding-3-small'

    # LangSmith
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ''
    langchain_project: str = ''

    # Langfuse
    langfuse_public_key: str = ''
    langfuse_secret_key: str = ''
    langfuse_base_url: str = 'https://cloud.langfuse.com'
    langfuse_tracing: bool = True

    # OpenSearch
    opensearch_url: str = 'http://localhost:9200'
    opensearch_documents_index: str = 'chatagent_documents'
    opensearch_cache_index: str = 'chatagent_cache'
    opensearch_user: str = ''
    opensearch_password: str = ''

    # MCP
    mcp_url: str
    mcp_remote_name: str
    # Bearer token for the MCP endpoint, which sits behind authenticateAny. Note the
    # server rejects expired JWTs, so this has to be a token that outlives the deploy.
    mcp_token: str = ''
    mcp_timeout_seconds: float = 30.0

    # Application
    app_env: str
    log_level: str
    rate_limit: str
    cache_ttl_seconds: int
    cache_max_entries: int = 1000
    # Raw OpenSearch kNN score, NOT a cosine similarity. The score formula depends
    # on the index's engine and space_type, so this must be recalibrated whenever
    # either changes. See SemanticCache.get.
    cache_score_threshold: float = 0.95
    # Fraction of cache_max_entries to evict down to, so that concurrent
    # maintenance passes across workers do not thrash at the boundary.
    cache_eviction_target_ratio: float = 0.9
    cache_maintenance_interval_seconds: int = 60
    max_retries: int = 0
    token_budget: int = 4000

    model_config = {'env_file': '.env', 'extra': 'ignore'}

    @property
    def is_production(self) -> bool:
        return self.app_env == 'production'


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
