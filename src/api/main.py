from contextlib import ExitStack, asynccontextmanager

from fastapi import FastAPI, Request
from langgraph.checkpoint.sqlite import SqliteSaver
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.agents.researcher.agent import ResearcherAgent
from app.agents.researcher.nodes import ResearcherNodes
from app.agents.researcher.routes import ResearcherRoutes
from app.agents.researcher.tools import ResearcherTools
from app.common.observability.metrics_collector import MetricsCollector
from app.common.rag.rag import Rag
from core.cache.hash_cache import HashCache
from core.cache.semantic_cache import SemanticCache
from core.config.settings import get_settings
from core.logging.logger import logger
from core.models.models import Models
from core.store.opensearch.opensearch import OpenSearch


@asynccontextmanager
async def lifespan(app: FastAPI):
  """Initialise all components."""

  global security, cache, metrics, agent

  settings = get_settings()

  logger.info(
    'Starting API',
    extra={
      'extra_data': {
        'environment': settings.app_env,
        'primary_model': settings.primary_model,
        'fallback_model': settings.fallback_model,
        'tracing_enabled': settings.langchain_tracing_v2,
      }
    },
  )

  models = Models()
  opensearch = OpenSearch()
  rag = Rag(opensearch.document_vectorstore)

  db_path = 'checkpoints.db'
  routes = ResearcherRoutes()
  tools = ResearcherTools(rag=rag)
  tool_list = [tools.get_relevant_documents]
  nodes = ResearcherNodes(tools=tool_list, models=models)

  class AvailableCache:
    semantic = SemanticCache(vectorstore=opensearch.cache_vectorstore)
    hash = HashCache()

  cache = AvailableCache()
  metrics = MetricsCollector()

  # from_conn_string is a context manager: it must stay open for the lifetime of
  # the app, otherwise the underlying sqlite connection is closed under the agent.
  with ExitStack() as stack:
    saver = stack.enter_context(SqliteSaver.from_conn_string(db_path))
    saver.setup()
    agent = ResearcherAgent(nodes=nodes, routes=routes, saver=saver)

    logger.info('All components initialised')

    yield

    logger.info('Shutting down...', extra={'extra_data': metrics.get_summary()})


limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
  title='ChatAgent API',
  description='Fast API for ChatAgent',
  version='0.1.0',
  lifespan=lifespan,
)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
  pass
