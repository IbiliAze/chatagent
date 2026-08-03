import asyncio
from contextlib import ExitStack, asynccontextmanager, suppress

from fastapi import FastAPI, Request
from langgraph.checkpoint.sqlite import SqliteSaver
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.agents.researcher.agent import ResearcherAgent
from app.agents.researcher.nodes import ResearcherNodes
from app.agents.researcher.routes import ResearcherRoutes
from app.agents.researcher.tools import ResearcherTools
from app.cache.hash_cache import HashCache
from app.cache.semantic_cache import SemanticCache
from app.mcp.mcp_client import McpClient
from app.observability.metrics_collector import MetricsCollector
from app.rag.rag import Rag
from app.security.input_sanitiser import InputSanitiser
from app.security.language_detector import LanguageDetector
from app.security.output_validator import OutputValidator
from app.security.pii_detector.pii_detector import PIIDetector
from app.security.security_guard import SecurityGuard
from app.security.security_pipeline import SecurityPipeline
from core.config.settings import get_settings
from core.logging.logger import logger
from core.models.models import Models
from core.store.vectorstore.opensearch import OpenSearch


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

  mcp_client = McpClient(name='eightmile')

  input_sanitiser = InputSanitiser()
  pii_detector = PIIDetector()
  language_detector = LanguageDetector()
  security_guard = SecurityGuard(llm=models.primary_llm)
  output_validator = OutputValidator(pii_detector=pii_detector)
  security = SecurityPipeline(
    pii_detector=pii_detector,
    output_validator=output_validator,
    security_guard=security_guard,
    input_sanitiser=input_sanitiser,
    language_detector=language_detector,
  )
  opensearch = OpenSearch()
  opensearch.provision_indexes(embedding_dimension=1536)
  rag = Rag(opensearch.document_vectorstore)

  db_path = 'checkpoints.db'
  routes = ResearcherRoutes()
  tools = ResearcherTools(rag=rag, mcp_client=mcp_client)
  tool_list = tools.load_tools()
  nodes = ResearcherNodes(models=models, tools=tool_list)

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

    maintenance = asyncio.create_task(cache.semantic.maintenance())

    logger.info('All components initialised')

    try:
      yield
    finally:
      maintenance.cancel()

      with suppress(asyncio.CancelledError):
        await maintenance

      # Flush here rather than from the task's cancel handler, where a second
      # cancellation or interpreter teardown can interrupt the await.
      try:
        await cache.semantic.run_maintenance_once()
      except Exception:
        logger.exception('Final cache maintenance failed')

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
