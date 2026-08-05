"""FastAPI app assembly: lifespan-managed components, limiter, and rate-limit handler."""

import asyncio
from contextlib import ExitStack, asynccontextmanager, suppress

from fastapi import FastAPI, Request
from langgraph.checkpoint.sqlite import SqliteSaver
from slowapi.errors import RateLimitExceeded

from api.dependencies import AvailableCache
from api.limiter import limiter
from api.routes.cache import router as cache_router
from api.routes.chat import router as chat_router
from api.routes.health import router as health_router
from api.routes.knowledge import router as knowledge_router
from api.routes.metrics import router as metrics_router
from app.agents.researcher.agent import ResearcherAgent
from app.agents.researcher.nodes import ResearcherNodes
from app.agents.researcher.routes import ResearcherRoutes
from app.agents.researcher.tools import ResearcherTools
from app.cache.hash_cache import HashCache
from app.cache.semantic_cache import SemanticCache
from app.cost_optimisation.token_budget import TokenBudget
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

opensearch = OpenSearch()
opensearch.provision_indexes(embedding_dimension=1536)
rag = Rag(opensearch.document_vectorstore)


def _build_security_pipeline(models: Models) -> SecurityPipeline:
    """Assemble the security pipeline from its component checks."""
    pii_detector = PIIDetector()
    return SecurityPipeline(
        pii_detector=pii_detector,
        output_validator=OutputValidator(pii_detector=pii_detector),
        security_guard=SecurityGuard(llm=models.primary_llm),
        input_sanitiser=InputSanitiser(),
        language_detector=LanguageDetector(),
    )


def _build_agent(
    models: Models, rag: Rag, mcp_client: McpClient, saver: SqliteSaver
) -> ResearcherAgent:
    """Assemble the researcher agent from its nodes, routes, and tools."""
    tools = ResearcherTools(rag=rag, mcp_client=mcp_client)
    nodes = ResearcherNodes(models=models, tools=tools.load_tools())
    return ResearcherAgent(nodes=nodes, routes=ResearcherRoutes(), saver=saver)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise all components and hang them off `app.state` for routes to read."""

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

    app.state.rag = rag
    app.state.token_budget = TokenBudget()
    app.state.security = _build_security_pipeline(models)

    db_path = 'checkpoints.db'

    app.state.cache = AvailableCache(
        semantic=SemanticCache(vectorstore=opensearch.cache_vectorstore),
        hash=HashCache(),
    )
    app.state.metrics = MetricsCollector()

    # from_conn_string is a context manager: it must stay open for the lifetime of
    # the app, otherwise the underlying sqlite connection is closed under the agent.
    with ExitStack() as stack:
        saver = stack.enter_context(SqliteSaver.from_conn_string(db_path))
        saver.setup()
        app.state.agent = _build_agent(
            models=models, rag=rag, mcp_client=mcp_client, saver=saver
        )

        maintenance = asyncio.create_task(app.state.cache.semantic.maintenance())

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
                await app.state.cache.semantic.run_maintenance_once()
            except Exception:  # pylint: disable=broad-exception-caught
                logger.exception('Final cache maintenance failed')

            logger.info(
                'Shutting down...', extra={'extra_data': app.state.metrics.get_summary()}
            )


app = FastAPI(
    title='ChatAgent API',
    description='Fast API for ChatAgent',
    version='0.1.0',
    lifespan=lifespan,
)
app.state.limiter = limiter

app.include_router(chat_router)
app.include_router(health_router)
app.include_router(metrics_router)
app.include_router(cache_router)
app.include_router(knowledge_router)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(  # pylint: disable=unused-argument
    request: Request, exc: RateLimitExceeded
):
    """Handle requests that exceed the rate limit."""
