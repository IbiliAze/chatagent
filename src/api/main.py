import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from langsmith import traceable
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.agents.researcher.agent import ResearcherAgent
from app.common.cache.semantic_cache import SemanticCache
from app.common.observability.metrics_collector import MetricsCollector
from core.config.settings import get_settings
from core.logging.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
  """Initialise all components"""

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

  agent = ResearcherAgent()
  cache = SemanticCache()
  metrics = MetricsCollector()

  logger.info('All components initialised')

  yield

  logger.info('Shutting down...', extra={'extra_data': {metrics.get_summary()}})


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
