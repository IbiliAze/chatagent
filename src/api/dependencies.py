"""FastAPI dependency providers for objects assembled during the app lifespan.

`api.main`'s `lifespan()` stores these on `app.state`; routes pull them back out
through `Request.app.state` rather than importing `api.main` module globals, so
route modules stay import-order-independent of `api.main`.
"""

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request

from app.agents.researcher.agent import ResearcherAgent
from app.cache.hash_cache import HashCache
from app.cache.semantic_cache import SemanticCache
from app.cost_optimisation.token_budget import TokenBudget
from app.observability.metrics_collector import MetricsCollector
from app.rag.rag import Rag
from app.security.security_pipeline import SecurityPipeline


@dataclass(frozen=True)
class AvailableCache:
    """The caches the app exposes, assembled at startup."""

    semantic: SemanticCache
    hash: HashCache


def get_agent(request: Request) -> ResearcherAgent:
    """Resolve the researcher agent assembled in the app lifespan."""
    return request.app.state.agent


def get_cache(request: Request) -> AvailableCache:
    """Resolve the hash/semantic caches assembled in the app lifespan."""
    return request.app.state.cache


def get_security(request: Request) -> SecurityPipeline:
    """Resolve the security pipeline assembled in the app lifespan."""
    return request.app.state.security


def get_metrics(request: Request) -> MetricsCollector:
    """Resolve the metrics collector assembled in the app lifespan."""
    return request.app.state.metrics


def get_token_budget(request: Request) -> TokenBudget:
    """Resolve the token budget tracker assembled in the app lifespan."""
    return request.app.state.token_budget


def get_rag(request: Request) -> Rag:
    """Resolve the RAG instance assembled in the app lifespan."""
    return request.app.state.rag


AgentDep = Annotated[ResearcherAgent, Depends(get_agent)]
CacheDep = Annotated[AvailableCache, Depends(get_cache)]
SecurityDep = Annotated[SecurityPipeline, Depends(get_security)]
MetricsDep = Annotated[MetricsCollector, Depends(get_metrics)]
TokenBudgetDep = Annotated[TokenBudget, Depends(get_token_budget)]
RagDep = Annotated[Rag, Depends(get_rag)]
