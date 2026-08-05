# chatagent

A FastAPI chat agent built on LangGraph, with security guardrails (input sanitisation, PII detection via Presidio, language-gate, output validation), OpenSearch-backed RAG and semantic caching, LLM fallback/circuit-breaking, token-budget tracking, and SQLite-backed conversation checkpointing.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Docker (for OpenSearch, used by RAG and the semantic cache)

## Setup

```sh
uv sync
docker compose up -d   # starts OpenSearch on :9200
```

Configuration is loaded from environment variables (see `src/core/config/settings.py`); a `.env` file is supported (see `.env.example`). Required variables include OpenAI/Anthropic API keys, MCP endpoint details, and app settings like `app_env`, `log_level`, and `rate_limit`.

## Run

```sh
uv run uvicorn api.main:app --app-dir src --reload
```

Conversation state is checkpointed to a local `checkpoints.db` SQLite file.

## Tests

```sh
uv run pytest                          # unit tests
uv run pytest -m integration           # real network calls (needs credentials)
uv run pytest -m regression            # LLM quality regression suite
```

## Project layout

- `src/api/` — FastAPI app assembly (lifespan-managed agent/cache/security/metrics) and routes: `/chat`, `/health`, `/metrics`, `/cache/stats`
- `src/app/agents/researcher/` — LangGraph researcher agent (nodes, routes, tools, state)
- `src/app/security/` — input sanitiser, PII detector (Presidio/GLiNER), language detector, output validator, and the security pipeline/guard that wires them together
- `src/app/rag/` — retrieval over the OpenSearch-backed document vector store
- `src/app/cache/` — hash and semantic response caches (OpenSearch-backed), with periodic maintenance/eviction
- `src/app/error_handling/` — retry, circuit breaker, and model fallback chain around LLM calls
- `src/app/cost_optimisation/` — token-budget estimation and enforcement
- `src/app/observability/` — request timing and metrics collection (also traced via LangSmith/Langfuse)
- `src/app/mcp/` — MCP client used by agent tools
- `src/app/evaluation/` — LLM-judge based evaluation
- `src/core/` — configuration, logging, models, and the OpenSearch vector store
- `docker-compose.yml` — local OpenSearch instance
- `tests/` — unit, integration, and regression suites, mirroring the `src/app/` layout
