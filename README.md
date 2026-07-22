# chatagent

A FastAPI chat agent built on LangGraph, with security guardrails (input sanitisation, PII detection via Presidio, output validation) and Postgres-backed conversation checkpointing.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## Setup

```sh
uv sync
```

Configuration is loaded from environment variables (see `src/core/config/settings.py`); a `.env` file is supported.

## Run

```sh
uv run uvicorn api.main:app --app-dir src --reload
```

## Tests

```sh
uv run pytest                          # unit tests
uv run pytest -m integration           # real network calls (needs credentials)
uv run pytest -m regression            # LLM quality regression suite
```

## Project layout

- `src/api/` — FastAPI app and routes (chat, health, metrics, cache)
- `src/app/security/` — input sanitiser, PII detector, output validator, security guard
- `src/core/` — configuration and logging
- `tests/` — unit, integration, and regression suites
