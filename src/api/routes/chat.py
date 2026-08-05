"""Main chat endpoint: security checks, caching, agent invocation, and metrics."""

from datetime import datetime, timezone

from fastapi import HTTPException, Request
from langsmith import traceable  # pyright: ignore[reportUnknownVariableType]

from api.main import agent, app, cache, limiter, metrics, security, token_budget
from api.models.chat import ChatRequest, ChatResponse
from app.observability.request_timer import RequestTimer
from core.config.settings import get_settings
from core.logging.logger import logger

settings = get_settings()


@app.post('/chat', response_model=ChatResponse)
@limiter.limit(settings.rate_limit)  # pyright: ignore[reportUntypedFunctionDecorator, reportUnknownMemberType]
@traceable(name='chat_endpoint')
async def chat(request: Request, body: ChatRequest):
    """Main chat endpoint."""

    with RequestTimer() as timer:
        thread_id = body.thread_id
        security_notes: list[str] = []

        # 1. Security check
        input_result = security.check_input(body.message)
        security_notes.extend(input_result.security_notes)

        if not input_result.is_allowed:
            logger.warning(
                'Request blocked by security check',
                extra={'reason': input_result.security_notes, 'thread_id': thread_id},
            )

            metrics.record_request(
                latency_ms=0,
                error=True,
                cache_hit=False,
                output_tokens=0,
                input_tokens=0,
            )

            raise HTTPException(
                status_code=400,
                detail='Your message was blocked by our security policy.',
            )

        input_text = input_result.cleaned_text

        # 2. Cache lookup
        entry = await cache.hash.get(input_text, thread_id)
        if entry is None:
            entry = await cache.semantic.get(input_text, thread_id)

        if entry is not None:
            logger.info(
                'Cache hit',
                extra={'thread_id': thread_id},
            )

            metrics.record_request(
                cache_hit=True,
                error=False,
                latency_ms=0,
                output_tokens=0,
                input_tokens=0,
            )

            response = entry['response'] if isinstance(entry, dict) else entry

            return ChatResponse(
                thread_id=thread_id,
                cached=True,
                model_used='cache',
                response=response,
                processing_time_ms=0,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        # 3. Invoke the agent
        try:
            config = agent.build_config(thread_id=thread_id)
            state = agent.build_message(text=input_text)
            output = agent.process_message(state=state, config=config)

        except Exception as e:
            logger.error(
                f'Agent invokation failed: {e}',
                extra={'error': str(e), thread_id: thread_id},
            )

            metrics.record_request(
                cache_hit=False,
                error=True,
                latency_ms=0,
                output_tokens=0,
                input_tokens=0,
            )

            raise HTTPException(
                status_code=500,
                detail='An error occured while processing your request.',
            )

        output_text = str(output['messages'][-1])
        model_used = output.get('model_used', settings.primary_model)

        # 4. Output validation
        output_result = security.check_output(output_text)
        if output_result.reason:
            security_notes.append(output_result.reason)

        # 5. Cache response
        await cache.hash.set(
            query=input_text, thread_id=thread_id, response=output_text
        )
        await cache.semantic.set(
            query=input_text, thread_id=thread_id, response=output_text
        )

    # 6. Log and record metrics
    input_tokens = token_budget.estimate_tokens(text=input_text, model=model_used)
    output_tokens = token_budget.estimate_tokens(text=output_text, model=model_used)
    metrics.record_request(
        latency_ms=timer.elapsed_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_hit=False,
    )

    logger.info(
        'Request completed',
        extra={
            'thread_id': thread_id,
            'model_used': model_used,
            'latency_ms': round(timer.elapsed_ms, 2),
        },
    )

    return ChatResponse(
        model_used=model_used,
        thread_id=thread_id,
        response=output_text,
        cached=False,
        timestamp=datetime.now(timezone.utc).isoformat(),
        processing_time_ms=timer.elapsed_ms,
    )
