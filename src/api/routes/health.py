"""Health check endpoint."""

from fastapi import APIRouter, Request

from api.models.health import HealthResponse
from core.config.settings import get_settings

router = APIRouter()


@router.get('/health', response_model=HealthResponse)
async def health(request: Request):
    """Get health"""

    settings = get_settings()

    checks = {
        'agent': hasattr(request.app.state, 'agent'),
        'security': hasattr(request.app.state, 'security'),
        'cache': hasattr(request.app.state, 'cache'),
    }

    all_healthy = all(checks.values())

    return HealthResponse(
        status='healthy' if all_healthy else 'degraded',
        environment=settings.app_env,
        checks=checks,
    )
