"""Health check endpoint."""

from api import main
from api.main import app
from api.models.health import HealthResponse
from core.config.settings import get_settings


@app.get('/health', response_model=HealthResponse)
async def health():
    """Get health"""

    settings = get_settings()

    checks = {
        'agent': hasattr(main, 'agent'),
        'security': hasattr(main, 'security'),
        'cache': hasattr(main, 'cache'),
    }

    all_healthy = all(checks.values())

    return HealthResponse(
        status='healthy' if all_healthy else 'degraded',
        environment=settings.app_env,
        checks=checks,
    )
