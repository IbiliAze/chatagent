from typing import Literal

from pydantic import BaseModel

from api.main import agent, app, cache, security
from core.config.settings import get_settings


class HealthResponse(BaseModel):
  """Health check response"""

  status: Literal['healthy', 'degraded'] = 'healthy'
  environment: str
  version: str = '1.0.0'
  checks: dict[str, bool] = {}


@app.post('/health', response_model=HealthResponse)
async def health():
  """Get health"""

  settings = get_settings()

  checks = {
    'agent': agent is not None,
    'security': security is not None,
    'cache': cache is not None,
  }

  all_healthy = all(checks.values())

  return HealthResponse(
    status='healthy' if all_healthy else 'degraded',
    environment=settings.app_env,
    checks=checks,
  )
