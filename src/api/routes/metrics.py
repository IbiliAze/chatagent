"""Metrics summary endpoint."""

from dataclasses import asdict

from fastapi import APIRouter

from api.dependencies import MetricsDep
from api.models.metrics import MetricsResponse

router = APIRouter()


@router.get('/metrics', response_model=MetricsResponse)
async def metrics_summary(metrics: MetricsDep):
    """Get metrics summary"""

    summary = metrics.get_summary()
    return MetricsResponse(**asdict(summary))
