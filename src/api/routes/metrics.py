"""Metrics summary endpoint."""

from dataclasses import asdict

from fastapi import APIRouter

from api import main
from api.models.metrics import MetricsResponse

router = APIRouter()


@router.get('/metrics', response_model=MetricsResponse)
async def metrics_summary():
    """Get metrics summary"""

    summary = main.metrics.get_summary()
    return MetricsResponse(**asdict(summary))
