"""Metrics summary endpoint."""

from dataclasses import asdict

from api.main import app, metrics
from api.models.metrics import MetricsResponse


@app.post('/metrics', response_model=MetricsResponse)
async def metrics_summary():
    """Get metrics summary"""

    summary = metrics.get_summary()
    return MetricsResponse(**asdict(summary))
