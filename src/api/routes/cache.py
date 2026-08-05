"""Cache stats endpoint."""

from fastapi import APIRouter

from api import main
from api.models.cache import CacheResponse

router = APIRouter()


@router.get('/cache/stats', response_model=CacheResponse)
async def cache_stats() -> CacheResponse:
    """Get cache stats"""

    return CacheResponse(
        semantic=await main.cache.semantic.get_stats(),
        hash=await main.cache.hash.get_stats(),
    )
