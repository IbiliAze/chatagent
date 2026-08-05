"""Cache stats endpoint."""

from fastapi import APIRouter

from api.dependencies import CacheDep
from api.models.cache import CacheResponse

router = APIRouter()


@router.get('/cache/stats', response_model=CacheResponse)
async def cache_stats(cache: CacheDep) -> CacheResponse:
    """Get cache stats"""

    return CacheResponse(
        semantic=await cache.semantic.get_stats(),
        hash=await cache.hash.get_stats(),
    )
