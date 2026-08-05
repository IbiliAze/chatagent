from pydantic import BaseModel

from app.cache.cache import CacheStats


class CacheResponse(BaseModel):
    """Cache endpoint response"""

    semantic: CacheStats
    hash: CacheStats
