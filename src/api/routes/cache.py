from api import main
from api.main import app
from api.models.cache import CacheResponse


@app.post('/cache/stats', response_model=CacheResponse)
async def cache_stats() -> CacheResponse:
  """Get cache stats"""

  return CacheResponse(
    semantic=await main.cache.semantic.get_stats(),
    hash=await main.cache.hash.get_stats(),
  )
