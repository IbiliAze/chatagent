from api.main import app, cache


@app.post(
  '/cache/stats',
)
async def cache_stats():
  """Get cache stats"""

  return cache.get_stats()
