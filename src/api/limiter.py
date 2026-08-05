"""Shared rate limiter.

Defined separately from `api.main` so route modules can use the
`@limiter.limit(...)` decorator without importing `api.main` (and hitting its
import-time OpenSearch/model setup) just to reach the limiter instance.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
