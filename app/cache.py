"""Versioned Redis cache for search responses.

Keys embed a namespace version; ingestion bumps the version, so every
previously cached response becomes unreachable at once — no per-key
invalidation, no stale results after a write. Entries also carry a short
TTL, which bounds memory since orphaned versions simply expire.

The cache fails open: if Redis is down, search still works (uncached) and
only /health reports the outage.
"""

import json
import logging
from functools import lru_cache

import redis

from app.config import get_settings

logger = logging.getLogger(__name__)

_VERSION_KEY = "novasearch:search:version"
_ENTRY_PREFIX = "novasearch:search:entry"


class SearchCache:
    def __init__(self, client: redis.Redis, ttl_seconds: int) -> None:
        self._client = client
        self._ttl_seconds = ttl_seconds

    def get(self, *, mode: str, query: str, limit: int) -> dict | None:
        try:
            raw = self._client.get(self._key(mode=mode, query=query, limit=limit))
        except redis.RedisError:
            logger.warning("Redis unavailable; serving search uncached", exc_info=True)
            return None

        return json.loads(raw) if raw is not None else None

    def put(self, *, mode: str, query: str, limit: int, payload: dict) -> None:
        try:
            self._client.set(
                self._key(mode=mode, query=query, limit=limit),
                json.dumps(payload),
                ex=self._ttl_seconds,
            )
        except redis.RedisError:
            logger.warning("Redis unavailable; search response not cached", exc_info=True)

    def invalidate_all(self) -> None:
        """Make every cached search response unreachable (namespace bump)."""
        try:
            self._client.incr(_VERSION_KEY)
        except redis.RedisError:
            logger.warning("Redis unavailable; cache not invalidated", exc_info=True)

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except redis.RedisError:
            return False

    def _key(self, *, mode: str, query: str, limit: int) -> str:
        version = self._client.get(_VERSION_KEY) or "0"
        return f"{_ENTRY_PREFIX}:{version}:{mode}:{limit}:{query}"


@lru_cache
def get_search_cache() -> SearchCache:
    settings = get_settings()
    client = redis.Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_timeout=1,
        socket_connect_timeout=1,
    )
    return SearchCache(client, ttl_seconds=settings.search_cache_ttl_seconds)
