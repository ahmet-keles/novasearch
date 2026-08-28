"""Versioned Redis cache for search responses.

Keys embed a namespace version; ingestion bumps the version, so every
previously cached response becomes unreachable at once — no per-key
invalidation, no stale results after a write. Entries also carry a short
TTL, which bounds memory since orphaned versions simply expire.

A request captures the version ONCE (:meth:`SearchCache.current_version`)
and uses it for both its lookup and its write. This closes the
miss-then-write race: if an ingestion bumps the namespace while a search
is running, the search's late write lands under the version it captured —
the old, already-retired namespace — and can never be served to a search
that starts after the invalidation.

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

    def current_version(self) -> str | None:
        """The namespace version for one request's get/put pair.

        Capture this once per request and pass the same value to both
        :meth:`get` and :meth:`put`. Returns None when Redis is
        unavailable, which makes both operations no-ops (fail open).
        """
        try:
            return self._client.get(_VERSION_KEY) or "0"
        except redis.RedisError:
            logger.warning("Redis unavailable; serving search uncached", exc_info=True)
            return None

    def get(
        self, *, version: str | None, mode: str, query: str, limit: int
    ) -> dict | None:
        if version is None:
            return None

        try:
            raw = self._client.get(self._key(version=version, mode=mode, query=query, limit=limit))
        except redis.RedisError:
            logger.warning("Redis unavailable; serving search uncached", exc_info=True)
            return None

        return json.loads(raw) if raw is not None else None

    def put(
        self, *, version: str | None, mode: str, query: str, limit: int, payload: dict
    ) -> None:
        if version is None:
            return

        try:
            self._client.set(
                self._key(version=version, mode=mode, query=query, limit=limit),
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

    def _key(self, *, version: str, mode: str, query: str, limit: int) -> str:
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
