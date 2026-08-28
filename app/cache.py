"""Search-response cache: PostgreSQL owns the epoch, Redis holds payloads.

Cache keys embed an invalidation epoch that lives in PostgreSQL (the
single-row ``cache_epoch`` table) and is incremented in the same
transaction as every successful ingestion. A search captures the epoch
ONCE and uses it for both its lookup and its write, so a write that races
an ingestion lands under the epoch the request started with — already
retired by the ingestion's commit — and can never be served again.

Because the epoch never lives in Redis, a Redis outage cannot lose an
invalidation: ingestion during the outage still commits its epoch bump
with the data, and when Redis comes back, every entry written before the
outage sits under a retired epoch — physically present until its TTL
expires, but unreachable. Redis is purely a disposable cache: its
failures on read or write fail open (search runs uncached), and only
/health reports the outage.
"""

import json
import logging
from functools import lru_cache

import redis
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import CacheEpoch

logger = logging.getLogger(__name__)

_ENTRY_PREFIX = "novasearch:search:entry"


def current_epoch(session: Session) -> int:
    """The invalidation epoch for one request's cache get/put pair.

    Capture this once per search and pass the same value to both
    :meth:`SearchCache.get` and :meth:`SearchCache.put`.
    """
    return session.execute(select(CacheEpoch.epoch)).scalar_one()


def bump_epoch(session: Session) -> None:
    """Retire every cached search response, atomically with the caller.

    Runs in the caller's transaction: the bump commits if and only if the
    ingestion commits, so there is no window where new data is visible
    under an epoch that still serves pre-ingestion cache entries.
    """
    session.execute(update(CacheEpoch).values(epoch=CacheEpoch.epoch + 1))


class SearchCache:
    def __init__(self, client: redis.Redis, ttl_seconds: int) -> None:
        self._client = client
        self._ttl_seconds = ttl_seconds

    def get(self, *, epoch: int, mode: str, query: str, limit: int) -> dict | None:
        try:
            raw = self._client.get(self._key(epoch=epoch, mode=mode, query=query, limit=limit))
        except redis.RedisError:
            logger.warning("Redis unavailable; serving search uncached", exc_info=True)
            return None

        return json.loads(raw) if raw is not None else None

    def put(self, *, epoch: int, mode: str, query: str, limit: int, payload: dict) -> None:
        try:
            self._client.set(
                self._key(epoch=epoch, mode=mode, query=query, limit=limit),
                json.dumps(payload),
                ex=self._ttl_seconds,
            )
        except redis.RedisError:
            logger.warning("Redis unavailable; search response not cached", exc_info=True)

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except redis.RedisError:
            return False

    def _key(self, *, epoch: int, mode: str, query: str, limit: int) -> str:
        return f"{_ENTRY_PREFIX}:{epoch}:{mode}:{limit}:{query}"


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
