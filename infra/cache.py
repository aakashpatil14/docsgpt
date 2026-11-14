"""Small cache abstraction.

Uses Redis when REDIS_URL is set, otherwise a bounded in-process dict. All
operations degrade gracefully: any backend error falls back to a cache miss so
a caching problem can never break a request.
"""
import hashlib
import json
import time
from threading import Lock

import config

_backend = None
_MEM_MAX = 1000


class _MemoryCache:
    def __init__(self):
        self._d = {}
        self._lock = Lock()

    def get(self, key):
        with self._lock:
            item = self._d.get(key)
            if item is None:
                return None
            value, expires_at = item
            if expires_at is not None and time.monotonic() >= expires_at:
                del self._d[key]
                return None
            return value

    def set(self, key, value, ex=None):
        expires_at = None if ex is None else time.monotonic() + ex
        with self._lock:
            if key not in self._d and len(self._d) >= _MEM_MAX:
                self._d.pop(next(iter(self._d)))
            self._d[key] = (value, expires_at)


def _client():
    global _backend
    if _backend is None:
        if config.REDIS_URL:
            try:
                import redis
                _backend = redis.from_url(config.REDIS_URL, decode_responses=True)
                _backend.ping()
            except Exception:
                _backend = _MemoryCache()
        else:
            _backend = _MemoryCache()
    return _backend


def _key(namespace, payload):
    blob = json.dumps(payload, sort_keys=True, default=str)
    h = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return f"docsgpt:{namespace}:{h}"


def get(namespace, payload):
    if not config.CACHE_ENABLED:
        return None
    try:
        raw = _client().get(_key(namespace, payload))
        return json.loads(raw) if raw else None
    except Exception:
        return None


def set(namespace, payload, value):
    if not config.CACHE_ENABLED:
        return
    try:
        _client().set(_key(namespace, payload), json.dumps(value), ex=config.CACHE_TTL)
    except Exception:
        pass
