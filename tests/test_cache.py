import config
from infra import cache


def test_memory_expiry_at_ttl_boundary(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(cache.time, "monotonic", lambda: clock[0])
    backend = cache._MemoryCache()
    backend.set("key", "value", ex=10)
    clock[0] = 109.9
    assert backend.get("key") == "value"
    clock[0] = 110.0
    assert backend.get("key") is None


def test_memory_overwrite_refreshes_expiry(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(cache.time, "monotonic", lambda: clock[0])
    backend = cache._MemoryCache()
    backend.set("key", "old", ex=10)
    clock[0] = 5
    backend.set("key", "new", ex=10)
    clock[0] = 10
    assert backend.get("key") == "new"
    clock[0] = 15
    assert backend.get("key") is None


def test_memory_capacity_and_overwrite(monkeypatch):
    monkeypatch.setattr(cache, "_MEM_MAX", 2)
    backend = cache._MemoryCache()
    backend.set("a", "one")
    backend.set("b", "two")
    backend.set("b", "updated")
    assert backend.get("a") == "one"
    backend.set("c", "three")
    assert backend.get("a") is None
    assert backend.get("b") == "updated"
    assert backend.get("c") == "three"


def test_cache_errors_degrade_to_misses(monkeypatch):
    class BrokenCache:
        def get(self, *args):
            raise ConnectionError("offline")

        def set(self, *args, **kwargs):
            raise ConnectionError("offline")

    monkeypatch.setattr(config, "CACHE_ENABLED", True)
    monkeypatch.setattr(cache, "_backend", BrokenCache())
    assert cache.get("answer", {"q": "q"}) is None
    cache.set("answer", {"q": "q"}, {"answer": "text"})
