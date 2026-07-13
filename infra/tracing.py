"""Optional Langfuse tracing. No-op unless both Langfuse keys are set.

Any tracing failure is swallowed so it can never break a request.
"""
import config

_client = None
_initialized = False


def _get_client():
    global _client, _initialized
    if _initialized:
        return _client
    _initialized = True
    if config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY:
        try:
            from langfuse import Langfuse
            _client = Langfuse(
                public_key=config.LANGFUSE_PUBLIC_KEY,
                secret_key=config.LANGFUSE_SECRET_KEY,
                host=config.LANGFUSE_HOST,
            )
        except Exception:
            _client = None
    return _client


def trace_ask(question, strategy, result, latency_ms, cache_hit=False):
    client = _get_client()
    if client is None:
        return
    try:
        client.trace(
            name="ask",
            input={"question": question, "strategy": strategy},
            output={"answer": result.get("answer", "")},
            metadata={
                "citations": [c["url"] for c in result.get("citations", [])],
                "used_chunks": result.get("used_chunks", 0),
                "latency_ms": round(latency_ms, 1),
                "cache_hit": cache_hit,
                "llm_model": config.LLM_MODEL,
                "embedding_backend": config.EMBEDDING_BACKEND,
                "store_backend": config.STORE_BACKEND,
            },
        )
    except Exception:
        pass
