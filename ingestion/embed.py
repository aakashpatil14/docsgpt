import time

import config

_openai_client = None
_st_model = None


def _openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _openai_client


def _local():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(config.EMBED_MODEL)
    return _st_model


def _embed_openai(texts):
    client = _openai()
    out = []
    for start in range(0, len(texts), config.EMBED_BATCH):
        batch = texts[start:start + config.EMBED_BATCH]
        for attempt in range(5):
            try:
                resp = client.embeddings.create(
                    model=config.EMBED_MODEL, input=batch
                )
                out.extend([d.embedding for d in resp.data])
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"[embed] retry in {wait}s ({e})")
                time.sleep(wait)
        else:
            raise RuntimeError("embedding failed after retries")
        print(f"[embed] {min(start + config.EMBED_BATCH, len(texts))}/{len(texts)}")
    return out


def _embed_local(texts):
    model = _local()
    embs = model.encode(
        texts,
        batch_size=config.EMBED_BATCH,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return [e.tolist() for e in embs]


def embed_texts(texts):
    """Embed documents/chunks."""
    if config.EMBEDDING_BACKEND == "openai":
        return _embed_openai(texts)
    return _embed_local(texts)


def embed_query(query):
    """Embed a single search query (cached)."""
    from infra import cache

    payload = {"m": config.EMBED_MODEL, "b": config.EMBEDDING_BACKEND, "q": query}
    hit = cache.get("query_emb", payload)
    if hit is not None:
        return hit

    if config.EMBEDDING_BACKEND == "openai":
        emb = _embed_openai([query])[0]
    else:
        model = _local()
        emb = model.encode(
            [config.QUERY_PREFIX + query], normalize_embeddings=True
        )[0].tolist()

    cache.set("query_emb", payload, emb)
    return emb
