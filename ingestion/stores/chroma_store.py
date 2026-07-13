import chromadb

import config

_collection = None
_bm25 = None
_bm25_ids = None
_records = None


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        _collection = client.get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def init_schema():
    get_collection()


def _clean_meta(meta):
    out = {}
    for k, v in (meta or {}).items():
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, list):
            out[k] = ", ".join(map(str, v))
        elif v is None:
            out[k] = ""
        else:
            out[k] = str(v)
    return out


def upsert_chunks(rows):
    col = get_collection()
    unique = {r["content_hash"]: r for r in rows}
    rows = list(unique.values())
    ids = [r["content_hash"] for r in rows]

    existing = set()
    for start in range(0, len(ids), config.CHROMA_UPSERT_BATCH):
        got = col.get(ids=ids[start:start + config.CHROMA_UPSERT_BATCH])
        existing.update(got["ids"])
    new_count = len([i for i in ids if i not in existing])

    for start in range(0, len(rows), config.CHROMA_UPSERT_BATCH):
        batch = rows[start:start + config.CHROMA_UPSERT_BATCH]
        col.upsert(
            ids=[r["content_hash"] for r in batch],
            embeddings=[r["embedding"] for r in batch],
            documents=[r["content"] for r in batch],
            metadatas=[
                _clean_meta({
                    "source_type": r["source_type"],
                    "title": r["title"] or "",
                    "url": r["url"] or "",
                    "chunk_index": r["chunk_index"],
                    **(r["metadata"] or {}),
                })
                for r in batch
            ],
        )
    return new_count


def vector_query(query_emb, limit):
    col = get_collection()
    res = col.query(
        query_embeddings=[query_emb],
        n_results=limit,
        include=["documents", "metadatas", "distances"],
    )
    out = []
    for i, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0],
        res["metadatas"][0], res["distances"][0],
    ):
        out.append({
            "id": i, "content": doc,
            "title": meta.get("title", ""), "url": meta.get("url", ""),
            "source_type": meta.get("source_type", ""),
            "similarity": 1.0 - dist,
        })
    return out


def _load_bm25():
    global _bm25, _bm25_ids, _records
    if _bm25 is not None:
        return
    from rank_bm25 import BM25Okapi

    col = get_collection()
    data = col.get(include=["documents", "metadatas"])
    _bm25_ids = data["ids"]
    _records = {}
    for i, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        _records[i] = {
            "id": i, "content": doc,
            "title": meta.get("title", ""), "url": meta.get("url", ""),
            "source_type": meta.get("source_type", ""),
        }
    _bm25 = BM25Okapi([d.lower().split() for d in data["documents"]])


def keyword_query(query, limit):
    _load_bm25()
    scores = _bm25.get_scores(query.lower().split())
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [_records[_bm25_ids[i]] for i in ranked[:limit] if scores[i] > 0]


def counts_by_source():
    col = get_collection()
    metas = col.get(include=["metadatas"])["metadatas"]
    counts = {}
    for m in metas:
        st = m.get("source_type", "?")
        counts[st] = counts.get(st, 0) + 1
    return counts
