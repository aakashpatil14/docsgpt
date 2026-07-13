import config
from ingestion import db
from ingestion.embed import embed_query
from retrieval.fusion import reciprocal_rank_fusion


def vector_search(query, limit=config.TOP_K):
    return db.vector_query(embed_query(query), limit)


def hybrid_search(query, limit=config.HYBRID_LIMIT, candidate_k=config.CANDIDATE_K):
    """Vector + keyword retrieval fused with Reciprocal Rank Fusion.

    Backend-agnostic: works over Postgres (tsvector) or Chroma (BM25).
    """
    q_emb = embed_query(query)
    vec = db.vector_query(q_emb, candidate_k)
    kw = db.keyword_query(query, candidate_k)

    record = {}
    for r in vec + kw:
        record.setdefault(r["id"], r)

    scores = reciprocal_rank_fusion(
        [[r["id"] for r in vec], [r["id"] for r in kw]], k=config.RRF_K
    )

    top_ids = sorted(scores, key=scores.get, reverse=True)[:limit]
    results = []
    for doc_id in top_ids:
        rec = dict(record[doc_id])
        rec["rrf_score"] = scores[doc_id]
        results.append(rec)
    return results
