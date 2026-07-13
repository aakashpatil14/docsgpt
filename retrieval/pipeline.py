import config
from retrieval.hybrid import hybrid_search, vector_search
from retrieval.rerank import rerank

STRATEGIES = ("vector", "hybrid", "hybrid_rerank")


def retrieve(query, strategy="hybrid_rerank", top_k=config.TOP_K):
    """Return the top_k chunks for a query using the chosen strategy.

    - vector:        dense similarity only
    - hybrid:        dense + keyword fused with RRF
    - hybrid_rerank: hybrid candidates re-scored by a cross-encoder
    """
    if strategy == "vector":
        return vector_search(query, limit=top_k)
    if strategy == "hybrid":
        return hybrid_search(query, limit=top_k)
    if strategy == "hybrid_rerank":
        candidates = hybrid_search(query, limit=config.HYBRID_LIMIT)
        return rerank(query, candidates, top_k=top_k)
    raise ValueError(f"unknown strategy: {strategy}")
