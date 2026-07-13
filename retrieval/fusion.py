def reciprocal_rank_fusion(ranked_id_lists, k=60):
    """Fuse multiple ranked id lists into a single {id: score} map.

    Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank(d)).
    Pure function — no I/O — so it can be unit-tested in isolation.
    """
    scores = {}
    for ids in ranked_id_lists:
        for rank, doc_id in enumerate(ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores
