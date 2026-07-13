import config

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(config.RERANK_MODEL)
    return _model


def rerank(query, candidates, top_k=config.TOP_K):
    if not candidates:
        return []
    model = _get_model()
    pairs = [(query, c["content"]) for c in candidates]
    scores = model.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]
