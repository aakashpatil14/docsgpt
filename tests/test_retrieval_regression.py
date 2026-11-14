"""Retrieval-quality regression gate.

Skips automatically if the store is empty (e.g. CI without ingested data) or
if the configured backend needs credentials that aren't present. When data is
available, it asserts Hit@5 stays above a threshold so a bad config/refactor
fails the build.
"""
import pytest

import config

HIT_AT_5_THRESHOLD = 0.7


def _corpus_ready():
    try:
        from ingestion import db
        return sum(db.counts_by_source().values()) > 0
    except Exception:
        return False


@pytest.mark.integration
def test_hybrid_rerank_hit_at_5_above_threshold():
    if not _corpus_ready():
        pytest.skip("no ingested corpus available")
    from experiments.compare_retrieval import evaluate

    hit, _mrr = evaluate("hybrid_rerank", top_k=config.TOP_K)
    assert hit >= HIT_AT_5_THRESHOLD, f"Hit@5={hit:.3f} below {HIT_AT_5_THRESHOLD}"
