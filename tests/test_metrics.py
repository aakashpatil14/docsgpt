from experiments.metrics import hit_at_k, reciprocal_rank
from experiments.testset import is_relevant


def test_is_relevant_matches_substring():
    assert is_relevant("https://x/tutorial/path-params/", ["path-params"])
    assert not is_relevant("https://x/tutorial/query-params/", ["path-params"])


def test_hit_at_k():
    results = [{"url": "https://x/tutorial/path-params/"}]
    assert hit_at_k(results, ["path-params"], is_relevant) == 1.0
    assert hit_at_k(results, ["security"], is_relevant) == 0.0


def test_reciprocal_rank_uses_first_relevant_rank():
    results = [{"url": "https://x/none/"}, {"url": "https://x/body/"}]
    assert reciprocal_rank(results, ["body"], is_relevant) == 0.5
    assert reciprocal_rank(results, ["missing"], is_relevant) == 0.0
