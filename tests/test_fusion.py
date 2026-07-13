from retrieval.fusion import reciprocal_rank_fusion


def test_rrf_rewards_items_ranked_high_in_both_lists():
    vec = ["x", "y", "z"]
    kw = ["y", "x", "w"]
    scores = reciprocal_rank_fusion([vec, kw], k=60)
    # y is #2 and #1; x is #1 and #2 -> both beat items in only one list
    assert scores["y"] > scores["z"]
    assert scores["x"] > scores["w"]


def test_rrf_single_list_preserves_order():
    scores = reciprocal_rank_fusion([["a", "b", "c"]], k=60)
    assert scores["a"] > scores["b"] > scores["c"]


def test_rrf_empty():
    assert reciprocal_rank_fusion([]) == {}
