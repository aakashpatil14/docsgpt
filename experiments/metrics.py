def hit_at_k(results, expected, is_relevant) -> float:
    """1.0 if any of the top-k results is relevant, else 0.0."""
    return 1.0 if any(is_relevant(r["url"], expected) for r in results) else 0.0


def reciprocal_rank(results, expected, is_relevant) -> float:
    """1/rank of the first relevant result (0 if none)."""
    for i, r in enumerate(results, start=1):
        if is_relevant(r["url"], expected):
            return 1.0 / i
    return 0.0
