"""Benchmark vector vs hybrid vs hybrid+rerank on the golden set.

Produces a table of Hit@k and MRR you can paste into the README.

    python -m experiments.compare_retrieval
"""
import config
from experiments.metrics import hit_at_k, reciprocal_rank
from experiments.testset import TESTSET, is_relevant
from retrieval.pipeline import STRATEGIES, retrieve


def evaluate(strategy, top_k):
    hits, rrs = [], []
    for item in TESTSET:
        results = retrieve(item["question"], strategy=strategy, top_k=top_k)
        expected = item["expected_url_contains"]
        hits.append(hit_at_k(results, expected, is_relevant))
        rrs.append(reciprocal_rank(results, expected, is_relevant))
    return sum(hits) / len(hits), sum(rrs) / len(rrs)


def main():
    top_k = config.TOP_K
    print(f"Evaluating {len(TESTSET)} questions (top_k={top_k})\n")
    print(f"{'strategy':<16}{'Hit@k':>10}{'MRR':>10}")
    print("-" * 36)
    for strategy in STRATEGIES:
        hit, mrr = evaluate(strategy, top_k)
        print(f"{strategy:<16}{hit:>10.3f}{mrr:>10.3f}")


if __name__ == "__main__":
    main()
