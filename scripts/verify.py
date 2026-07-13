"""Quick sanity check: counts + a similarity query against the corpus."""
from ingestion.db import counts_by_source, vector_query
from ingestion.embed import embed_query


def main():
    print("counts:", counts_by_source())

    question = "how do I add a path parameter?"
    q = embed_query(question)
    print(f"\ntop matches for: {question!r}")
    for r in vector_query(q, 5):
        print(f"  {r['similarity']:.3f}  {r['title']}  {r['url']}")


if __name__ == "__main__":
    main()
