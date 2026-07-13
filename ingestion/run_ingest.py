from ingestion import chunk, clean, db, embed, fetch_docs, fetch_issues


def main():
    db.init_schema()

    raw = fetch_docs.fetch_docs() + fetch_issues.fetch_issues()
    cleaned = clean.clean_records(raw)
    chunks = chunk.chunk_records(cleaned)

    vectors = embed.embed_texts([c["content"] for c in chunks])
    for c, v in zip(chunks, vectors):
        c["embedding"] = v

    inserted = db.upsert_chunks(chunks)
    print(
        f"[done] inserted {inserted} new rows "
        f"(of {len(chunks)} chunks processed)"
    )


if __name__ == "__main__":
    main()
