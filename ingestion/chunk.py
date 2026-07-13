import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""],
)


def chunk_records(records):
    chunks = []
    for r in records:
        pieces = splitter.split_text(r["text"])
        for i, piece in enumerate(pieces):
            content = (
                f"{r['title']}\n\n{piece}"
                if r["source_type"] == "issue"
                else piece
            )
            h = hashlib.sha256(content.encode("utf-8")).hexdigest()
            chunks.append({
                "source_type": r["source_type"],
                "title": r["title"],
                "url": r["url"],
                "chunk_index": i,
                "content": content,
                "metadata": r["metadata"],
                "content_hash": h,
            })
    print(f"[chunk] created {len(chunks)} chunks")
    return chunks
