import json

import psycopg2
from pgvector.psycopg2 import register_vector

import config

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id            BIGSERIAL PRIMARY KEY,
    source_type   TEXT NOT NULL,
    title         TEXT,
    url           TEXT,
    chunk_index   INT,
    content       TEXT NOT NULL,
    content_tsv   TSVECTOR,
    metadata      JSONB,
    embedding     VECTOR(%(dim)s),
    content_hash  TEXT UNIQUE
);

CREATE INDEX IF NOT EXISTS documents_embedding_idx
    ON documents USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS documents_tsv_idx
    ON documents USING GIN (content_tsv);
"""


def get_conn():
    conn = psycopg2.connect(config.DATABASE_URL)
    register_vector(conn)
    return conn


def init_schema():
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(SCHEMA % {"dim": config.EMBED_DIM})
        conn.commit()


def upsert_chunks(rows):
    inserted = 0
    with get_conn() as conn, conn.cursor() as cur:
        for r in rows:
            cur.execute(
                """
                INSERT INTO documents
                    (source_type, title, url, chunk_index, content,
                     content_tsv, metadata, embedding, content_hash)
                VALUES
                    (%s, %s, %s, %s, %s,
                     to_tsvector('english', %s), %s::jsonb, %s, %s)
                ON CONFLICT (content_hash) DO NOTHING
                """,
                (
                    r["source_type"], r["title"], r["url"], r["chunk_index"],
                    r["content"], r["content"], json.dumps(r["metadata"]),
                    r["embedding"], r["content_hash"],
                ),
            )
            inserted += cur.rowcount
        conn.commit()
    return inserted


def _row_to_dict(row):
    return {
        "id": row[0], "title": row[1], "url": row[2],
        "content": row[3], "source_type": row[4],
    }


def vector_query(query_emb, limit):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, title, url, content, source_type, "
            "1 - (embedding <=> %s) AS similarity "
            "FROM documents ORDER BY embedding <=> %s LIMIT %s",
            (query_emb, query_emb, limit),
        )
        out = []
        for row in cur.fetchall():
            d = _row_to_dict(row)
            d["similarity"] = row[5]
            out.append(d)
    return out


def keyword_query(query, limit):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, title, url, content, source_type
            FROM documents
            WHERE content_tsv @@ plainto_tsquery('english', %s)
            ORDER BY ts_rank_cd(content_tsv, plainto_tsquery('english', %s)) DESC
            LIMIT %s
            """,
            (query, query, limit),
        )
        return [_row_to_dict(row) for row in cur.fetchall()]


def counts_by_source():
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT source_type, COUNT(*) FROM documents GROUP BY source_type")
        return dict(cur.fetchall())
