"""Storage dispatcher: selects the backend from config.STORE_BACKEND."""
import config

if config.STORE_BACKEND == "chroma":
    from ingestion.stores import chroma_store as _store
else:
    from ingestion.stores import postgres_store as _store

init_schema = _store.init_schema
upsert_chunks = _store.upsert_chunks
vector_query = _store.vector_query
keyword_query = _store.keyword_query
counts_by_source = _store.counts_by_source
