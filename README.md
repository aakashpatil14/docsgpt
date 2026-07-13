# DocsGPT — Production RAG over FastAPI docs + GitHub issues

A retrieval-augmented generation assistant built over a real, messy corpus
(the FastAPI documentation plus its GitHub issues). Phase 2 covers the
ingestion pipeline: fetch → clean → chunk → embed → store.

## Stack (pluggable backends)

Embeddings and storage are swappable via env vars, so the same code runs
locally without keys **or** deploys with the production stack.

| Layer | Deploy (`openai`/`postgres`) | Local fallback (`local`/`chroma`) |
| --- | --- | --- |
| Embeddings | OpenAI `text-embedding-3-small` (1536-d) | `bge-small-en-v1.5` (384-d, on-device) |
| Vector store | PostgreSQL + pgvector | ChromaDB (file-based) |
| Keyword search | Postgres `tsvector` | in-memory `rank-bm25` |

Shared: **cross-encoder reranker**, **LangChain** chunking, **FastAPI** (later),
**Ollama/OpenAI** LLM (later).

Backends are selected in `.env`:

```env
EMBEDDING_BACKEND=openai   # or: local
STORE_BACKEND=postgres     # or: chroma
```

## Setup (deploy: OpenAI + Postgres)

1. Copy the env template and fill in credentials:

   ```bash
   cp .env.example .env
   # set OPENAI_API_KEY and DATABASE_URL; keep EMBEDDING_BACKEND=openai / STORE_BACKEND=postgres
   ```

2. Start Postgres + pgvector (Docker) — or point `DATABASE_URL` at a hosted DB:

   ```bash
   docker compose up -d
   ```

3. Create a virtualenv and install deps:

   ```bash
   python -m venv .venv
   # Windows:  .venv\Scripts\activate
   # macOS/Linux:  source .venv/bin/activate
   pip install -r requirements.txt
   ```

> To run locally with no keys/server, set `EMBEDDING_BACKEND=local` and
> `STORE_BACKEND=chroma` in `.env` (ChromaDB persists to `data/chroma/`).

## Run ingestion (Phase 2)

```bash
python -m ingestion.run_ingest
```

This clones the FastAPI repo, pulls its issues, cleans + dedupes,
chunks (700 tokens / 100 overlap), embeds, and upserts into the store.
Re-running is idempotent (dedupe via `content_hash`).

## Verify

```bash
python -m scripts.verify
```

Prints chunk counts by source and the top-5 matches for a sample query.

## Retrieval + benchmark (Phase 3)

Three retrieval strategies are implemented in `retrieval/`:

- `vector` — dense similarity only
- `hybrid` — dense + keyword (BM25) fused with Reciprocal Rank Fusion
- `hybrid_rerank` — hybrid candidates re-scored by a cross-encoder

Run the benchmark over the golden set to compare them:

```bash
python -m experiments.compare_retrieval
```

This prints a Hit@k / MRR table for each strategy — paste the results into
the benchmark table below once you've run ingestion.

| strategy       | Hit@5 | MRR  |
| -------------- | ----- | ---- |
| vector         | TBD   | TBD  |
| hybrid         | TBD   | TBD  |
| hybrid_rerank  | TBD   | TBD  |

## API + UI (Phase 4)

A FastAPI app serves grounded answers with citations, plus a chat UI.

```bash
uvicorn app:app --reload
# open http://localhost:8000  (UI)
# docs at   http://localhost:8000/docs
```

Endpoints:

- `GET /health` — reports active backends (embedding / store / llm)
- `POST /ask` — `{ "question": "..." }` → `{ answer, citations, used_chunks }`
- `GET /` — chat UI

LLM backend is pluggable via `.env`:

```env
LLM_BACKEND=openai     # or: ollama
LLM_MODEL=gpt-4o-mini  # or e.g. llama3.1 for ollama
```

Answers are constrained to the retrieved context and reply
"I don't know based on the provided documentation." when unsupported.

### Run with Docker

```bash
docker compose up -d          # Postgres
docker build -t docsgpt .
docker run --env-file .env -p 8000:8000 docsgpt
```

## Evaluation (Phase 5)

Two layers of evaluation:

**1. RAGAS quality eval (the RAG Triad + context recall)** — runs the live
pipeline over a golden set and scores it with an LLM judge. Needs the OpenAI
backend + an ingested corpus.

```bash
pip install ragas datasets langchain-openai
python -m experiments.ragas_eval      # prints scores, writes data/ragas_results.csv
```

Metrics: **Faithfulness**, **ResponseRelevancy**, **ContextPrecision**,
**ContextRecall**. Target: faithfulness ≥ 0.85, context recall ≥ 0.80.

**2. Fast deterministic gate (CI)** — pure unit tests for the RRF fusion and
metrics, plus a retrieval regression test that asserts Hit@5 ≥ 0.70 when a
corpus is present.

```bash
pytest -q
```

**CI** (`.github/workflows/ci.yml`): the `unit` job runs the fast gate on every
push/PR; the `rag-eval` job runs the full RAGAS eval on manual dispatch (spins
up a pgvector service, needs the `OPENAI_API_KEY` secret).

## Deploy + observability (Phase 6)

**Caching** — query embeddings and full answers are cached. Set `REDIS_URL`
to use Redis in production; otherwise a small in-process cache is used. Any
cache error degrades to a miss (never breaks a request). `/ask` returns
`cache_hit` and `latency_ms`.

**Tracing** — set `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` to trace every
`/ask` (question, retrieved sources, answer, latency, cache hit). No-op when
unset.

**Run the full stack locally** (API + Postgres + Redis):

```bash
docker compose -f docker-compose.prod.yml up --build
# one-time, in another shell, populate the DB:
docker compose -f docker-compose.prod.yml exec api python -m ingestion.run_ingest
```

**Deploy to Render** — the included `render.yaml` provisions the web service,
a pgvector Postgres, and Redis. Set `OPENAI_API_KEY` in the dashboard, deploy,
then run `python -m ingestion.run_ingest` once from the Render shell.
(Fly.io / Railway work too — point `DATABASE_URL` + `REDIS_URL` at their
managed services.)

## Project layout

```
docsgpt/
├── docker-compose.yml      # Postgres + pgvector (deploy DB)
├── docker-compose.prod.yml # full stack: API + Postgres + Redis
├── render.yaml             # Render blueprint (web + pg + redis)
├── Dockerfile              # container for the API
├── app.py                  # FastAPI app (/ask, /health, UI)
├── static/index.html       # chat UI
├── config.py               # tunables + backend selection
├── requirements.txt
├── infra/
│   ├── cache.py            # Redis / in-process cache
│   └── tracing.py          # optional Langfuse tracing
├── generation/
│   ├── prompt.py           # grounded, cited prompt
│   ├── llm.py              # LLM dispatcher (openai | ollama)
│   └── answer.py           # retrieve → prompt → answer + citations
├── ingestion/
│   ├── db.py               # storage dispatcher (postgres | chroma)
│   ├── stores/
│   │   ├── postgres_store.py  # pgvector + tsvector
│   │   └── chroma_store.py    # Chroma + in-memory BM25
│   ├── fetch_docs.py       # clone repo, read markdown
│   ├── fetch_issues.py     # GitHub issues via API
│   ├── clean.py            # normalize, strip HTML, dedupe
│   ├── chunk.py            # token-based, structure-aware chunking
│   ├── embed.py            # embeddings dispatcher (openai | local)
│   └── run_ingest.py       # orchestrator
├── retrieval/
│   ├── hybrid.py           # vector + keyword search fused with RRF
│   ├── rerank.py           # cross-encoder reranker
│   └── pipeline.py         # retrieve() with strategy switch
├── retrieval/
│   └── fusion.py           # pure RRF (unit-tested)
├── experiments/
│   ├── testset.py          # golden Q / expected-source set
│   ├── metrics.py          # Hit@k, MRR
│   ├── compare_retrieval.py # benchmark the 3 strategies
│   ├── eval_dataset.py     # golden Q + reference answers (RAGAS)
│   └── ragas_eval.py       # RAG Triad evaluation
├── tests/                  # unit + retrieval regression tests
├── .github/workflows/ci.yml
└── scripts/
    └── verify.py           # sanity similarity query
```

## Roadmap

- **Phase 2** — ingestion pipeline ✅
- **Phase 3** — hybrid search + reranking + retrieval benchmark ✅
- **Phase 4** — FastAPI `/ask` endpoint + citations + UI ✅
- **Phase 5** — RAGAS evaluation harness (RAG Triad) wired into CI ✅
- **Phase 6** — deploy + Langfuse tracing + Redis cache ✅
- **Phase 7** — README benchmarks + polish
