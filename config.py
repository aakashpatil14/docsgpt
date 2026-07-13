import os

from dotenv import load_dotenv

load_dotenv()

# --- credentials ---
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/docsgpt"
)

# --- pluggable backends (set in .env) ---
# EMBEDDING_BACKEND: "openai" (deploy) | "local" (no keys, on-device)
# STORE_BACKEND:     "postgres" (deploy) | "chroma" (no server)
EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "openai")
STORE_BACKEND = os.environ.get("STORE_BACKEND", "postgres")

# --- corpus ---
REPO_OWNER = "fastapi"
REPO_NAME = "fastapi"
DOCS_SUBPATH = "docs/en/docs"
DOCS_SITE_BASE = "https://fastapi.tiangolo.com"
ISSUE_STATE = "all"
ISSUE_CUTOFF_DAYS = 730
MAX_ISSUE_PAGES = int(os.environ.get("MAX_ISSUE_PAGES", "30"))
RAW_DIR = "data/raw"

# --- chunking ---
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

# --- embeddings (resolved from backend) ---
if EMBEDDING_BACKEND == "openai":
    EMBED_MODEL = "text-embedding-3-small"
    EMBED_DIM = 1536
    EMBED_BATCH = 100
else:
    EMBED_MODEL = "BAAI/bge-small-en-v1.5"
    EMBED_DIM = 384
    EMBED_BATCH = 64
# bge (local) expects this instruction prefix on the query side; ignored by OpenAI.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# --- storage ---
CHROMA_DIR = "data/chroma"
CHROMA_COLLECTION = "docsgpt"
CHROMA_UPSERT_BATCH = 2000

# --- retrieval ---
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60
CANDIDATE_K = 50
HYBRID_LIMIT = 20
TOP_K = 5

# --- generation (LLM) ---
# LLM_BACKEND: "openai" (deploy) | "ollama" (local)
LLM_BACKEND = os.environ.get("LLM_BACKEND", "openai")
if LLM_BACKEND == "openai":
    LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
else:
    LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.1")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LLM_TEMPERATURE = 0.1
DEFAULT_STRATEGY = "hybrid_rerank"

# --- caching (Phase 6) ---
# REDIS_URL unset -> falls back to a small in-process cache.
CACHE_ENABLED = os.environ.get("CACHE_ENABLED", "true").lower() == "true"
REDIS_URL = os.environ.get("REDIS_URL")
CACHE_TTL = int(os.environ.get("CACHE_TTL", "86400"))  # seconds

# --- observability: Langfuse tracing (Phase 6) ---
# Tracing is a no-op unless both keys are set.
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
