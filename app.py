import os
import time

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from generation.answer import answer_question
from infra import tracing

app = FastAPI(title="DocsGPT", description="RAG over FastAPI docs + GitHub issues")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


class AskRequest(BaseModel):
    question: str
    strategy: str = config.DEFAULT_STRATEGY


class Citation(BaseModel):
    title: str
    url: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    used_chunks: int
    cache_hit: bool = False
    latency_ms: float = 0.0


@app.get("/health")
def health():
    return {
        "status": "ok",
        "embedding_backend": config.EMBEDDING_BACKEND,
        "store_backend": config.STORE_BACKEND,
        "llm_backend": config.LLM_BACKEND,
        "llm_model": config.LLM_MODEL,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    start = time.perf_counter()
    result = answer_question(req.question, strategy=req.strategy)
    latency_ms = (time.perf_counter() - start) * 1000
    result["latency_ms"] = round(latency_ms, 1)
    tracing.trace_ask(
        req.question, req.strategy, result, latency_ms,
        cache_hit=result.get("cache_hit", False),
    )
    return result


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
