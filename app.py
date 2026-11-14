import os
import time

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Literal

from pydantic import BaseModel, Field, field_validator

import config
from generation.answer import answer_question
from infra import tracing

app = FastAPI(title="DocsGPT", description="RAG over FastAPI docs + GitHub issues")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    strategy: Literal["vector", "hybrid", "hybrid_rerank"] = config.DEFAULT_STRATEGY

    @field_validator("question", mode="before")
    @classmethod
    def strip_question(cls, value):
        return value.strip() if isinstance(value, str) else value


class Citation(BaseModel):
    id: int = Field(ge=1)
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
