import config
from generation.llm import chat
from generation.citations import prepare_sources
from generation.prompt import PROMPT_VERSION, UNKNOWN_ANSWER, build_prompt
from infra import cache


def retrieve(*args, **kwargs):
    # Keep API validation and isolated generation tests independent of DB/model
    # initialization. Actual retrieval still loads the configured backend.
    from retrieval.pipeline import retrieve as run_retrieval
    return run_retrieval(*args, **kwargs)


def answer_question(question, strategy=None, top_k=None):
    strategy = strategy or config.DEFAULT_STRATEGY
    top_k = top_k or config.TOP_K

    payload = {
        "q": question, "s": strategy, "k": top_k, "m": config.LLM_MODEL,
        "llm_backend": config.LLM_BACKEND,
        "embedding_backend": config.EMBEDDING_BACKEND,
        "embedding_model": config.EMBED_MODEL,
        "store": config.STORE_BACKEND,
        "corpus": config.CORPUS_VERSION,
        "prompt": PROMPT_VERSION,
        "reranker": config.RERANK_MODEL,
        "candidate_k": config.CANDIDATE_K,
        "hybrid_limit": config.HYBRID_LIMIT,
        "rrf_k": config.RRF_K,
        "temperature": config.LLM_TEMPERATURE,
    }
    cached = cache.get("answer", payload)
    if cached is not None:
        cached["cache_hit"] = True
        return cached

    chunks, citations = prepare_sources(
        retrieve(question, strategy=strategy, top_k=top_k)
    )
    if chunks:
        system, user = build_prompt(question, chunks)
        text = chat(system, user)
    else:
        text = UNKNOWN_ANSWER
    result = {
        "answer": text,
        "citations": citations,
        "contexts": [c["content"] for c in chunks],
        "used_chunks": len(chunks),
        "cache_hit": False,
    }
    cache.set("answer", payload, result)
    return result
