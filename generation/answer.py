import config
from generation.llm import chat
from generation.prompt import build_prompt
from infra import cache
from retrieval.pipeline import retrieve


def _dedupe_citations(chunks):
    seen = set()
    citations = []
    for c in chunks:
        url = c.get("url", "")
        if url and url not in seen:
            seen.add(url)
            citations.append({"title": c.get("title", ""), "url": url})
    return citations


def answer_question(question, strategy=None, top_k=None):
    strategy = strategy or config.DEFAULT_STRATEGY
    top_k = top_k or config.TOP_K

    payload = {"q": question, "s": strategy, "k": top_k, "m": config.LLM_MODEL}
    cached = cache.get("answer", payload)
    if cached is not None:
        cached["cache_hit"] = True
        return cached

    chunks = retrieve(question, strategy=strategy, top_k=top_k)
    system, user = build_prompt(question, chunks)
    text = chat(system, user)
    result = {
        "answer": text,
        "citations": _dedupe_citations(chunks),
        "contexts": [c["content"] for c in chunks],
        "used_chunks": len(chunks),
        "cache_hit": False,
    }
    cache.set("answer", payload, result)
    return result
