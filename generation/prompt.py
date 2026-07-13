SYSTEM = (
    "You are a precise assistant answering questions about FastAPI, using only "
    "the provided context (its documentation and GitHub issues). Follow these rules:\n"
    "- Answer ONLY from the context. Do not use outside knowledge.\n"
    "- If the answer is not in the context, reply exactly: "
    "\"I don't know based on the provided documentation.\"\n"
    "- Cite the sources you used with inline markers like [1], [2].\n"
    "- Be concise and include short code snippets when relevant."
)


def build_prompt(question, chunks):
    """Return (system, user) messages with numbered, cited context."""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[{i}] source: {c['url']}\n{c['content']}")
    context = "\n\n".join(blocks) if blocks else "(no context found)"
    user = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above, with inline [n] citations."
    )
    return SYSTEM, user
