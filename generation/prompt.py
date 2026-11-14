PROMPT_VERSION = "source-ids-v1"
UNKNOWN_ANSWER = "I don't know based on the provided documentation."

SYSTEM = (
    "You are a precise assistant answering questions about FastAPI, using only "
    "the provided context (its documentation and GitHub issues). Follow these rules:\n"
    "- Answer ONLY from the context. Do not use outside knowledge.\n"
    "- If the answer is not in the context, reply exactly: "
    "\"I don't know based on the provided documentation.\"\n"
    "- Cite the sources you used with inline markers like [1], [2].\n"
    "- Use only the supplied source IDs; passages may share a source ID.\n"
    "- Be concise and include short code snippets when relevant."
)


def build_prompt(question, chunks):
    """Return (system, user) messages with numbered, cited context."""
    blocks = []
    for c in chunks:
        blocks.append(f"[{c['citation_id']}] source: {c['url']}\n{c['content']}")
    context = "\n\n".join(blocks) if blocks else "(no context found)"
    user = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above, with inline [n] citations."
    )
    return SYSTEM, user
