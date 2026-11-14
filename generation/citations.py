"""Assign source IDs once, before prompting, and preserve them in the API."""


def prepare_sources(chunks):
    """Return copied chunks and a source list sharing URL-based citation IDs.

    IDs are stable within one answer. Multiple passages from the same URL
    share an ID. Chunks without a source URL cannot provide a citation.
    """
    sources = {}
    prepared = []
    for chunk in chunks:
        url = (chunk.get("url") or "").strip()
        if not url:
            continue
        if url not in sources:
            sources[url] = {
                "id": len(sources) + 1,
                "title": chunk.get("title") or url,
                "url": url,
            }
        prepared.append({**chunk, "url": url, "citation_id": sources[url]["id"]})
    return prepared, list(sources.values())
