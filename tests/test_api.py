import pytest
from fastapi.testclient import TestClient

import app as api
import config
from generation import answer
from infra import cache


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api.tracing, "trace_ask", lambda *a, **kw: None)
    with TestClient(api.app) as client:
        yield client


@pytest.mark.parametrize("payload", [
    {}, {"question": ""}, {"question": "   "}, {"question": "x" * 4001},
    {"question": "hello", "strategy": "unsupported"}, {"question": None},
])
def test_invalid_request_never_reaches_generation(client, monkeypatch, payload):
    monkeypatch.setattr(api, "answer_question", lambda *a, **kw: pytest.fail("invalid input"))
    assert client.post("/ask", json=payload).status_code == 422


def test_api_serializes_stable_citation_ids_and_cache(client, monkeypatch):
    monkeypatch.setattr(config, "CACHE_ENABLED", True)
    monkeypatch.setattr(cache, "_backend", cache._MemoryCache())
    received = []

    def retrieve(question, **kwargs):
        received.append(question)
        return [{"url": url, "title": url, "content": "passage"}
                for url in ["https://x/a", "https://x/a", "https://x/b"]]

    monkeypatch.setattr(answer, "retrieve", retrieve)
    monkeypatch.setattr(answer, "chat", lambda *a: "A [1], B [2].")
    response = client.post("/ask", json={"question": "  question  "})
    assert response.status_code == 200
    result = response.json()
    assert received == ["question"]
    assert [c["id"] for c in result["citations"]] == [1, 2]
    assert result["used_chunks"] == 3
    assert result["latency_ms"] >= 0
    assert "contexts" not in result
    repeated = client.post("/ask", json={"question": "question"}).json()
    assert repeated["cache_hit"] is True
    assert repeated["citations"] == result["citations"]
    assert received == ["question"]
