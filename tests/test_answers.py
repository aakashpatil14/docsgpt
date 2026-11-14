import pytest

import config
from generation import answer
from generation.citations import prepare_sources
from generation.prompt import UNKNOWN_ANSWER, build_prompt
from infra import cache


def chunk(url, content="passage", title="Source"):
    return {"url": url, "content": content, "title": title}


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch):
    monkeypatch.setattr(cache, "_backend", cache._MemoryCache())
    monkeypatch.setattr(config, "CACHE_ENABLED", True)


def test_duplicate_urls_share_ids_in_prompt_and_response(monkeypatch):
    raw = [chunk("https://example.com/a", "A1"),
           chunk("https://example.com/a", "A2"),
           chunk("https://example.com/b", "B1")]
    monkeypatch.setattr(answer, "retrieve", lambda *a, **kw: raw)
    prompts = []

    def fake_chat(system, user):
        prompts.append(user)
        return "Use A [1] and B [2]."

    monkeypatch.setattr(answer, "chat", fake_chat)
    result = answer.answer_question("question")
    assert prompts[0].count("[1] source: https://example.com/a") == 2
    assert "[2] source: https://example.com/b" in prompts[0]
    assert "[3] source:" not in prompts[0]
    assert [(c["id"], c["url"]) for c in result["citations"]] == [
        (1, "https://example.com/a"), (2, "https://example.com/b")]
    assert result["used_chunks"] == 3
    assert result["contexts"] == ["A1", "A2", "B1"]
    assert all("citation_id" not in c for c in raw)


def test_distinct_urls_keep_distinct_ids_even_with_same_title():
    prepared, sources = prepare_sources([chunk("https://x/a"), chunk("https://x/b")])
    assert [c["citation_id"] for c in prepared] == [1, 2]
    assert [s["id"] for s in sources] == [1, 2]
    assert "[2] source: https://x/b" in build_prompt("q", prepared)[1]


@pytest.mark.parametrize("raw", [[], [chunk("")], [chunk(None)], [chunk("   ")]])
def test_no_citable_context_abstains_without_llm(monkeypatch, raw):
    monkeypatch.setattr(answer, "retrieve", lambda *a, **kw: raw)
    monkeypatch.setattr(answer, "chat", lambda *a: pytest.fail("LLM must not run"))
    result = answer.answer_question("unknown question")
    assert result["answer"] == UNKNOWN_ANSWER
    assert result["citations"] == []
    assert result["contexts"] == []
    assert result["used_chunks"] == 0


def test_missing_url_does_not_shift_valid_source_numbers():
    prepared, sources = prepare_sources([chunk(""), chunk(" https://x/a ")])
    assert len(prepared) == 1
    assert sources[0]["id"] == prepared[0]["citation_id"] == 1
    assert sources[0]["url"] == "https://x/a"


def test_cache_hit_preserves_citations_and_skips_generation(monkeypatch):
    monkeypatch.setattr(answer, "retrieve", lambda *a, **kw: [chunk("https://x/a")])
    monkeypatch.setattr(answer, "chat", lambda *a: "Answer [1].")
    first = answer.answer_question("q")
    monkeypatch.setattr(answer, "retrieve", lambda *a, **kw: pytest.fail("cache miss"))
    second = answer.answer_question("q")
    assert second["cache_hit"] is True
    assert first["cache_hit"] is False
    assert second["citations"] == first["citations"]


@pytest.mark.parametrize("setting,new_value", [
    ("CORPUS_VERSION", "updated"), ("STORE_BACKEND", "another-store"),
    ("EMBED_MODEL", "another-embedding"), ("LLM_BACKEND", "another-provider"),
    ("RERANK_MODEL", "another-reranker"), ("HYBRID_LIMIT", 37),
])
def test_configuration_changes_invalidate_answers(monkeypatch, setting, new_value):
    calls = []
    monkeypatch.setattr(answer, "retrieve", lambda *a, **kw: [chunk("https://x/a")])
    monkeypatch.setattr(answer, "chat", lambda *a: calls.append(1) or "Answer [1].")
    answer.answer_question("q")
    monkeypatch.setattr(config, setting, new_value)
    assert answer.answer_question("q")["cache_hit"] is False
    assert len(calls) == 2


def test_prompt_changes_invalidate_answers(monkeypatch):
    monkeypatch.setattr(answer, "retrieve", lambda *a, **kw: [chunk("https://x/a")])
    monkeypatch.setattr(answer, "chat", lambda *a: "Answer [1].")
    answer.answer_question("q")
    monkeypatch.setattr(answer, "PROMPT_VERSION", "next")
    assert answer.answer_question("q")["cache_hit"] is False
