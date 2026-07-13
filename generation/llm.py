import config

_openai_client = None


def _openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _openai_client


def _chat_openai(system, user):
    resp = _openai().chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=config.LLM_TEMPERATURE,
    )
    return resp.choices[0].message.content


def _chat_ollama(system, user):
    import requests
    resp = requests.post(
        f"{config.OLLAMA_HOST}/api/chat",
        json={
            "model": config.LLM_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": config.LLM_TEMPERATURE},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def chat(system, user):
    if config.LLM_BACKEND == "openai":
        return _chat_openai(system, user)
    return _chat_ollama(system, user)
