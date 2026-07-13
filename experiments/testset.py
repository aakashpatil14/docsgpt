"""A small golden set for measuring retrieval quality.

Each item: a question + substrings that should appear in the URL of a
correctly-retrieved chunk. Grow this to 40-80 questions for Phase 5.
"""

TESTSET = [
    {
        "question": "How do I declare a path parameter in FastAPI?",
        "expected_url_contains": ["path-params"],
    },
    {
        "question": "How do I declare query parameters with default values?",
        "expected_url_contains": ["query-params"],
    },
    {
        "question": "How do I define a request body using Pydantic models?",
        "expected_url_contains": ["body"],
    },
    {
        "question": "How can I add validation and metadata to query parameters?",
        "expected_url_contains": ["query-params-str-validations"],
    },
    {
        "question": "How do I use dependency injection with Depends?",
        "expected_url_contains": ["dependencies"],
    },
    {
        "question": "How do I secure endpoints with OAuth2 and JWT tokens?",
        "expected_url_contains": ["security", "oauth2"],
    },
    {
        "question": "How do I run background tasks after returning a response?",
        "expected_url_contains": ["background-tasks"],
    },
    {
        "question": "How do I handle errors and raise HTTPException?",
        "expected_url_contains": ["handling-errors"],
    },
    {
        "question": "How do I upload files in a request?",
        "expected_url_contains": ["request-files"],
    },
    {
        "question": "How do I use middleware to process requests?",
        "expected_url_contains": ["middleware"],
    },
    {
        "question": "How do I configure CORS in FastAPI?",
        "expected_url_contains": ["cors"],
    },
    {
        "question": "How do I use async and await in path operations?",
        "expected_url_contains": ["async"],
    },
]


def is_relevant(url: str, expected) -> bool:
    if not url:
        return False
    low = url.lower()
    return any(sub.lower() in low for sub in expected)
