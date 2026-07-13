"""Golden eval set for RAGAS (RAG Triad).

Each item has a question and a `reference` (ideal grounded answer). Grow this
to 40-80 items for a robust eval. `expected_url_contains` is reused by the
retrieval benchmark in `experiments/testset.py`.
"""

EVAL_DATASET = [
    {
        "question": "How do I declare a path parameter in FastAPI?",
        "reference": (
            "Declare a path parameter by adding it to the route string in curly "
            "braces and to the function signature, e.g. "
            "@app.get('/items/{item_id}') def read_item(item_id: int). "
            "FastAPI parses and validates the type automatically."
        ),
    },
    {
        "question": "How do I declare query parameters with default values?",
        "reference": (
            "Function parameters that are not part of the path become query "
            "parameters. Give them a default value (e.g. skip: int = 0) to make "
            "them optional; without a default they are required."
        ),
    },
    {
        "question": "How do I define a request body using Pydantic models?",
        "reference": (
            "Create a class inheriting from pydantic.BaseModel with typed fields "
            "and declare it as a function parameter. FastAPI reads the JSON body, "
            "validates it against the model, and gives you an instance."
        ),
    },
    {
        "question": "How do I use dependency injection with Depends?",
        "reference": (
            "Define a callable (function or class) and declare it as a parameter "
            "using Depends(dependency). FastAPI resolves it per request and "
            "injects the result; dependencies can themselves have dependencies."
        ),
    },
    {
        "question": "How do I handle errors and return a specific HTTP status?",
        "reference": (
            "Raise HTTPException(status_code=..., detail=...) inside a path "
            "operation to return an error response with the given status code "
            "and JSON detail."
        ),
    },
    {
        "question": "How do I run work after returning the response?",
        "reference": (
            "Add a BackgroundTasks parameter to the path operation and call "
            "background_tasks.add_task(func, ...). The task runs after the "
            "response is sent."
        ),
    },
    {
        "question": "How do I configure CORS in FastAPI?",
        "reference": (
            "Add CORSMiddleware via app.add_middleware(CORSMiddleware, "
            "allow_origins=[...], allow_methods=[...], allow_headers=[...]) to "
            "allow cross-origin requests from the given origins."
        ),
    },
    {
        "question": "How do I upload a file in a request?",
        "reference": (
            "Declare a parameter of type UploadFile (or bytes) using File(). "
            "UploadFile exposes async read/write and file metadata."
        ),
    },
    {
        "question": "How do I add validation and metadata to query parameters?",
        "reference": (
            "Use Query() as the default, e.g. q: str = Query(default=None, "
            "min_length=3, max_length=50, pattern=...), to add constraints and "
            "OpenAPI metadata."
        ),
    },
    {
        "question": "How do I secure endpoints with OAuth2 password flow?",
        "reference": (
            "Use OAuth2PasswordBearer as a dependency to extract the bearer "
            "token, and OAuth2PasswordRequestForm on the token endpoint. Validate "
            "the token and return the current user from a dependency."
        ),
    },
]
