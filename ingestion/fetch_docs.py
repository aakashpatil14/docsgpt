import os
import subprocess

import config

CLONE_DIR = os.path.join(config.RAW_DIR, "repo")


def _ensure_repo():
    if not os.path.exists(CLONE_DIR):
        os.makedirs(config.RAW_DIR, exist_ok=True)
        url = f"https://github.com/{config.REPO_OWNER}/{config.REPO_NAME}.git"
        subprocess.run(
            ["git", "clone", "--depth", "1", url, CLONE_DIR], check=True
        )


def _to_url(rel_path: str) -> str:
    p = rel_path.replace(config.DOCS_SUBPATH, "").lstrip("/")
    p = p[:-3] if p.endswith(".md") else p
    if p.endswith("index"):
        p = p[: -len("index")]
    return f"{config.DOCS_SITE_BASE}/{p}".rstrip("/") + "/"


def fetch_docs():
    _ensure_repo()
    docs_root = os.path.join(CLONE_DIR, config.DOCS_SUBPATH)
    records = []
    for root, _, files in os.walk(docs_root):
        for f in files:
            if not f.endswith(".md"):
                continue
            full = os.path.join(root, f)
            rel = os.path.relpath(full, CLONE_DIR).replace("\\", "/")
            with open(full, encoding="utf-8") as fh:
                text = fh.read()
            records.append({
                "source_type": "docs",
                "title": f.replace(".md", ""),
                "url": _to_url(rel),
                "text": text,
                "metadata": {"path": rel},
            })
    print(f"[docs] loaded {len(records)} markdown files")
    return records
