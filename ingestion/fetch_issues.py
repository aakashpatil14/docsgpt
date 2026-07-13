import time
from datetime import datetime, timedelta, timezone

import requests

import config

API = "https://api.github.com"


def _headers():
    h = {"Accept": "application/vnd.github+json"}
    if config.GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {config.GITHUB_TOKEN}"
    return h


def fetch_issues():
    cutoff = datetime.now(timezone.utc) - timedelta(days=config.ISSUE_CUTOFF_DAYS)
    records = []
    for page in range(1, config.MAX_ISSUE_PAGES + 1):
        url = f"{API}/repos/{config.REPO_OWNER}/{config.REPO_NAME}/issues"
        params = {"state": config.ISSUE_STATE, "per_page": 100, "page": page}
        resp = requests.get(url, headers=_headers(), params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        for it in batch:
            if "pull_request" in it:
                continue
            created = datetime.fromisoformat(
                it["created_at"].replace("Z", "+00:00")
            )
            if created < cutoff:
                continue
            body = it.get("body") or ""
            records.append({
                "source_type": "issue",
                "title": it["title"],
                "url": it["html_url"],
                "text": f"{it['title']}\n\n{body}",
                "metadata": {
                    "issue_number": it["number"],
                    "issue_state": it["state"],
                    "created_at": it["created_at"],
                    "labels": [l["name"] for l in it.get("labels", [])],
                },
            })
        time.sleep(0.3)
    print(f"[issues] loaded {len(records)} issues")
    return records
