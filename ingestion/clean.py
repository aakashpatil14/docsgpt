import hashlib
import re

from bs4 import BeautifulSoup


def _strip_html(text: str) -> str:
    if "<" in text and ">" in text:
        return BeautifulSoup(text, "html.parser").get_text()
    return text


def _normalize(text: str) -> str:
    text = _strip_html(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def clean_records(records):
    seen_hashes = set()
    cleaned = []
    dropped = 0
    for r in records:
        text = _normalize(r["text"])
        if len(text) < 40:
            dropped += 1
            continue
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if h in seen_hashes:
            dropped += 1
            continue
        seen_hashes.add(h)
        r["text"] = text
        cleaned.append(r)
    print(f"[clean] kept {len(cleaned)}, dropped {dropped}")
    return cleaned
