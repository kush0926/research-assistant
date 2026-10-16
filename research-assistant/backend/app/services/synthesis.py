"""Verify generated synthesis against the exact structured inputs it received."""
import json
import re

from . import rag


def verify_synthesis(text: str, records: list[dict], paper_tags: bool = False) -> dict:
    body = "\n".join(line for line in text.splitlines() if not re.match(r"^\s*#{1,6}\s", line))
    if paper_tags:
        body = re.sub(r"\[P(\d+)\]", r"[\1]", body)
    markers = [int(m) for m in re.findall(r"\[(\d+)\]", body)]
    sources = [{"text": json.dumps(record, ensure_ascii=False)} for record in records]
    claims = rag.verify_answer(body, sources)
    passed = bool(claims) and bool(markers) and all(1 <= m <= len(records) for m in markers) and all(c["verified"] for c in claims)
    return {"status": "passed" if passed else "fallback", "scope": "exact_structured_inputs",
            "claims": claims, "unverified_claims": sum(not c["verified"] for c in claims)}
