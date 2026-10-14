"""Light coordinator (Part 18): a rule-based router over fixed sub-pipelines — deliberately NOT an open-ended planner.
Bounded: one route, <= 3 steps, no loops; document text can never trigger tools (only the user's request routes)."""
import re

from sqlalchemy.orm import Session

from ..db import Paper
from . import comparison, gaps, litreview, paper_search, rag, recommend

ROUTES = [
    ("gap", r"\b(gap|under-?explored|not (been )?(tried|explored)|what('s| is| hasn't)|research direction|open problem)"),
    ("compare", r"\b(compare|comparison|versus|vs\.?|difference between)\b"),
    ("review", r"\b(literature review|review draft|survey of)\b"),
    ("recommend", r"\b(recommend|similar papers|read next|related papers)\b"),
    ("search", r"\b(search|find (me )?papers|look up|latest papers)\b"),
]


def route_request(text: str) -> str:
    for name, rx in ROUTES:
        if re.search(rx, text, re.I):
            return name
    return "query"


def run(db: Session, user_id: int, request: str, paper_ids: list[int] | None = None) -> dict:
    route = route_request(request)
    steps = [f"router → {route}"]
    ids = rag.owned_ready_ids(db, user_id, paper_ids)
    if route == "search":
        res = paper_search.search(request, "semanticscholar", 8); steps.append("search agent: external APIs")
    elif route == "compare":
        top = rag.retrieve(db, request, ids)
        pick = paper_ids or list(dict.fromkeys(c["paper_id"] for c in top))[:4]
        papers = db.query(Paper).filter(Paper.id.in_(pick), Paper.upload_user_id == user_id).all()
        if len(papers) < 2:
            return {"route": route, "steps": steps, "result": {"error": "Select at least 2 papers to compare."}}
        res = comparison.compare(db, papers[:5]); steps.append(f"comparison agent: {len(papers[:5])} papers")
    elif route == "gap":
        res = gaps.detect_gaps(db, user_id, paper_ids, topic=request); steps.append("gap agent: sparsity + evidence + scoring")
    elif route == "review":
        papers = db.query(Paper).filter(Paper.id.in_(ids)).all()
        res = litreview.generate(db, papers, user_id); steps.append("report agent")
    elif route == "recommend":
        res = recommend.recommend(db, user_id, topic=request); steps.append("recommendation")
    else:
        res = rag.answer_question(db, user_id, request, paper_ids); steps.append("evidence agent → RAG")
        steps.append(f"verification agent: {res['unverified']} unverified claim(s)")
    return {"route": route, "steps": steps, "result": res}
