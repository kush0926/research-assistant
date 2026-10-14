"""Hybrid recommendation (Part 17): content-based + citation-graph + gap-aware boosting."""
import numpy as np
from sqlalchemy.orm import Session

from ..db import Citation, Paper, PaperDataset, PaperMethod, ResearchGap, Dataset, Method
from .embeddings import get_embedder


def recommend(db: Session, user_id: int, paper_id: int | None = None, topic: str | None = None,
              gap_id: int | None = None, k: int = 8) -> list[dict]:
    papers = db.query(Paper).filter(Paper.upload_user_id == user_id, Paper.status == "ready").all()
    if not papers:
        return []
    emb = get_embedder()
    vs = emb.encode([f"{p.title}. {p.abstract or ''}" for p in papers])
    idx = {p.id: i for i, p in enumerate(papers)}
    if paper_id in idx:
        q = vs[idx[paper_id]]
    elif topic:
        q = emb.encode([topic], is_query=True)[0]
    else:
        q = vs.mean(axis=0)
    content = vs @ q
    # graph signal: direct citation neighbours (1.0) and co-citation / shared-neighbour strength (<=0.6)
    graph = np.zeros(len(papers))
    if paper_id in idx:
        cites = db.query(Citation).all()
        out_n = {c.cited_paper_id for c in cites if c.citing_paper_id == paper_id}
        in_n = {c.citing_paper_id for c in cites if c.cited_paper_id == paper_id}
        for pid in out_n | in_n:
            if pid in idx:
                graph[idx[pid]] = 1.0
        for c in cites:  # co-citation
            if c.cited_paper_id in out_n and c.citing_paper_id != paper_id and c.citing_paper_id in idx:
                graph[idx[c.citing_paper_id]] = max(graph[idx[c.citing_paper_id]], 0.6)
    boost = np.zeros(len(papers)); reasons = {}
    if gap_id:
        g = db.get(ResearchGap, gap_id)
        if g and g.user_id == user_id:
            mname = db.get(Method, g.method_id).name if g.method_id else None
            dname = db.get(Dataset, g.dataset_id).name if g.dataset_id else None
            m_ids = {r.paper_id for r in db.query(PaperMethod).filter_by(method_id=g.method_id)} if g.method_id else set()
            d_ids = {r.paper_id for r in db.query(PaperDataset).filter_by(dataset_id=g.dataset_id)} if g.dataset_id else set()
            for p in papers:
                b = (0.5 if p.id in m_ids else 0) + (0.5 if p.id in d_ids else 0)
                boost[idx[p.id]] = b
                if b:
                    reasons[p.id] = "gap-aware: uses " + " and ".join(x for x in [mname if p.id in m_ids else None, dname if p.id in d_ids else None] if x)
    w = (0.45, 0.2, 0.35) if gap_id else (0.75, 0.25, 0.0)
    score = w[0] * content + w[1] * graph + w[2] * boost
    out = []
    for i in np.argsort(-score):
        p = papers[i]
        if p.id == paper_id:
            continue
        why = reasons.get(p.id) or ("citation-graph neighbour" if graph[i] >= 0.6 else f"content similarity {content[i]:.2f}")
        out.append({"paper_id": p.id, "title": p.title, "year": p.year, "score": round(float(score[i]), 4), "reason": why})
        if len(out) >= k:
            break
    return out
