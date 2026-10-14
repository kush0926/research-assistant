"""Evaluation harness (blueprint Parts 4-G, 27, 28).

  python eval/run_eval.py --corpus sample_papers --gold sample_papers/gold_gaps.json \
         --ie-gold sample_papers/ie_gold.json --queries sample_papers/retrieval_queries.json

Reports: gap-detection P/R/F1@K + MRR for {sparsity-only, evidence-only, hybrid} ablations; retrieval
Recall@K / MRR / NDCG@K for {TF-IDF keyword, embedding-only, embedding+rerank}; IE precision/recall/F1.
Numbers are only as meaningful as the corpus and gold labels you supply — on the bundled SYNTHETIC corpus this is a
smoke test of the harness, NOT a result to report. Build a real 15-25 paper benchmark with manually derived gaps.
"""
import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return round(p, 3), round(r, 3), round(2 * p * r / (p + r), 3) if p + r else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True); ap.add_argument("--gold"); ap.add_argument("--ie-gold")
    ap.add_argument("--queries"); ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    tmp = tempfile.mkdtemp()
    os.environ.setdefault("DATA_DIR", tmp); os.environ.setdefault("DATABASE_URL", f"sqlite:///{tmp}/eval.db")
    os.environ.setdefault("SYNC_INGEST", "true"); os.environ.setdefault("ENABLE_METADATA_LOOKUP", "false")
    os.environ.setdefault("RATE_LIMIT_PER_MIN", "100000")
    from fastapi.testclient import TestClient
    from app.main import app
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from app.db import Chunk, SessionLocal
    from app.services import rag
    from app.services.embeddings import get_embedder
    from app.services.vectorstore import get_store

    with TestClient(app) as c:
        tok = c.post("/auth/register", json={"name": "eval", "email": "eval@x.io", "password": "evalpass123"}).json()["token"]
        c.headers["Authorization"] = f"Bearer {tok}"
        for pdf in sorted(Path(a.corpus).glob("*.pdf")):
            r = c.post("/papers/upload", files={"file": (pdf.name, pdf.read_bytes(), "application/pdf")})
            assert r.status_code == 200, r.text
        papers = c.get("/papers").json()["papers"]
        bad = [p for p in papers if p["status"] != "ready"]
        print(f"Ingested {len(papers) - len(bad)}/{len(papers)} papers" + (f"  FAILED: {[(p['id'], p['error']) for p in bad]}" if bad else ""))
        report = {"corpus_size": len(papers)}

        # ---------- research-gap detection ----------
        if a.gold:
            gold = [(g["method"].lower(), g["dataset"].lower()) for g in json.load(open(a.gold))]
            configs = {"hybrid (all signals)": None,
                       "sparsity-only": {"sparsity": 1.0},
                       "evidence-only": {"evidence": 1.0}}
            report["gap_detection"] = {}
            for name, w in configs.items():
                res = c.post("/research-gaps", json={"max_gaps": 30, "weights": w}).json()
                ranked = [((g["method"] or "").lower(), (g["dataset"] or "").lower()) for g in res["gaps"] if g["type"] == "combination"]
                top = ranked[: a.k]
                tp = len(set(top) & set(gold)); fp = len(top) - tp; fn = len(set(gold) - set(top))
                ranks = [ranked.index(g) + 1 for g in gold if g in ranked]
                p, r, f = prf(tp, fp, fn)
                report["gap_detection"][name] = {f"P@{a.k}": p, f"R@{a.k}": r, f"F1@{a.k}": f,
                                                 "gold_ranks": ranks, "MRR": round(sum(1 / x for x in ranks) / len(gold), 3)}

        # ---------- information extraction ----------
        if a.ie_gold:
            gold = json.load(open(a.ie_gold)); tp = fp = fn = 0
            by_title = {p["title"]: p["id"] for p in papers}
            for g in gold:
                pid = by_title.get(g["title"]) or next((i for t, i in by_title.items() if g["title"][:40] in (t or "")), None)
                if not pid:
                    fn += len(g["methods"]) + len(g["datasets"]); continue
                f = c.get(f"/papers/{pid}").json()["fields"]
                for kind in ("methods", "datasets"):
                    pred = {x["name"] for x in f.get(kind, {}).get("value", [])
                            if not (kind == "methods" and x.get("category") in ("feature", "technique"))}
                    ref = set(g[kind]); tp += len(pred & ref); fp += len(pred - ref); fn += len(ref - pred)
            p, r, f1 = prf(tp, fp, fn)
            report["information_extraction"] = {"precision": p, "recall": r, "F1": f1}

        # ---------- retrieval ----------
        if a.queries:
            qs = json.load(open(a.queries)); db = SessionLocal()
            uid = c.get("/auth/me").json()["id"]
            ids = rag.owned_ready_ids(db, uid)
            chunks = db.query(Chunk).filter(Chunk.paper_id.in_(ids)).all()
            title_of = {p["id"]: p["title"] for p in papers}
            tf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit([x.text for x in chunks])
            M = tf.transform([x.text for x in chunks]); emb = get_embedder()
            K = a.k
            def paper_rank(paper_ids):
                seen = []
                for pid in paper_ids:
                    if pid not in seen: seen.append(pid)
                return seen
            def metrics(rank_fn):
                rec = mrr = ndcg = 0.0
                for q in qs:
                    rel = {pid for pid, t in title_of.items() if any(r in (t or "") for r in q["relevant_titles"])}
                    got = paper_rank(rank_fn(q["q"]))[:K]
                    hits = [1 if g in rel else 0 for g in got]
                    rec += sum(hits) / (len(rel) or 1)
                    mrr += next((1 / (i + 1) for i, h in enumerate(hits) if h), 0)
                    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits)); idcg = sum(1 / math.log2(i + 2) for i in range(min(len(rel), K)))
                    ndcg += dcg / idcg if idcg else 0
                n = len(qs); return {f"Recall@{K}": round(rec / n, 3), "MRR": round(mrr / n, 3), f"NDCG@{K}": round(ndcg / n, 3)}
            def tfidf(q):
                s = (M @ tf.transform([q]).T).toarray().ravel(); return [chunks[i].paper_id for i in np.argsort(-s)[:30]]
            def embonly(q):
                return [h[2]["paper_id"] for h in get_store().query(emb.encode([q], is_query=True)[0], 30, paper_ids=ids)]
            def full(q):
                return [x["paper_id"] for x in rag.retrieve(db, q, ids, k=30, n=30)]
            report["retrieval"] = {"Baseline 1: TF-IDF keyword": metrics(tfidf), "Baseline 2: embedding only": metrics(embonly),
                                   "Proposed: embedding + rerank": metrics(full)}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
