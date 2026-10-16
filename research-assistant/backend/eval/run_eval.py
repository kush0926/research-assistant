"""Evaluation harness (blueprint Parts 4-G, 27, 28).

  python eval/run_eval.py --corpus sample_papers --gold sample_papers/gold_gaps.json \
         --ie-gold sample_papers/ie_gold.json --queries sample_papers/retrieval_queries.json

Reports: gap-detection P/R/F1@K + MRR for {sparsity-only, evidence-only, hybrid} ablations; retrieval
Recall@K / MRR / NDCG@K for {TF-IDF keyword, embedding-only, embedding+rerank}; IE precision/recall/F1.
Numbers are only as meaningful as the corpus and gold labels you supply — on the bundled SYNTHETIC corpus this is a
smoke test of the harness, NOT a result to report. The ParkVoice files under eval/ are pilot inputs, not a manually
validated gold benchmark. Gap gold records use exact (problem, dataset, method) labels.
"""
import argparse
import json
import math
import os
import re
import sys
import tempfile
import time
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
    ap.add_argument("--ie-review-draft", type=Path,
                    help="write extraction predictions and source passages to a blank double-review worksheet")
    ap.add_argument("--queries"); ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--smoke-workflow", action="store_true",
                     help="exercise analysis, cited Q&A, comparison, gaps, graph, recommendations, review and agent on the real corpus")
    ap.add_argument("--gap-review-draft", type=Path,
                    help="write system-generated gap candidates as a reviewer worksheet; this is not gold data")
    ap.add_argument("--rag-ablation-draft", type=Path,
                    help="compare Baseline 3 Basic RAG with the proposed RAG and write a human-review worksheet; requires --queries and a live LLM key")
    a = ap.parse_args()
    if a.gap_review_draft and not a.smoke_workflow:
        ap.error("--gap-review-draft requires --smoke-workflow")
    if a.rag_ablation_draft and not a.queries:
        ap.error("--rag-ablation-draft requires --queries")
    tmp = tempfile.mkdtemp()
    os.environ.setdefault("DATA_DIR", tmp); os.environ.setdefault("DATABASE_URL", f"sqlite:///{tmp}/eval.db")
    os.environ.setdefault("SYNC_INGEST", "true"); os.environ.setdefault("ENABLE_METADATA_LOOKUP", "false")
    os.environ.setdefault("RATE_LIMIT_PER_MIN", "100000")
    from fastapi.testclient import TestClient
    from app.main import app
    from app.config import settings
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from app.db import Chunk, Paper, SessionLocal
    from app.services import rag
    from app.services.embeddings import get_embedder
    from app.services.vectorstore import get_store
    from app.services.llm import llm
    from eval.rag_ablation import answer_basic_rag, review_record
    from eval.ie_review import build_ie_review_draft, load_ie_gold_document

    if a.rag_ablation_draft and not llm.available:
        ap.error("--rag-ablation-draft requires a configured live LLM provider; offline fallback output is not a valid Basic RAG comparison")

    with TestClient(app) as c:
        tok = c.post("/auth/register", json={"name": "eval", "email": "eval@x.io", "password": "evalpass123"}).json()["token"]
        c.headers["Authorization"] = f"Bearer {tok}"
        uploaded_files = {}
        for pdf in sorted(Path(a.corpus).glob("*.pdf")):
            r = c.post("/papers/upload", files={"file": (pdf.name, pdf.read_bytes(), "application/pdf")})
            assert r.status_code == 200, r.text
            uploaded_files[r.json()["paper_id"]] = pdf.name
        papers = c.get("/papers").json()["papers"]
        for paper in papers:
            paper["filename"] = uploaded_files.get(paper["id"])
        bad = [p for p in papers if p["status"] != "ready"]
        print(f"Ingested {len(papers) - len(bad)}/{len(papers)} papers" + (f"  FAILED: {[(p['id'], p['error']) for p in bad]}" if bad else ""),
              file=sys.stderr)
        report = {"corpus_size": len(papers)}

        if a.ie_review_draft:
            def get_chunk_for_review(chunk_id):
                response = c.get(f"/chunks/{chunk_id}")
                return response.json() if response.status_code == 200 else None

            worksheet = build_ie_review_draft(
                papers,
                lambda paper_id: c.get(f"/papers/{paper_id}").json(),
                get_chunk_for_review,
            )
            worksheet["corpus_size"] = len(papers)
            a.ie_review_draft.parent.mkdir(parents=True, exist_ok=True)
            a.ie_review_draft.write_text(json.dumps(worksheet, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
            report["ie_review_draft"] = str(a.ie_review_draft)

        if a.smoke_workflow:
            ready_ids = [p["id"] for p in papers if p["status"] == "ready"]
            if len(ready_ids) < 3:
                raise ValueError("The real-corpus workflow check requires at least 3 successfully ingested papers")
            selected = ready_ids[:3]
            workflow = {}
            r = c.post(f"/papers/{selected[0]}/analyze")
            assert r.status_code == 200, f"Analyze failed: {r.status_code} {r.text}"
            workflow["analysis_fields"] = len(r.json().get("extracted_fields", {}))
            r = c.post("/query", json={"question": "What speech representation or model was used for Parkinson disease classification?",
                                       "paper_ids": selected})
            assert r.status_code == 200, f"Cited Q&A failed: {r.status_code} {r.text}"
            rag_result = r.json()
            assert rag_result.get("citations"), f"Expected citations for a supported real-corpus question: {rag_result}"
            for citation in rag_result["citations"]:
                resolved = c.get(f"/chunks/{citation['chunk_id']}")
                assert resolved.status_code == 200, f"Citation failed to resolve: {citation}"
            workflow["rag_citations"] = len(rag_result["citations"])
            workflow["rag_unverified_claims"] = rag_result.get("unverified", 0)
            r = c.post("/compare", json={"paper_ids": selected})
            assert r.status_code == 200, f"Comparison failed: {r.status_code} {r.text}"
            workflow["comparison_rows"] = len(r.json().get("table", {}).get("rows", []))
            assert workflow["comparison_rows"] == len(selected), "Comparison did not return one row per selected paper"
            workflow["comparison_evidence_cells"] = 0
            for row in r.json()["table"]["rows"]:
                for dimension in ("dataset", "algorithm", "metric", "results", "limitations", "future_work"):
                    cell = row[dimension]
                    if cell["text"] != "—":
                        assert cell.get("chunk_id"), f"Comparison cell has no source chunk: {dimension} {cell}"
                        resolved = c.get(f"/chunks/{cell['chunk_id']}")
                        assert resolved.status_code == 200, f"Comparison evidence failed to resolve: {cell}"
                        workflow["comparison_evidence_cells"] += 1
            r = c.post("/research-gaps", json={"paper_ids": ready_ids, "max_gaps": 8})
            assert r.status_code == 200, f"Gap detection failed: {r.status_code} {r.text}"
            gap_result = r.json()
            workflow["candidate_gaps"] = len(gap_result.get("gaps", []))
            workflow["gap_evidence_items"] = sum(len(g.get("evidence", [])) for g in gap_result.get("gaps", []))
            for gap in gap_result.get("gaps", []):
                if gap.get("tier") == "evidence_backed":
                    assert any(e.get("type") in ("limitation", "future_work", "contradiction")
                               for e in gap.get("evidence", [])), f"Evidence-backed gap lacks direct textual/contradiction support: {gap}"
                if gap.get("components", {}).get("contradiction"):
                    assert any(e.get("type") == "contradiction" and e.get("chunk_id")
                               for e in gap.get("evidence", [])), f"Contradiction score has no linked conflict: {gap}"
                for evidence in gap.get("evidence", []):
                    if evidence.get("chunk_id"):
                        resolved = c.get(f"/chunks/{evidence['chunk_id']}")
                        assert resolved.status_code == 200, f"Gap evidence failed to resolve: {evidence}"
            if a.gap_review_draft:
                a.gap_review_draft.parent.mkdir(parents=True, exist_ok=True)
                worksheet = {
                    "status": "draft_candidates_awaiting_team_and_guide_review_not_gold_labels",
                    "corpus_directory": str(Path(a.corpus)),
                    "corpus_size": gap_result.get("corpus_size"),
                    "review_fields": ["valid_research_gap", "independently_cross_checked", "reviewer", "review_notes"],
                    "candidates": [
                        {**g, "valid_research_gap": None, "independently_cross_checked": None,
                         "reviewer": None, "review_notes": ""}
                        for g in gap_result.get("gaps", [])
                    ],
                }
                a.gap_review_draft.write_text(json.dumps(worksheet, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
                workflow["gap_review_draft"] = str(a.gap_review_draft)
            r = c.get("/knowledge-graph")
            assert r.status_code == 200, f"Knowledge graph failed: {r.status_code} {r.text}"
            workflow["graph_nodes"] = len(r.json().get("nodes", []))
            workflow["graph_edges"] = len(r.json().get("edges", []))
            r = c.get(f"/recommendations?paper_id={selected[0]}")
            assert r.status_code == 200, f"Recommendations failed: {r.status_code} {r.text}"
            workflow["recommendations"] = len(r.json().get("recommendations", []))
            r = c.post("/literature-review", json={"paper_ids": selected})
            assert r.status_code == 200, f"Literature review failed: {r.status_code} {r.text}"
            review = r.json().get("markdown", "")
            assert "## References" in review and "[P1]" in review, "Review draft is missing its reference section/citation tags"
            workflow["review_chars"] = len(review)
            r = c.post("/agent", json={"request": "compare these Parkinson speech papers", "paper_ids": selected})
            assert r.status_code == 200, f"Agent coordinator failed: {r.status_code} {r.text}"
            workflow["agent_route"] = r.json().get("route")
            report["real_corpus_workflow"] = workflow

        # ---------- research-gap detection ----------
        if a.gold:
            norm = lambda value: " ".join((value or "").casefold().split())
            gold = {(norm(g["problem"]), norm(g["dataset"]), norm(g["method"])) for g in json.load(open(a.gold, encoding="utf-8"))}
            configs = {"hybrid (all signals)": None,
                       "sparsity-only": {"sparsity": 1.0},
                       "evidence-only": {"evidence": 1.0}}
            report["gap_detection"] = {}
            for name, w in configs.items():
                res = c.post("/research-gaps", json={"max_gaps": 30, "weights": w}).json()
                ranked = [(norm(g["problem"]), norm(g["dataset"]), norm(g["method"]))
                          for g in res["gaps"] if g["type"] == "combination"]
                top = ranked[: a.k]
                tp = len(set(top) & gold); fp = len(set(top) - gold); fn = len(gold - set(top))
                ranks = [ranked.index(g) + 1 for g in gold if g in ranked]
                precision_at_k = tp / a.k
                recall_at_k = tp / len(gold) if gold else 0.0
                f1_at_k = 2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k) if precision_at_k + recall_at_k else 0.0
                report["gap_detection"][name] = {f"Precision@{a.k}": round(precision_at_k, 3),
                                                 f"Recall@{a.k}": round(recall_at_k, 3), f"F1@{a.k}": round(f1_at_k, 3),
                                                 "gold_ranks": ranks,
                                                 "MRR": round(sum(1 / x for x in ranks) / len(gold), 3) if gold else 0.0}

        # ---------- information extraction ----------
        if a.ie_gold:
            gold_doc = json.load(open(a.ie_gold, encoding="utf-8"))
            gold = load_ie_gold_document(gold_doc); tp = fp = fn = 0
            by_title = {p["title"]: p["id"] for p in papers}
            for g in gold:
                pid = next((i for i, filename in uploaded_files.items() if filename == g.get("filename")), None)
                title = g.get("title", "")
                pid = pid or by_title.get(title) or next((i for t, i in by_title.items() if title and title[:40] in (t or "")), None)
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
            qs = json.load(open(a.queries, encoding="utf-8")); db = SessionLocal()
            uid = c.get("/auth/me").json()["id"]
            ids = rag.owned_ready_ids(db, uid)
            chunks = db.query(Chunk).filter(Chunk.paper_id.in_(ids)).all()
            abstract_papers = db.query(Paper).filter(Paper.id.in_(ids)).order_by(Paper.id).all()
            missing_files = sorted({name for q in qs for name in q["relevant_files"]} - set(uploaded_files.values()))
            if missing_files:
                raise ValueError(f"Retrieval labels reference files outside this corpus: {missing_files}")
            abstract_texts = [f"{p.title or ''}. {p.abstract or ''}" for p in abstract_papers]
            tf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit(abstract_texts)
            M = tf.transform(abstract_texts); emb = get_embedder()
            K = a.k
            report["retrieval_config"] = {"candidate_chunks": settings.RETRIEVE_K,
                                          "reranker_output_chunks": settings.RERANK_N}
            def paper_rank(paper_ids):
                seen = []
                for pid in paper_ids:
                    if pid not in seen: seen.append(pid)
                return seen
            def metrics(rank_fn):
                prec = rec = mrr = ndcg = elapsed = 0.0
                for q in qs:
                    rel = {pid for pid, filename in uploaded_files.items() if filename in q["relevant_files"]}
                    if not rel:
                        raise ValueError(f"Query has no relevant source file in this corpus: {q['q']!r}")
                    started = time.perf_counter()
                    got = paper_rank(rank_fn(q["q"]))[:K]
                    elapsed += time.perf_counter() - started
                    hits = [1 if g in rel else 0 for g in got]
                    prec += sum(hits) / K
                    rec += sum(hits) / (len(rel) or 1)
                    mrr += next((1 / (i + 1) for i, h in enumerate(hits) if h), 0)
                    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits)); idcg = sum(1 / math.log2(i + 2) for i in range(min(len(rel), K)))
                    ndcg += dcg / idcg if idcg else 0
                n = len(qs); return {f"Precision@{K}": round(prec / n, 3), f"Recall@{K}": round(rec / n, 3),
                                      "MRR": round(mrr / n, 3), f"NDCG@{K}": round(ndcg / n, 3),
                                      "mean_query_seconds": round(elapsed / n, 4)}
            def tfidf(q):
                s = (M @ tf.transform([q]).T).toarray().ravel(); return [abstract_papers[i].id for i in np.argsort(-s)[:settings.RETRIEVE_K]]
            def embonly(q):
                return [h[2]["paper_id"] for h in get_store().query(emb.encode([q], is_query=True)[0], settings.RETRIEVE_K, paper_ids=ids)]
            def full(q):
                return [x["paper_id"] for x in rag.retrieve(db, q, ids, k=settings.RETRIEVE_K, n=settings.RERANK_N)]
            report["retrieval"] = {"Baseline 1: TF-IDF keyword": metrics(tfidf), "Baseline 2: embedding only": metrics(embonly),
                                   "Proposed: embedding + rerank": metrics(full)}

            if a.rag_ablation_draft:
                filename_by_paper_id = uploaded_files
                rag_rows = []
                for query in qs:
                    question = query["q"]
                    basic = answer_basic_rag(db, question, ids, llm.complete)
                    proposed_response = rag.answer_question(db, uid, question, ids)
                    if proposed_response.get("generation_mode", proposed_response.get("mode")) != "llm":
                        raise RuntimeError("Proposed RAG did not use live LLM generation; "
                                           "check provider errors or rate limits before reporting a live ablation")
                    proposed_contexts = rag.retrieve(db, question, ids, k=settings.RETRIEVE_K, n=settings.RERANK_N)
                    proposed_markers = [int(value) for value in re.findall(r"\[(\d+)\]", proposed_response.get("answer", ""))]
                    proposed_invalid = sorted({marker for marker in proposed_markers
                                               if marker < 1 or marker > len(proposed_contexts)})
                    proposed = {
                        "answer": proposed_response.get("answer", ""),
                        "contexts": proposed_contexts,
                        "citations": proposed_response.get("citations", []),
                        "invalid_citation_markers": proposed_invalid,
                        "mode": proposed_response.get("mode"),
                        "automated_unverified_claims": proposed_response.get("unverified"),
                    }
                    row = review_record(question, query["relevant_files"],
                                        {"baseline_3_basic_rag": basic, "proposed_rag": proposed},
                                        filename_by_paper_id)
                    row["methods"]["baseline_3_basic_rag"]["generation_mode"] = "live_llm"
                    row["methods"]["proposed_rag"]["generation_mode"] = proposed_response.get("generation_mode", proposed_response.get("mode"))
                    row["methods"]["proposed_rag"]["delivered_mode"] = proposed_response.get("mode")
                    row["methods"]["proposed_rag"]["omitted_unverified_claims"] = proposed_response.get("omitted_unverified_claims", 0)
                    row["methods"]["proposed_rag"]["automated_unverified_claims"] = proposed_response.get("unverified")
                    rag_rows.append(row)
                a.rag_ablation_draft.parent.mkdir(parents=True, exist_ok=True)
                worksheet = {
                    "status": "live_outputs_awaiting_human_faithfulness_relevance_and_citation_review_not_final_metrics",
                    "corpus_size": len(papers),
                    "provider": llm.provider,
                    "model": llm.model,
                    "retrieval": {"candidate_k": settings.RETRIEVE_K, "context_n": settings.RERANK_N,
                                  "baseline_3": "embedding similarity; no reranking",
                                  "proposed": "embedding similarity followed by configured reranker"},
                    "review_fields": ["faithfulness_1_to_5", "answer_relevance_1_to_5",
                                      "context_relevance_1_to_5", "citation_correctness_1_to_5", "reviewer", "notes"],
                    "questions": rag_rows,
                }
                a.rag_ablation_draft.write_text(json.dumps(worksheet, ensure_ascii=False, indent=2) + "\n",
                                                encoding="utf-8")
                report["rag_ablation_draft"] = str(a.rag_ablation_draft)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
