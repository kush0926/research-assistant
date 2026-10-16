"""Research-gap detection (Parts 14 & 15) — the project's core contribution.

A candidate gap needs BOTH a structural signal (sparsity in the problem x dataset x method matrix, surrounded by
well-populated neighbours) AND, ideally, a textual signal (independent limitation / future-work statements,
contradictions). Sparsity-only candidates are kept but explicitly down-weighted and labelled "under-explored".
GapScore is an explicit formula with named, adjustable weights — not an LLM judgement.
"""
import json
import math
import uuid
from collections import Counter, defaultdict
from itertools import product

import numpy as np
from sqlalchemy.orm import Session

from ..config import settings
from ..db import (Chunk, Dataset, GapEvidence, Method, Paper, PaperDataset, PaperMethod, PaperProblem,
                  Problem, ResearchGap)
from .comparison import find_contradictions
from .embeddings import get_embedder
from .extraction import find_entities, get_fields, is_public
from .llm import llm

MATRIX_EXCLUDED_CATEGORIES = {"feature", "technique"}  # inputs/tricks are not "the method" axis
COMPONENTS = ["novelty", "sparsity", "importance", "feasibility", "temporal", "evidence", "opportunity", "contradiction"]


def _thresholds():
    bge = get_embedder().backend == "bge"
    return {"cluster": 0.72 if bge else 0.42, "semantic_match": 0.62 if bge else 0.30}


def _cluster_papers(vecs: np.ndarray) -> list[int]:
    n = len(vecs)
    if n < 6:
        return [0] * n
    try:
        from sklearn.cluster import HDBSCAN
        lab = HDBSCAN(min_cluster_size=2).fit_predict(vecs)
        nxt = int(lab.max()) + 1 if (lab >= 0).any() else 0
        return [int(l) if l >= 0 else nxt + i for i, l in enumerate(lab)] if (lab >= 0).any() else _kmeans(vecs)
    except Exception:
        return _kmeans(vecs)


def _kmeans(vecs):
    from sklearn.cluster import KMeans
    k = max(2, min(4, len(vecs) // 3))
    return [int(x) for x in KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(vecs)]


def _greedy_clusters(vecs: np.ndarray, thr: float) -> list[list[int]]:
    clusters: list[list[int]] = []
    cents: list[np.ndarray] = []
    for i, v in enumerate(vecs):
        best, bs = -1, thr
        for j, c in enumerate(cents):
            s = float(v @ c) / (np.linalg.norm(c) or 1)
            if s >= bs:
                best, bs = j, s
        if best < 0:
            clusters.append([i]); cents.append(v.copy())
        else:
            clusters[best].append(i); cents[best] = cents[best] + v
    return clusters


def _related_contradictions(problem: str, dataset: str, method: str, contradictions: list[dict],
                            paper_problems: dict[int, set[str]]) -> list[dict]:
    """Return conflicts on the same dataset/method and the candidate's problem scope."""
    related = []
    for ct in contradictions:
        if ct.get("dataset") != dataset or ct.get("method") != method:
            continue
        problem_sets = [paper_problems.get(pair["paper_id"], set()) for pair in ct.get("papers", [])]
        # If labels exist, every labeled source paper must include this problem. Unknown labels remain usable.
        if any(problem_sets) and not all(not labels or problem in labels for labels in problem_sets):
            continue
        related.append(ct)
    return related


def _matches_candidate_evidence(problem: str, dataset: str, method: str, item: dict, similarity: float,
                                threshold: float, paper_problems: dict[int, set[str]]) -> bool:
    """Keep limitation evidence within the candidate's problem scope when labels are available."""
    known_problems = item["p"] or paper_problems.get(item["paper_id"], set())
    if known_problems and problem not in known_problems:
        return False
    return dataset in item["d"] or method in item["m"] or similarity >= threshold


def select_corpus(db: Session, user_id: int, paper_ids=None, topic=None) -> list[Paper]:
    q = db.query(Paper).filter(Paper.upload_user_id == user_id, Paper.status == "ready")
    if paper_ids:
        q = q.filter(Paper.id.in_(paper_ids))
    papers = q.all()
    if topic and len(papers) > 3:
        emb = get_embedder()
        vs = emb.encode([f"{p.title}. {p.abstract or ''}" for p in papers])
        qv = emb.encode([topic], is_query=True)[0]
        order = np.argsort(-(vs @ qv))
        keep = max(3, math.ceil(0.75 * len(papers)))
        papers = [papers[i] for i in order[:keep]]
    return papers


def detect_gaps(db: Session, user_id: int, paper_ids=None, topic=None, weights=None, max_gaps: int = 15) -> dict:
    papers = select_corpus(db, user_id, paper_ids, topic)
    if len(papers) < 3:
        return {"error": "Need at least 3 processed papers for gap detection.", "gaps": [], "corpus_size": len(papers)}
    W = {k: float((weights or settings.GAP_WEIGHTS).get(k, 0.0)) for k in COMPONENTS}
    tot = sum(W.values()) or 1.0
    W = {k: v / tot for k, v in W.items()}
    emb, th = get_embedder(), _thresholds()
    pid_list = [p.id for p in papers]
    by_id = {p.id: p for p in papers}

    # ---- 1-3. clustering + entity tables ----
    vecs = emb.encode([f"{p.title}. {p.abstract or ''}" for p in papers])
    cluster_of = dict(zip(pid_list, _cluster_papers(vecs)))
    names = lambda model, link, col: {  # noqa
        r[0]: r[1] for r in db.query(model.id, model.name).all()}
    prob_names, ds_names, m_names = (names(Problem, 0, 0), names(Dataset, 0, 0), names(Method, 0, 0))
    m_cat = {m.name: m.category for m in db.query(Method).all()}
    P_of, D_of, M_of = defaultdict(set), defaultdict(set), defaultdict(set)
    for r in db.query(PaperProblem).filter(PaperProblem.paper_id.in_(pid_list)):
        P_of[r.paper_id].add(prob_names[r.problem_id])
    for r in db.query(PaperDataset).filter(PaperDataset.paper_id.in_(pid_list)):
        D_of[r.paper_id].add(ds_names[r.dataset_id])
    for r in db.query(PaperMethod).filter(PaperMethod.paper_id.in_(pid_list)):
        if m_cat.get(m_names[r.method_id]) not in MATRIX_EXCLUDED_CATEGORIES:
            M_of[r.paper_id].add(m_names[r.method_id])

    # ---- 4-5. sparsity matrix ----
    T, PD, PM, DM = Counter(), Counter(), Counter(), Counter()
    P, D, M = Counter(), Counter(), Counter()
    tri_papers, pd_papers, pm_papers = defaultdict(list), defaultdict(list), defaultdict(list)
    for pid in pid_list:
        ps, ds, ms = P_of[pid], D_of[pid], M_of[pid]
        for p in ps: P[p] += 1
        for d in ds: D[d] += 1
        for m in ms: M[m] += 1
        for p, d in product(ps, ds): PD[(p, d)] += 1; pd_papers[(p, d)].append(pid)
        for p, m in product(ps, ms): PM[(p, m)] += 1; pm_papers[(p, m)].append(pid)
        for d, m in product(ds, ms): DM[(d, m)] += 1
        for t in product(ps, ds, ms): T[t] += 1; tri_papers[t].append(pid)
    years = {p.id: p.year for p in papers}
    max_year = max([y for y in years.values() if y] or [None]) if any(years.values()) else None
    min_nb = 2 if len(papers) >= 10 else 1

    # ---- limitation / future-work mining (needed for evidence + cluster candidates) ----
    ev_items = []
    for pid in pid_list:
        f = get_fields(db, pid)
        for typ in ("limitations", "future_work"):
            for it in f.get(typ, {}).get("value") or []:
                ev_items.append({"paper_id": pid, "type": "limitation" if typ == "limitations" else "future_work",
                                 "text": it["text"], "chunk_id": it.get("chunk_id"),
                                 "m": set(find_entities(it["text"], "methods")), "d": set(find_entities(it["text"], "datasets")),
                                 "p": set(find_entities(it["text"], "problems"))})
    ev_vecs = emb.encode([e["text"] for e in ev_items]) if ev_items else np.zeros((0, emb.dim))
    contradictions = find_contradictions(db, pid_list)

    # ---- 6-7. candidate generation: well-populated neighbours, empty (or abandoned) cell ----
    well_cells = [t for t, c in T.items() if c >= min_nb]
    well_vecs = emb.encode([" | ".join(t) for t in well_cells]) if well_cells else np.zeros((0, emb.dim))
    max_cit = max([p.citation_count or 0 for p in papers] + [1])
    prob_stats = {}
    for p in P:
        ps = [by_id[i] for i in pid_list if p in P_of[i]]
        recent = [x for x in ps if x.year and max_year and x.year >= max_year - 2]
        prob_stats[p] = {"n": len(ps), "cit": sum((x.citation_count or 0) for x in ps),
                         "recent": len(recent), "has_years": any(x.year for x in ps)}
    max_pn = max([s["n"] for s in prob_stats.values()] + [1])
    max_pc = max([s["cit"] for s in prob_stats.values()] + [1])

    # Embed every candidate cell in ONE batch (per-candidate encode calls were the slow path with BGE).
    pending = [(p, d, m) for p, d in PD if PD[(p, d)] >= min_nb for m in M if M[m] >= min_nb]
    cand_vec = dict(zip(pending, emb.encode([" | ".join(t) for t in pending]))) if pending else {}

    def comps(p, d, m, cell, nrow, ncol):
        v = cand_vec[(p, d, m)]
        novelty = 1.0 - float((well_vecs @ v).max()) if len(well_vecs) else 1.0
        sparsity = max(0.0, 1.0 - cell / (((nrow + ncol) / 2) or 1))
        st = prob_stats[p]
        importance = 0.5 * st["n"] / max_pn + 0.5 * (st["cit"] / max_pc if max_pc > 1 or st["cit"] else st["n"] / max_pn)
        feasibility = (is_public("datasets", d) + is_public("methods", m)) / 2
        temporal = 0.5 if not st["has_years"] else 0.5 * (1.0 if st["recent"] else 0.0) + 0.5 * (st["recent"] / st["n"])
        pd_clusters = {cluster_of[i] for i in pd_papers.get((p, d), [])}
        elsewhere = [i for i in pid_list if m in M_of[i] and cluster_of[i] not in pd_clusters]
        opportunity = min(1.0, len(elsewhere) / 3)
        # Match explicit entities or semantic relevance within the statement's problem scope.
        sem = (ev_vecs @ v) if len(ev_vecs) else np.zeros(0)
        related_items = [e for e, similarity in zip(ev_items, sem)
                         if _matches_candidate_evidence(p, d, m, e, float(similarity), th["semantic_match"], P_of)]
        # Semantic similarity also retrieves broad topic-level future work. Keep it
        # as related context; only an explicit dataset AND method anchor contributes to
        # a combination's evidence score. Repeated themes are scored separately.
        hit_items = [e for e in related_items if d in e["d"] and m in e["m"]]
        related_context = [e for e in related_items if e not in hit_items]
        indep = {e["paper_id"] for e in hit_items}
        evidence = min(len(indep), 5) / 5
        # overlap must be on the dataset or method — problem-level overlap is too coarse in a single-topic corpus
        related_contradictions = _related_contradictions(p, d, m, contradictions, P_of)
        contradiction = 1.0 if related_contradictions else 0.0
        return ({"novelty": novelty, "sparsity": sparsity, "importance": importance, "feasibility": feasibility,
                 "temporal": temporal, "evidence": evidence, "opportunity": opportunity, "contradiction": contradiction},
                hit_items, elsewhere, related_contradictions, related_context)

    def score_of(c, tier_evidence):
        s = sum(W[k] * c[k] for k in COMPONENTS)
        if c["evidence"] == 0 and c["contradiction"] == 0:
            s *= settings.SPARSITY_ONLY_PENALTY
        return round(float(s), 4)

    cands = []
    for p, d in PD:
        if PD[(p, d)] < min_nb:
            continue
        for m in M:
            if M[m] < min_nb:
                continue
            cell = T.get((p, d, m), 0)
            yrs = [years[i] for i in tri_papers.get((p, d, m), []) if years.get(i)]
            abandoned = cell == 1 and yrs and max_year and max(yrs) <= max_year - 3
            if cell > 0 and not abandoned:
                continue
            nrow, ncol = PD[(p, d)], M[m]
            c, hits, elsewhere, related_contradictions, related_context = comps(p, d, m, cell, nrow, ncol)
            cands.append({"type": "combination", "problem": p, "dataset": d, "method": m, "cell_count": cell,
                          "neighbor_row": nrow, "neighbor_col": ncol, "components": c, "hits": hits,
                          "related_contradictions": related_contradictions,
                          "related_context": related_context,
                          "kind": "explored_early_then_abandoned" if abandoned else "never_explored",
                          "score": score_of(c, bool(hits)), "elsewhere": elsewhere})
    cands.sort(key=lambda x: -x["score"])
    cands = cands[: max_gaps * 2]

    # ---- evidence-cluster candidates (textual signal on its own) ----
    for cl in sorted(_greedy_clusters(ev_vecs, th["cluster"]) if len(ev_items) else [], key=len, reverse=True)[:6]:
        indep = {ev_items[i]["paper_id"] for i in cl}
        if len(indep) < 2:
            continue
        rep = max(cl, key=lambda i: float(ev_vecs[i] @ ev_vecs[cl].mean(axis=0)))
        probs = Counter(p for i in cl for p in P_of[ev_items[i]["paper_id"]])
        p0 = probs.most_common(1)[0][0] if probs else "corpus"
        v = ev_vecs[rep]
        novelty = 1.0 - float((well_vecs @ v).max()) if len(well_vecs) else 1.0
        c = {"novelty": novelty, "sparsity": 0.0, "importance": prob_stats.get(p0, {"n": 1})["n"] / max_pn,
             "feasibility": 0.5, "temporal": 0.5, "evidence": min(len(indep), 5) / 5, "opportunity": 0.0,
             "contradiction": 0.0}
        cands.append({"type": "evidence_cluster", "problem": p0, "dataset": None, "method": None, "cell_count": None,
                      "neighbor_row": None, "neighbor_col": None, "components": c,
                      "hits": [ev_items[i] for i in cl], "kind": "repeated_future_work",
                      "score": score_of(c, True), "rep_text": ev_items[rep]["text"], "elsewhere": []})

    # ---- contradiction candidates ----
    for ct in contradictions[:4]:
        pset = {p for x in ct["papers"] for p in P_of[x["paper_id"]]}
        c = {"novelty": 0.5, "sparsity": 0.0, "importance": 0.5, "feasibility": 1.0, "temporal": 0.5,
             "evidence": 0.0, "opportunity": 0.0, "contradiction": 1.0}
        cands.append({"type": "contradiction", "problem": sorted(pset)[0] if pset else None, "dataset": ct["dataset"],
                      "method": ct["method"], "cell_count": None, "neighbor_row": None, "neighbor_col": None,
                      "components": c, "hits": [], "kind": "conflicting_results", "contradiction": ct,
                      "score": score_of(c, False), "elsewhere": []})
    cands.sort(key=lambda x: -x["score"])
    cands = cands[:max_gaps]

    # ---- 11-12. evidence verification + output ----
    gaps = []
    for c in cands:
        ev = []
        for h in sorted(c["hits"], key=lambda e: e["paper_id"])[:8]:
            ch = db.get(Chunk, h["chunk_id"]) if h.get("chunk_id") else None
            ev.append({"type": h["type"], "paper_id": h["paper_id"], "paper_title": by_id[h["paper_id"]].title,
                       "chunk_id": h.get("chunk_id"), "page": ch.page_number if ch else None, "text": h["text"]})
        for h in sorted(c.get("related_context", []), key=lambda e: e["paper_id"])[:5]:
            ch = db.get(Chunk, h["chunk_id"]) if h.get("chunk_id") else None
            ev.append({"type": "related_context", "paper_id": h["paper_id"],
                       "paper_title": by_id[h["paper_id"]].title, "chunk_id": h.get("chunk_id"),
                       "page": ch.page_number if ch else None, "text": h["text"]})
        if c["type"] == "combination":
            nb_row = [{"paper_id": i, "title": by_id[i].title} for i in pd_papers[(c["problem"], c["dataset"])][:5]]
            nb_col = [{"paper_id": i, "title": by_id[i].title} for i in [q for q in pid_list if c["method"] in M_of[q]][:5]]
            ev.append({"type": "sparsity",
                       "text": f"Cell ({c['problem']} × {c['dataset']} × {c['method']}) has {c['cell_count']} paper(s); "
                               f"{c['dataset']} appears in {c['neighbor_row']} paper(s) on this problem and {c['method']} in {c['neighbor_col']} paper(s) overall.",
                       "neighbors": {"dataset_papers": nb_row, "method_papers": nb_col}})
            for ct in c.get("related_contradictions", []):
                for pair in ct["papers"]:
                    chunk = db.get(Chunk, pair["chunk_id"]) if pair.get("chunk_id") else None
                    ev.append({"type": "contradiction", "paper_id": pair["paper_id"],
                               "paper_title": by_id[pair["paper_id"]].title, "chunk_id": pair.get("chunk_id"),
                               "page": chunk.page_number if chunk else None, "text": pair["sentence"],
                               "note": f"Possible conflict: {ct['metric']} differs by {ct['delta']} points on "
                                       f"{ct['dataset']} ({ct.get('method') or 'method unspecified'})."})
            desc = (f"The extracted records in this uploaded corpus link {c['method']} with {c['dataset']} for "
                    f"{c['problem']} in {c['cell_count']} paper(s), although {c['dataset']} appears in "
                    f"{c['neighbor_row']} paper(s) on this problem and {c['method']} in {c['neighbor_col']} paper(s) overall. "
                    "Check the source papers and wider literature before treating this as an unevaluated combination.")
            trend = {"kind": c["kind"],
                     "dataset_years": sorted(years[i] for i in pd_papers[(c['problem'], c['dataset'])] if years.get(i)),
                     "method_years": sorted(years[q] for q in pid_list if c["method"] in M_of[q] and years.get(q))}
        elif c["type"] == "evidence_cluster":
            desc = f"Repeated limitation/future-work theme across {len({h['paper_id'] for h in c['hits']})} independent papers: “{c['rep_text']}”"
            trend = None
        else:
            ct = c["contradiction"]
            ev = [{"type": "contradiction", "paper_id": x["paper_id"], "paper_title": by_id[x["paper_id"]].title,
                   "chunk_id": x["chunk_id"], "text": x["sentence"]} for x in ct["papers"]]
            desc = (f"Unresolved conflict: papers report {ct['metric']} values differing by {ct['delta']} points for "
                    f"{ct['method'] or 'the same method'} on {ct['dataset']}. {ct['note']}")
            trend = None
        has_text = c["components"]["evidence"] > 0 or c["components"]["contradiction"] > 0
        tier = "evidence_backed" if has_text else "under_explored"
        conf = "Low" if tier == "under_explored" else ("High" if c["score"] >= 0.7 else "Medium" if c["score"] >= 0.45 else "Low")
        gaps.append({"type": c["type"], "tier": tier, "confidence": conf, "description": desc,
                     "problem": c["problem"], "dataset": c["dataset"], "method": c["method"],
                     "score": c["score"], "components": {k: round(v, 3) for k, v in c["components"].items()},
                     "weights": W, "cell_count": c["cell_count"], "trend": trend, "evidence": ev})

    if llm.available:  # LLM only narrates "why it matters", grounded on the evidence — never invents the gap
        for g in gaps[:8]:
            try:
                g["rationale"] = llm.complete(
                    "In 2 sentences explain why this candidate research gap matters. Use ONLY the facts given. "
                    "Related context is background; it does not establish a specific dataset/method gap or global novelty.",
                    json.dumps({"gap": g["description"], "evidence": [
                        {"type": e["type"], "text": e["text"]} for e in g["evidence"]][:6]}), 220)
            except Exception:
                break

    run_id = uuid.uuid4().hex[:12]
    _persist(db, user_id, run_id, gaps)
    return {"run_id": run_id, "corpus_size": len(papers), "scope": topic or "selected/all processed papers",
            "gaps": gaps, "matrix": _matrix_view(PD, PM, D, M, P), "trends": _trend_view(papers, M_of, years),
            "contradictions": contradictions, "weights": W,
            "disclaimer": "Candidate gaps, not certainties: scored from this corpus only; verify against the evidence."}


def _matrix_view(PD, PM, D, M, P):
    probs = [p for p, _ in P.most_common(6)]
    ds = [d for d, _ in D.most_common(8)]
    ms = [m for m, _ in M.most_common(8)]
    return {"problem_dataset": {"rows": probs, "cols": ds, "cells": [[PD.get((p, d), 0) for d in ds] for p in probs]},
            "problem_method": {"rows": probs, "cols": ms, "cells": [[PM.get((p, m), 0) for m in ms] for p in probs]}}


def _trend_view(papers, M_of, years):
    yrs = sorted({y for y in years.values() if y})
    methods = [m for m, _ in Counter(m for p in papers for m in M_of[p.id]).most_common(6)]
    return {"years": yrs, "series": {m: [sum(1 for p in papers if years.get(p.id) == y and m in M_of[p.id]) for y in yrs] for m in methods}}


def _persist(db, user_id, run_id, gaps):
    pid = {p.name: p.id for p in db.query(Problem).all()}
    did = {d.name: d.id for d in db.query(Dataset).all()}
    mid = {m.name: m.id for m in db.query(Method).all()}
    for g in gaps:
        row = ResearchGap(user_id=user_id, run_id=run_id, description=g["description"], gap_type=g["type"], tier=g["tier"],
                          problem_id=pid.get(g["problem"]), dataset_id=did.get(g["dataset"]), method_id=mid.get(g["method"]),
                          gap_score=g["score"], novelty=g["components"]["novelty"], sparsity=g["components"]["sparsity"],
                          evidence_score=g["components"]["evidence"],
                          components=json.dumps({"components": g["components"], "weights": g["weights"]}))
        db.add(row); db.flush()
        g["id"] = row.id
        for e in g["evidence"]:
            db.add(GapEvidence(gap_id=row.id, evidence_type=e["type"], source_paper_id=e.get("paper_id"),
                               source_chunk_id=e.get("chunk_id"), note=e["text"][:1000]))
    db.commit()
