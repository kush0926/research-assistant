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
SCORING_VERSION = "2"


def _normalize_weights(weights=None) -> dict[str, float]:
    raw = settings.GAP_WEIGHTS if weights is None else weights
    if set(raw) - set(COMPONENTS):
        raise ValueError("Unknown gap scoring component.")
    try:
        values = {k: float(raw.get(k, 0.0)) for k in COMPONENTS}
    except (TypeError, ValueError) as exc:
        raise ValueError("Gap weights must be finite, non-negative numbers.") from exc
    if any(not math.isfinite(v) or v < 0 for v in values.values()):
        raise ValueError("Gap weights must be finite, non-negative numbers.")
    # Scale before summing so even very large finite custom weights remain usable.
    largest = max(values.values())
    if largest <= 0:
        raise ValueError("At least one gap scoring weight must be greater than zero.")
    scaled = {k: v / largest for k, v in values.items()}
    total = sum(scaled.values())
    return {k: v / total for k, v in scaled.items()}


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _semantic_novelty(vector, occupied_vectors) -> float:
    if not len(occupied_vectors):
        return 1.0
    return 1.0 - _bounded(float((occupied_vectors @ vector).max()))


def _sparsity_score(cell_count, dataset_support, method_support, problem_support,
                    global_method_support, corpus_size) -> float:
    # Expected overlap under independence within the problem, smoothed by overall
    # method prevalence so transfers from another problem remain possible. An empty
    # cell with one weak neighbour must not score like a hole in a mature literature.
    expected = dataset_support * (0.75 * method_support / max(problem_support, 1)
                                  + 0.25 * global_method_support / max(corpus_size, 1))
    return _bounded((expected - cell_count) / (expected + 1.0))


def _evidence_score(paper_ids) -> float:
    count = len(set(paper_ids))
    return count / (count + 4.0)


def _temporal_score(paper_ids, years, latest_year) -> float:
    ids = set(paper_ids)
    if not ids or latest_year is None:
        return 0.5
    # Three-year half-life relative to the latest paper in the selected corpus.
    # Missing dates contribute a neutral prior rather than being treated as recent.
    return sum(2.0 ** (-max(0, latest_year - years[i]) / 3.0) if years.get(i) else 0.5
               for i in ids) / len(ids)


def _conflict_strength(conflicts) -> float:
    strengths = []
    tolerance = max(settings.CONTRADICTION_TOLERANCE, 1.0)
    for conflict in conflicts:
        count = len({p["paper_id"] for p in conflict.get("papers", [])})
        if count < 2:
            continue
        delta = max(0.0, float(conflict.get("delta", 0)))
        strengths.append((delta / (delta + tolerance)) * count / (count + 1.0))
    return max(strengths, default=0.0)


def _candidate_order(candidate):
    # Real ties remain ties; canonical labels make their ordering repeatable.
    return (-candidate["score"], candidate["type"], candidate.get("problem") or "",
            candidate.get("dataset") or "", candidate.get("method") or "",
            candidate.get("rep_text") or "", candidate.get("contradiction", {}).get("metric", ""))


def _thresholds():
    bge = get_embedder().backend == "bge"
    return {"cluster": 0.72 if bge else 0.42, "semantic_match": 0.62 if bge else 0.30}


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
    papers = q.order_by(Paper.id).all()
    if topic and len(papers) > 3:
        emb = get_embedder()
        vs = emb.encode([f"{p.title}. {p.abstract or ''}" for p in papers])
        qv = emb.encode([topic], is_query=True)[0]
        order = np.argsort(-(vs @ qv), kind="stable")
        keep = max(3, math.ceil(0.75 * len(papers)))
        papers = [papers[i] for i in order[:keep]]
    return papers


def detect_gaps(db: Session, user_id: int, paper_ids=None, topic=None, weights=None, max_gaps: int = 15) -> dict:
    W = _normalize_weights(weights)
    papers = select_corpus(db, user_id, paper_ids, topic)
    if len(papers) < 3:
        return {"error": "Need at least 3 processed papers for gap detection.", "gaps": [], "corpus_size": len(papers)}
    emb, th = get_embedder(), _thresholds()
    pid_list = [p.id for p in papers]
    by_id = {p.id: p for p in papers}

    # ---- 1-3. paper representations + entity tables ----
    vecs = emb.encode([f"{p.title}. {p.abstract or ''}" for p in papers])
    paper_vecs = dict(zip(pid_list, vecs))
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
        ps, ds, ms = sorted(P_of[pid]), sorted(D_of[pid]), sorted(M_of[pid])
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
    seen_evidence = set()
    for pid in pid_list:
        f = get_fields(db, pid)
        for typ in ("limitations", "future_work"):
            for it in f.get(typ, {}).get("value") or []:
                identity = (pid, " ".join(it["text"].lower().split()))
                if identity in seen_evidence:
                    continue
                seen_evidence.add(identity)
                ev_items.append({"paper_id": pid, "type": "limitation" if typ == "limitations" else "future_work",
                                 "text": it["text"], "chunk_id": it.get("chunk_id"),
                                 "m": set(find_entities(it["text"], "methods")), "d": set(find_entities(it["text"], "datasets")),
                                 "p": set(find_entities(it["text"], "problems"))})
    ev_items.sort(key=lambda e: (e["text"].casefold(), e["paper_id"], e["type"]))
    ev_vecs = emb.encode([e["text"] for e in ev_items]) if ev_items else np.zeros((0, emb.dim))
    contradictions = find_contradictions(db, pid_list)

    # ---- 6-7. candidate generation: well-populated neighbours, empty (or abandoned) cell ----
    well_cells = [t for t, c in T.items() if c >= min_nb]
    well_vecs = emb.encode([" | ".join(t) for t in well_cells]) if well_cells else np.zeros((0, emb.dim))
    log_citations = {p.id: math.log1p(max(0, p.citation_count or 0)) for p in papers}
    total_citations = sum(log_citations.values())

    def importance_of(ids):
        ids = set(ids)
        coverage = math.log1p(len(ids)) / math.log1p(len(papers))
        if not total_citations:
            return coverage
        # Log scaling limits the influence of a single highly cited paper. Zero
        # citation metadata throughout the corpus falls back to observed coverage.
        return 0.7 * coverage + 0.3 * sum(log_citations[i] for i in ids) / total_citations

    def feasibility_of(items):
        datasets = {d for item in items for d in item["d"]}
        methods = {m for item in items for m in item["m"]}
        availability = ([is_public("datasets", d) for d in sorted(datasets)]
                        + [is_public("methods", m) for m in sorted(methods)])
        return sum(availability) / len(availability) if availability else 0.5

    def score_of(components):
        contributions = {k: W[k] * components[k] for k in COMPONENTS}
        weighted_sum = sum(contributions.values())
        penalty = (_bounded(settings.SPARSITY_ONLY_PENALTY)
                   if components["evidence"] == 0 and components["contradiction"] == 0 else 1.0)
        return weighted_sum * penalty, {"weighted_sum": weighted_sum, "penalty": penalty,
                                        "contributions": contributions}

    def add_candidate(candidate, support_ids, evidence_ids):
        candidate["components"] = {k: _bounded(candidate["components"][k]) for k in COMPONENTS}
        candidate["score"], candidate["score_breakdown"] = score_of(candidate["components"])
        candidate["support"] = {"paper_count": len(set(support_ids)),
                                "evidence_paper_count": len(set(evidence_ids)),
                                "dated_paper_count": sum(bool(years.get(i)) for i in set(support_ids))}
        cands.append(candidate)

    # Embed every candidate cell in ONE batch (per-candidate encode calls were the slow path with BGE).
    pending = [(p, d, m) for p, d in PD if PD[(p, d)] >= min_nb for m in M if M[m] >= min_nb]
    cand_vec = dict(zip(pending, emb.encode([" | ".join(t) for t in pending]))) if pending else {}

    def comps(p, d, m, cell, nrow, ncol):
        v = cand_vec[(p, d, m)]
        pair_novelty = 1.0 - DM[(d, m)] / max(1, min(D[d], M[m]))
        novelty = 0.5 * _semantic_novelty(v, well_vecs) + 0.5 * pair_novelty
        sparsity = _sparsity_score(cell, nrow, PM[(p, m)], P[p], ncol, len(papers))
        row_ids = set(pd_papers[(p, d)])
        local_method_ids = set(pm_papers.get((p, m), []))
        support_ids = row_ids | local_method_ids
        importance = importance_of(support_ids)
        availability = (is_public("datasets", d) + is_public("methods", m)) / 2
        feasibility = 0.75 * availability + 0.25 * len(local_method_ids) / (len(local_method_ids) + 1.0)
        method_ids = [i for i in pid_list if m in M_of[i]]
        temporal = (_temporal_score(row_ids, years, max_year)
                    + _temporal_score(method_ids, years, max_year)) / 2
        elsewhere = [i for i in method_ids if i not in row_ids]
        relevance = [_bounded(float(paper_vecs[i] @ v)) for i in elsewhere]
        effective_support = sum(0.5 + 0.5 * similarity for similarity in relevance)
        opportunity = effective_support / (effective_support + 2.0)
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
        evidence = _evidence_score(indep)
        # overlap must be on the dataset or method — problem-level overlap is too coarse in a single-topic corpus
        related_contradictions = _related_contradictions(p, d, m, contradictions, P_of)
        contradiction = _conflict_strength(related_contradictions)
        return ({"novelty": novelty, "sparsity": sparsity, "importance": importance, "feasibility": feasibility,
                 "temporal": temporal, "evidence": evidence, "opportunity": opportunity, "contradiction": contradiction},
                hit_items, elsewhere, related_contradictions, related_context, support_ids)

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
            c, hits, elsewhere, related_contradictions, related_context, support_ids = comps(p, d, m, cell, nrow, ncol)
            conflict_ids = {pair["paper_id"] for ct in related_contradictions for pair in ct["papers"]}
            evidence_ids = {h["paper_id"] for h in hits} | conflict_ids
            add_candidate({"type": "combination", "problem": p, "dataset": d, "method": m, "cell_count": cell,
                          "neighbor_row": nrow, "neighbor_col": ncol, "components": c, "hits": hits,
                          "related_contradictions": related_contradictions,
                          "related_context": related_context,
                          "kind": "explored_early_then_abandoned" if abandoned else "never_explored",
                          "elsewhere": elsewhere}, support_ids | evidence_ids | set(elsewhere), evidence_ids)
    cands.sort(key=_candidate_order)
    cands = cands[: max_gaps * 2]

    # ---- evidence-cluster candidates (textual signal on its own) ----
    for cl in _greedy_clusters(ev_vecs, th["cluster"]) if len(ev_items) else []:
        indep = {ev_items[i]["paper_id"] for i in cl}
        if len(indep) < 2:
            continue
        centroid = ev_vecs[cl].mean(axis=0)
        rep = min(cl, key=lambda i: (-float(ev_vecs[i] @ centroid), ev_items[i]["text"].casefold(),
                                     ev_items[i]["paper_id"]))
        probs = Counter(p for i in cl for p in P_of[ev_items[i]["paper_id"]])
        p0 = min(probs, key=lambda p: (-probs[p], p)) if probs else "corpus"
        v = ev_vecs[rep]
        hits = [ev_items[i] for i in cl]
        c = {"novelty": _semantic_novelty(v, well_vecs), "sparsity": 0.0, "importance": importance_of(indep),
             "feasibility": feasibility_of(hits), "temporal": _temporal_score(indep, years, max_year),
             "evidence": _evidence_score(indep), "opportunity": 0.0,
             "contradiction": 0.0}
        add_candidate({"type": "evidence_cluster", "problem": p0, "dataset": None, "method": None, "cell_count": None,
                      "neighbor_row": None, "neighbor_col": None, "components": c,
                      "hits": hits, "kind": "repeated_future_work",
                      "rep_text": ev_items[rep]["text"], "elsewhere": []}, indep, indep)

    # ---- contradiction candidates ----
    conflict_problems = []
    for ct in contradictions:
        pset = {p for x in ct["papers"] for p in P_of[x["paper_id"]]}
        conflict_problems.append(min(pset) if pset else None)
    conflict_vecs = emb.encode([f"{p or 'corpus'} | {ct['dataset']} | {ct['method']}"
                               for p, ct in zip(conflict_problems, contradictions)]) if contradictions else []
    for p0, ct, v in zip(conflict_problems, contradictions, conflict_vecs):
        indep = {x["paper_id"] for x in ct["papers"]}
        c = {"novelty": _semantic_novelty(v, well_vecs), "sparsity": 0.0,
             "importance": importance_of(indep),
             "feasibility": (is_public("datasets", ct["dataset"]) + is_public("methods", ct["method"])) / 2,
             "temporal": _temporal_score(indep, years, max_year),
             "evidence": _evidence_score(indep), "opportunity": 0.0,
             "contradiction": _conflict_strength([ct])}
        add_candidate({"type": "contradiction", "problem": p0, "dataset": ct["dataset"],
                      "method": ct["method"], "cell_count": None, "neighbor_row": None, "neighbor_col": None,
                      "components": c, "hits": [], "kind": "conflicting_results", "contradiction": ct,
                      "elsewhere": []}, indep, indep)
    cands.sort(key=_candidate_order)
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
        source_count = c["support"]["evidence_paper_count"]
        conf = "Low" if tier == "under_explored" or source_count < 2 else "High" if source_count >= 3 else "Medium"
        gaps.append({"type": c["type"], "tier": tier, "confidence": conf, "description": desc,
                     "problem": c["problem"], "dataset": c["dataset"], "method": c["method"],
                     "score": round(c["score"], 4), "components": {k: round(v, 4) for k, v in c["components"].items()},
                     "score_breakdown": {"weighted_sum": round(c["score_breakdown"]["weighted_sum"], 6),
                                         "penalty": c["score_breakdown"]["penalty"],
                                         "contributions": {k: round(v, 6) for k, v in c["score_breakdown"]["contributions"].items()}},
                     "support": c["support"], "scoring_version": SCORING_VERSION,
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
            "gaps": gaps, "matrix": _matrix_view(PD, PM, D, M, P, T), "trends": _trend_view(papers, M_of, years),
            "contradictions": contradictions, "weights": W, "scoring_version": SCORING_VERSION,
            "disclaimer": "Candidate gaps, not certainties: scored from this corpus only; verify against the evidence."}


def _matrix_view(PD, PM, D, M, P, T):
    probs = [p for p, _ in P.most_common(6)]
    ds = [d for d, _ in D.most_common(8)]
    ms = [m for m, _ in M.most_common(8)]
    # Keep the full axes for the detector's three-way matrix. Sparse counts avoid a
    # dense problem x dataset x method payload; combinations absent here have count 0.
    return {"problem_dataset_method": {
                "problems": [p for p, _ in P.most_common()],
                "datasets": [d for d, _ in D.most_common()],
                "methods": [m for m, _ in M.most_common()],
                "counts": [{"problem": p, "dataset": d, "method": m, "count": count}
                           for (p, d, m), count in T.items() if count > 0]},
            "problem_dataset": {"rows": probs, "cols": ds, "cells": [[PD.get((p, d), 0) for d in ds] for p in probs]},
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
                          components=json.dumps({"components": g["components"], "weights": g["weights"],
                                                 "score_breakdown": g["score_breakdown"], "support": g["support"],
                                                 "scoring_version": SCORING_VERSION}))
        db.add(row); db.flush()
        g["id"] = row.id
        for e in g["evidence"]:
            db.add(GapEvidence(gap_id=row.id, evidence_type=e["type"], source_paper_id=e.get("paper_id"),
                               source_chunk_id=e.get("chunk_id"), note=e["text"][:1000]))
    db.commit()
