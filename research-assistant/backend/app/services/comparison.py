"""Multi-paper comparison (Part 13): table = DB projection (no LLM); narrative = LLM over the table only."""
import json
from collections import defaultdict

from sqlalchemy.orm import Session

from ..config import settings
from ..db import Paper
from .extraction import get_fields
from .llm import llm

DIMENSIONS = ["problem", "dataset", "methodology", "algorithm", "metric", "results", "limitations", "future_work"]


def _cell(text, chunk_id=None, paper_id=None, page=None):
    return {"text": text or "—", "chunk_id": chunk_id, "paper_id": paper_id}


def build_table(db: Session, papers: list[Paper]) -> dict:
    rows = []
    for p in papers:
        f = get_fields(db, p.id)
        v = lambda k: f.get(k, {}).get("value")  # noqa
        cid = lambda k: f.get(k, {}).get("chunk_id")  # noqa
        lst = lambda k, n=2: [x for x in (v(k) or [])][:n]  # noqa
        best = sorted(v("result_records") or [], key=lambda r: -r["value"])[:2]
        row = {
            "paper_id": p.id, "title": p.title, "year": p.year,
            "problem": _cell((v("research_problem") or {}).get("text"), (v("research_problem") or {}).get("chunk_id"), p.id),
            "dataset": _cell(", ".join(d["name"] for d in v("datasets") or []), None, p.id),
            "methodology": _cell((v("methodology") or {}).get("text"), (v("methodology") or {}).get("chunk_id"), p.id),
            "algorithm": _cell(", ".join(f"{m['name']}{' (proposed)' if m['role']=='proposed' else ''}" for m in v("methods") or []), None, p.id),
            "metric": _cell(", ".join(m["name"] for m in v("metrics") or []), None, p.id),
            "results": _cell("; ".join(f"{r['value']}% {r['metric']}" + (f" ({r['method']}" if r.get('method') else " (") +
                                       (f" on {r['dataset']}" if r.get('dataset') else "") + ")" for r in best),
                             best[0]["chunk_id"] if best else None, p.id),
            "limitations": _cell(" ".join(x["text"] for x in lst("limitations")), (lst("limitations") or [{}])[0].get("chunk_id"), p.id),
            "future_work": _cell(" ".join(x["text"] for x in lst("future_work")), (lst("future_work") or [{}])[0].get("chunk_id"), p.id),
        }
        rows.append(row)
    return {"dimensions": DIMENSIONS, "rows": rows}


def find_contradictions(db: Session, paper_ids: list[int]) -> list[dict]:
    """Same dataset + metric + method, different papers, values differing beyond tolerance -> flag for human review."""
    groups = defaultdict(list)
    for pid in paper_ids:
        for r in get_fields(db, pid).get("result_records", {}).get("value") or []:
            if r.get("dataset"):
                groups[(r["dataset"], r["metric"], r.get("method"))].append({**r, "paper_id": pid})
    out = []
    for (ds, metric, method), recs in groups.items():
        by_paper = {}
        for r in recs:
            by_paper.setdefault(r["paper_id"], r)
        if len(by_paper) < 2:
            continue
        vals = [r["value"] for r in by_paper.values()]
        if max(vals) - min(vals) > settings.CONTRADICTION_TOLERANCE:
            out.append({"dataset": ds, "metric": metric, "method": method, "delta": round(max(vals) - min(vals), 2),
                        "papers": [{"paper_id": pid, "value": r["value"], "sentence": r["sentence"], "chunk_id": r["chunk_id"]}
                                   for pid, r in by_paper.items()],
                        "note": "Possible contradiction — comparable dataset/metric/method but different reported values. Needs human review."})
    return out


def narrative(table: dict, contradictions: list[dict]) -> str:
    rows = table["rows"]
    if llm.available:
        slim = [{"id": r["paper_id"], "title": r["title"], **{d: r[d]["text"] for d in DIMENSIONS}} for r in rows]
        try:
            return llm.complete(
                "You write a short comparative synthesis of research papers. Use ONLY the table provided; "
                "never assert anything not in the table. Cover similarities, differences, methodological trends, and conflicts. "
                "Refer to papers by title.", json.dumps({"table": slim, "possible_contradictions": contradictions})[:12000], 700)
        except Exception:
            pass
    # deterministic narrative
    ds = [set(x.strip() for x in r["dataset"]["text"].split(",") if x.strip() and x.strip() != "—") for r in rows]
    ms = [set(x.replace(" (proposed)", "").strip() for x in r["algorithm"]["text"].split(",") if x.strip() and x.strip() != "—") for r in rows]
    shared_d = set.intersection(*ds) if ds and all(ds) else set()
    shared_m = set.intersection(*ms) if ms and all(ms) else set()
    all_m = set().union(*ms) if ms else set()
    parts = [f"Compared {len(rows)} papers: " + "; ".join(f"“{r['title']}” ({r['year'] or 'n.d.'})" for r in rows) + "."]
    parts.append(f"Shared datasets: {', '.join(sorted(shared_d)) or 'none common to all papers'}. "
                 f"Shared methods: {', '.join(sorted(shared_m)) or 'none common to all papers'}.")
    uniq = [f"“{r['title']}” uniquely uses {', '.join(sorted(m - set.union(*[x for j, x in enumerate(ms) if j != i])))}"
            for i, (r, m) in enumerate(zip(rows, ms)) if len(ms) > 1 and (m - set.union(*[x for j, x in enumerate(ms) if j != i]))]
    if uniq:
        parts.append("Differences: " + "; ".join(uniq) + ".")
    if all_m:
        parts.append(f"Methods across the set: {', '.join(sorted(all_m))}.")
    if contradictions:
        parts.append(f"{len(contradictions)} possible contradiction(s) flagged for human review (same dataset, metric and method, different reported values).")
    return " ".join(parts)


def compare(db: Session, papers: list[Paper]) -> dict:
    table = build_table(db, papers)
    contra = find_contradictions(db, [p.id for p in papers])
    return {"table": table, "narrative": narrative(table, contra), "contradictions": contra}
