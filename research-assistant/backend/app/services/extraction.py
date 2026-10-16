"""Hybrid structured extraction (Part 12): rule/API fields + dictionary-NER entities + LLM interpretive fields.

Output per paper is stored in `paper_fields` with an evidence pointer (source_chunk_id) and a `method` tag
(rule | ner | llm) so every field is auditable. Also fills the relational method/dataset/problem tables
that the gap-detection matrix is built from.
"""
import json
import re
from collections import Counter
from pathlib import Path

from sqlalchemy.orm import Session

from ..db import (Author, Chunk, Dataset, Method, Paper, PaperAuthor, PaperDataset, PaperField, PaperMethod,
                  PaperProblem, PaperTopic, Problem, Topic)
from .embeddings import get_embedder, tokenize
from .llm import llm
from .pdf_processing import split_sentences

DICT = json.loads((Path(__file__).resolve().parent.parent / "data" / "domain_dict.json").read_text())

# ---------------- dictionary NER ----------------
def _compile(kind: str):
    out = []
    for canon, spec in DICT[kind].items():
        syns = spec["syn"] if isinstance(spec, dict) else spec
        alts = sorted({re.escape(s) for s in syns + [canon.lower()]}, key=len, reverse=True)
        out.append((canon, re.compile(r"(?<![A-Za-z0-9])(" + "|".join(alts) + r")(?![A-Za-z0-9])", re.I), [s for s in syns]))
    return out


_MATCH = {k: _compile(k) for k in ("methods", "datasets", "problems", "metrics")}


def find_entities(text: str, kind: str) -> Counter:
    found = Counter()
    for canon, rx, _ in _MATCH[kind]:
        for m in rx.finditer(text):
            tok = m.group(1)
            if len(tok) <= 4 and kind in ("methods", "datasets") and not (tok.isupper() or any(c.isupper() for c in tok[1:])):
                continue  # avoid 'gan', 'rag', 'gru' as ordinary words
            found[canon] += 1
    return found


def _entities_with_evidence(chunks: list[Chunk], kind: str, limit: int) -> list[dict]:
    """Aggregate entity counts while retaining a source chunk for each canonical entity."""
    counts, evidence = Counter(), {}
    for chunk in chunks:
        for name, count in find_entities(chunk.text, kind).items():
            counts[name] += count
            evidence.setdefault(name, chunk.id)
    return [{"name": name, "count": count, "chunk_id": evidence[name]}
            for name, count in counts.most_common(limit)]


_GENERIC = {"the", "this", "our", "a", "an", "public", "new", "large", "small", "speech", "voice", "audio", "training",
            "test", "testing", "validation", "clinical", "benchmark", "standard", "available", "original", "full", "same",
            "following", "above", "recording", "recordings", "healthy", "patient", "data",
            "each", "every", "both", "these", "those", "another", "other", "several", "multiple", "all", "any"}


def novel_datasets(text: str, known: set[str]) -> Counter:
    """Dataset names the dictionary does not know: '<Capitalised Name> dataset|corpus|database'. Fragments of
    already-known names (e.g. 'Speech' inside 'Italian Parkinson's Voice and Speech') and generic words are dropped."""
    known_low = " | ".join(k.lower() for k in known) + " | " + " | ".join(
        s.lower() for spec in DICT["datasets"].values() for s in spec["syn"]) + " | " + " | ".join(
        k.lower() for k in DICT["datasets"])
    c = Counter()
    for m in re.finditer(r"\b((?:[A-Z][A-Za-z0-9\-']+)(?: [A-Z][A-Za-z0-9\-']+){0,3}) (?:dataset|corpus|database)\b", text):
        name = m.group(1).strip()
        words = [w.lower() for w in name.split()]
        if all(w in _GENERIC for w in words) or name.lower() in known_low or len(name) < 3:
            continue
        c[name] += 1
    return c


def canonical(kind: str, name: str) -> str:
    low = name.lower().strip()
    for canon, _, syns in _MATCH[kind]:
        if low == canon.lower() or low in [s.lower() for s in syns]:
            return canon
    return name.strip()


def is_public(kind: str, name: str) -> float:
    spec = DICT[kind].get(name)
    if isinstance(spec, dict) and "public" in spec:
        return 1.0 if spec["public"] else 0.0
    return 0.5  # unknown availability


def method_category(name: str) -> str:
    spec = DICT["methods"].get(name)
    return spec.get("category", "") if isinstance(spec, dict) else ""


# ---------------- sentence cues ----------------
LIMIT_RE = re.compile(r"\b(limitations?|limited (by|to)|small (sample|dataset|corpus)|"
                      r"small number of (speakers|subjects|patients|participants|samples|recordings|examples)|"
                      r"only (evaluated|tested|trained|validated)|"
                      r"not (been )?(evaluated|tested|validated|explored)|lack(s|ed)? of|restricted to|does not generali[sz]e|"
                      r"poor generali[sz]ation|class imbalance|imbalanced|single dataset|few (speakers|subjects|participants|patients)|"
                      r"remain(s)? (unclear|unexplored|unaddressed)|unable to)\b", re.I)
FUTURE_RE = re.compile(r"\b(future (work|research|studies|directions?)|in the future|we plan to|should be (explored|investigated|evaluated|tested)|"
                       r"remains? (to be|an open)|further (work|research|investigation|studies)|worth (exploring|investigating)|"
                       r"would be (interesting|valuable|beneficial)|will (explore|investigate|evaluate) )\b", re.I)
CONTRIB_RE = re.compile(r"\b(we (propose|present|introduce|develop|contribute|demonstrate|show that)|our (main )?contributions?|"
                        r"this (paper|work|study) (propose|present|introduce|develop)s?)\b", re.I)
OBJ_RE = re.compile(r"\b((aim|objective|goal)s? (is|are|of this)|this (paper|study|work|article) (aims|seeks|investigates|addresses|explores|examines)|"
                    r"we (aim|seek|investigate|address|explore|examine))\b", re.I)
PROPOSE_RE = re.compile(r"\b(we (propose|present|introduce|develop|design)|proposed|our (model|method|approach|framework|system))\b", re.I)
SETUP_RE = re.compile(r"\b(split|folds?|cross-validation|learning rate|epochs?|batch size|optimi[sz]er|hyper-?parameters?|train(ing)? (set|data)|test set|validation)\b", re.I)
RESULT_RE = re.compile(r"(\d{1,3}(?:\.\d+)?)\s?%?\s*(?:\(?[\w\- ]{0,12}\)?\s*)?(accuracy|f1(?:[- ]score)?|auc|auroc|precision|recall|sensitivity|specificity)"
                       r"|(accuracy|f1(?:[- ]score)?|auc|auroc|precision|recall|sensitivity|specificity)(?: score)?\s*(?:of|was|is|reached|=|:)?\s*(\d{1,3}(?:\.\d+)?)\s?%?", re.I)
KEY_NORM = {"f1-score": "F1", "f1 score": "F1", "f1": "F1", "auroc": "AUC", "auc": "AUC", "sensitivity": "Recall"}


def _norm_metric(m: str) -> str:
    m = m.lower()
    return KEY_NORM.get(m, m.capitalize())


def _sections(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    d: dict[str, list[Chunk]] = {}
    for c in chunks:
        d.setdefault(c.section, []).append(c)
    return d


def _sentence_with_chunk(chunks: list[Chunk], regex: re.Pattern, limit: int, skip_seen: set | None = None):
    out, seen = [], skip_seen if skip_seen is not None else set()
    for c in chunks:
        for s in split_sentences(c.text):
            if 25 < len(s) < 420 and regex.search(s) and s not in seen:
                seen.add(s); out.append({"text": s, "chunk_id": c.id, "page": c.page_number})
                if len(out) >= limit:
                    return out
    return out


# ---------------- results records (for comparison + contradictions) ----------------
def extract_results(chunks: list[Chunk], default_dataset: str | None, default_method: str | None) -> list[dict]:
    recs = []
    for c in chunks:
        if c.section not in ("results", "experiments", "abstract", "discussion", "conclusion"):
            continue
        for s in split_sentences(c.text):
            for m in RESULT_RE.finditer(s):
                if re.search(r"%\s*of\b|\b(maximum|maximal|relative)\b", m.group(0), re.I):
                    continue
                if re.match(r"\s+(improvement|increase|gain|reduction|decrease)\b", s[m.end():], re.I):
                    continue
                val = m.group(1) or m.group(4)
                met = m.group(2) or m.group(3)
                try:
                    v = float(val)
                except Exception:
                    continue
                if v <= 1.0 and "%" not in m.group(0):
                    v *= 100
                if not (0 < v <= 100):
                    continue
                ds = list(find_entities(s, "datasets")) or ([default_dataset] if default_dataset else [])
                md = [name for name in find_entities(s, "methods")
                      if method_category(name) not in ("feature", "technique")] or ([default_method] if default_method else [])
                recs.append({"dataset": ds[0] if len(ds) == 1 else None, "method": md[0] if len(md) == 1 else None,
                             "metric": _norm_metric(met), "value": round(v, 2), "sentence": s,
                             "chunk_id": c.id, "page": c.page_number})
    seen, out = set(), []
    for r in recs:  # de-duplicate
        k = (r["dataset"], r["method"], r["metric"], r["value"])
        if k not in seen:
            seen.add(k); out.append(r)
    return out[:25]


# ---------------- LLM interpretive extraction ----------------
_LLM_SYS = ("You extract structured fields from a research paper. Use ONLY the supplied text. "
            "Copy limitation and future_work items VERBATIM as sentences from the text (do not paraphrase). "
            "If a field is not present return an empty string / empty list.")


def llm_fields(sec: dict[str, list[Chunk]], title: str) -> dict | None:
    if not llm.available:
        return None

    def txt(*names, cap=2600):
        return " ".join(c.text for n in names for c in sec.get(n, []))[:cap]

    user = (f"TITLE: {title}\n\nABSTRACT+INTRO:\n{txt('abstract', 'introduction', cap=3200)}\n\n"
            f"METHOD:\n{txt('method', cap=2000)}\n\nDISCUSSION/LIMITATIONS/CONCLUSION:\n"
            f"{txt('discussion', 'limitations', 'future_work', 'conclusion', cap=3200)}\n\n"
            'Return JSON: {"problem": str, "problem_label": "<=6 word canonical task name", "objectives": [str], '
            '"research_questions": [str], "methodology": str, "contributions": [str], '
            '"experimental_setup": str, "limitations": [str], "future_work": [str], '
            '"methods": [{"name": str, "role": "proposed|used|baseline", "category": "classical|deep|transformer|llm|encoder|model", "quote": str}], '
            '"datasets": [{"name": str, "quote": str}]}. '
            'For methods include task classifiers and representation encoders actually used or evaluated, including baselines. '
            'Exclude metrics, software libraries, pooling, feature descriptors and preprocessing-only models. '
            'For datasets include proper names, not generic phrases or invented names for unnamed collections. '
            'For every entity copy one short, complete, verbatim supporting sentence from the supplied text as quote. '
            'Do not paraphrase quotes or join excerpts with ellipses. Return [] when no named entity is present.')
    data = llm.complete_json(_LLM_SYS, user, 2600)
    return data if isinstance(data, dict) else None


def _locate_chunk(chunks: list[Chunk], sentence: str) -> int | None:
    key = re.sub(r"\s+", " ", sentence)[:60]
    for c in chunks:
        if key in re.sub(r"\s+", " ", c.text):
            return c.id
    return None


def grounded_llm_entities(items, kind: str, chunks: list[Chunk]) -> list[dict] | None:
    """Accept only named entities anchored by a complete quote in one source chunk.

    None means the response is unusable and dictionary extraction should remain.
    A valid empty array explicitly reports no named entities in the supplied context.
    """
    if not isinstance(items, list):
        return None
    normalized = lambda value: " ".join(value.split()).casefold()
    accepted, seen = [], set()
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not isinstance(item.get("quote"), str):
            continue
        name, quote = item["name"].strip(), item["quote"].strip()
        if len(name) < 3 or len(quote) < 12 or all(w in _GENERIC for w in re.findall(r"[a-z]+", name.lower())):
            continue
        canon = canonical(kind, name)
        aliases = [name, canon]
        if canon in DICT[kind]:
            aliases.extend(DICT[kind][canon]["syn"])
        compact = lambda value: re.sub(r"[^a-z0-9]", "", value.casefold())
        if not any(compact(alias) in compact(quote) for alias in aliases if len(compact(alias)) >= 3):
            continue
        chunk = next((c for c in chunks if normalized(quote) in normalized(c.text)), None)
        if chunk is None or canon.casefold() in seen:
            continue
        result = {"name": canon, "count": 1, "chunk_id": chunk.id, "quote": quote}
        if kind == "methods":
            role = item.get("role", "used")
            if role not in ("proposed", "used", "baseline"):
                continue
            category = method_category(canon) or item.get("category", "model")
            if category not in ("classical", "deep", "transformer", "llm", "model", "encoder"):
                continue
            result.update(role=role, category=category)
        seen.add(canon.casefold())
        accepted.append(result)
    return accepted if accepted or not items else None


# ---------------- main entry ----------------
def analyze_paper(db: Session, paper: Paper, references_text: str = "") -> dict:
    chunks = db.query(Chunk).filter(Chunk.paper_id == paper.id).order_by(Chunk.chunk_index).all()
    sec = _sections(chunks)
    all_text = " ".join(c.text for c in chunks)
    fields: dict[str, tuple[object, int | None, str]] = {}

    def put(name, value, chunk_id=None, method="rule"):
        if value in (None, "", [], {}):
            return
        fields[name] = (value, chunk_id, method)

    # --- deterministic ---
    put("title", {"text": paper.title}, None, "rule")
    put("year", {"text": paper.year}, None, "rule")
    put("venue", {"text": paper.venue}, None, "rule")
    abs_chunk = (sec.get("abstract") or [None])[0]
    put("abstract", {"text": paper.abstract, "chunk_id": abs_chunk.id if abs_chunk else None}, abs_chunk.id if abs_chunk else None)
    if references_text:
        refs = [r.strip() for r in re.split(r"\n|(?<=\.)\s+(?=\[?\d+\]?\s)", references_text) if len(r.strip()) > 20][:80]
        put("references", refs)

    # --- NER / dictionary ---
    scope_m = [c for c in chunks if c.section in ("abstract", "method", "experiments", "results", "front", "conclusion")] or chunks
    scope_d = [c for c in chunks if c.section in ("abstract", "method", "experiments", "results", "front")] or chunks
    mtxt, dtxt = " ".join(c.text for c in scope_m), " ".join(c.text for c in scope_d)
    method_items = _entities_with_evidence(scope_m, "methods", 8)
    dataset_items = _entities_with_evidence(scope_d, "datasets", 10)
    known_datasets = {item["name"] for item in dataset_items}
    dataset_evidence = {item["name"]: item["chunk_id"] for item in dataset_items}
    dataset_counts = Counter({item["name"]: item["count"] for item in dataset_items})
    for chunk in scope_d:
        for name, count in novel_datasets(chunk.text, known_datasets).items():
            dataset_counts[name] += count
            dataset_evidence.setdefault(name, chunk.id)
    dataset_items = [{"name": name, "count": count, "chunk_id": dataset_evidence[name]}
                     for name, count in dataset_counts.most_common(10)]
    metric_scope = sec.get("results", []) + sec.get("experiments", []) or chunks
    metric_items = _entities_with_evidence(metric_scope, "metrics", 6)
    methods = Counter({item["name"]: item["count"] for item in method_items})
    proposed = set()
    for s in split_sentences(mtxt):
        if PROPOSE_RE.search(s):
            proposed |= set(find_entities(s, "methods"))
    # features (MFCC/spectrogram) are inputs, techniques are not "the method" for the gap matrix
    method_evidence = {item["name"]: item["chunk_id"] for item in method_items}
    m_list = [{"name": k, "role": "proposed" if k in proposed else "used", "count": v,
               "category": method_category(k), "chunk_id": method_evidence[k]} for k, v in methods.most_common(8)]
    d_list = dataset_items[:5]
    put("methods", m_list, None, "ner"); put("datasets", d_list, None, "ner")
    put("metrics", metric_items, None, "ner")

    # --- interpretive: LLM first, heuristic fallback ---
    L = llm_fields(sec, paper.title or "")
    if L:
        llm_methods = grounded_llm_entities(L.get("methods"), "methods", chunks)
        llm_datasets = grounded_llm_entities(L.get("datasets"), "datasets", chunks)
        if llm_methods:
            # The LLM sees bounded excerpts. Preserve dictionary discoveries outside
            # those excerpts and prefer quoted LLM records for matching names.
            names = {item["name"].casefold() for item in llm_methods}
            m_list = [{**item, "source_method": "llm"} for item in llm_methods] + [
                {**item, "source_method": "ner"} for item in m_list if item["name"].casefold() not in names]
            fields.pop("methods", None)
            put("methods", m_list, None, "hybrid")
        if llm_datasets:
            names = {item["name"].casefold() for item in llm_datasets}
            d_list = [{**item, "source_method": "llm"} for item in llm_datasets] + [
                {**item, "source_method": "ner"} for item in d_list if item["name"].casefold() not in names]
            fields.pop("datasets", None)
            put("datasets", d_list, None, "hybrid")
    ctx_chunks = [c for c in chunks if c.section in ("abstract", "introduction", "front")]
    fut_chunks = [c for c in chunks if c.section in ("limitations", "future_work", "discussion", "conclusion", "results")]
    seen: set = set()
    problem_text = ""
    if L and L.get("problem"):
        problem_text = L["problem"]
        put("research_problem", {"text": problem_text, "chunk_id": abs_chunk.id if abs_chunk else None}, None, "llm")
        put("objectives", [{"text": s, "chunk_id": _locate_chunk(chunks, s)} for s in L.get("objectives", [])], None, "llm")
        put("research_questions", [{"text": s, "chunk_id": None} for s in L.get("research_questions", [])], None, "llm")
        put("methodology", {"text": L.get("methodology", ""), "chunk_id": (sec.get("method") or [None])[0] and sec["method"][0].id}, None, "llm")
        put("contributions", [{"text": s, "chunk_id": _locate_chunk(chunks, s)} for s in L.get("contributions", [])], None, "llm")
        put("experimental_setup", {"text": L.get("experimental_setup", ""), "chunk_id": None}, None, "llm")
        put("limitations", [{"text": s, "chunk_id": _locate_chunk(chunks, s)} for s in L.get("limitations", [])], None, "llm")
        put("future_work", [{"text": s, "chunk_id": _locate_chunk(chunks, s)} for s in L.get("future_work", [])], None, "llm")
    # heuristic fill for anything the LLM did not return
    if "contributions" not in fields:
        put("contributions", _sentence_with_chunk(ctx_chunks + [c for c in chunks if c.section == "conclusion"], CONTRIB_RE, 3, seen))
    if "objectives" not in fields:
        o = _sentence_with_chunk(ctx_chunks, OBJ_RE, 3, seen)
        put("objectives", o)
    if "research_problem" not in fields:
        o = (fields.get("objectives", ([{}],))[0] or [{}])[0] if fields.get("objectives") else {}
        if o:
            problem_text = o["text"]
            put("research_problem", {"text": o["text"], "chunk_id": o.get("chunk_id")})
        else:
            sents = split_sentences(paper.abstract or all_text[:800])
            problem_text = sents[0] if sents else paper.title or ""
            put("research_problem", {"text": problem_text, "chunk_id": abs_chunk.id if abs_chunk else None})
    if "research_questions" not in fields:
        rq = [{"text": s, "chunk_id": c.id} for c in ctx_chunks for s in split_sentences(c.text) if s.endswith("?")][:3]
        put("research_questions", rq)
    if "methodology" not in fields:
        mc = sec.get("method", [])
        if mc:
            put("methodology", {"text": " ".join(split_sentences(mc[0].text)[:3]), "chunk_id": mc[0].id})
    if "experimental_setup" not in fields:
        es = _sentence_with_chunk([c for c in chunks if c.section in ("experiments", "method")], SETUP_RE, 3, set())
        if es:
            put("experimental_setup", {"text": " ".join(x["text"] for x in es), "chunk_id": es[0]["chunk_id"]})
    if "limitations" not in fields:
        put("limitations", _sentence_with_chunk(fut_chunks, LIMIT_RE, 5, seen))
    if "future_work" not in fields:
        put("future_work", _sentence_with_chunk(fut_chunks, FUTURE_RE, 5, seen))

    # --- results ---
    dd = d_list[0]["name"] if len(d_list) == 1 else None
    model_items = [m for m in m_list if m.get("category") not in ("feature", "technique")]
    proposed_items = [m for m in model_items if m["role"] == "proposed"]
    pm = proposed_items[0]["name"] if len(proposed_items) == 1 else (
        model_items[0]["name"] if len(model_items) == 1 else None)
    results = extract_results(chunks, dd, pm)
    put("result_records", results, results[0]["chunk_id"] if results else None)
    if results:
        top = sorted(results, key=lambda r: -r["value"])[:3]
        put("results", [{"text": r["sentence"], "chunk_id": r["chunk_id"]} for r in top], top[0]["chunk_id"])
    # --- keywords ---
    km = re.search(r"keywords?\s*[:—\-–]\s*([^.]{5,160})", all_text, re.I)
    kw = [k.strip() for k in re.split(r"[;,·]", km.group(1))][:8] if km else [w for w, _ in Counter(tokenize(paper.title + " " + (paper.abstract or ""))).most_common(6)]
    put("keywords", kw)

    # --- problem label (normalised) ---
    prob_scope = (paper.title or "") + ". " + (paper.abstract or "") + " " + " ".join(c.text for c in sec.get("introduction", [])[:2])
    probs = find_entities(prob_scope, "problems")
    labels = [k for k, _ in probs.most_common(2)]
    if not labels and L and L.get("problem_label"):
        labels = [L["problem_label"]]
    if not labels:
        base = re.sub(r"[^a-z0-9' ]", " ", (paper.title or problem_text).lower())
        toks = [t for t in tokenize(base) if t not in {"based", "approach", "using", "novel", "study", "analysis", "deep", "learning"}]
        labels = [" ".join(toks[:5]) or "general"]
    put("problems", [{"name": n} for n in labels], None, "ner" if probs else "rule")

    _persist(db, paper, fields, [(m["name"], m["role"]) for m in m_list], [d["name"] for d in d_list], labels, chunks)
    return {k: v[0] for k, v in fields.items()}


def _get_or_create(db, model, **kw):
    obj = db.query(model).filter_by(**kw).first()
    if not obj:
        obj = model(**kw); db.add(obj); db.flush()
    return obj


def resolve_problem(db: Session, label: str) -> Problem:
    """Entity normalisation: reuse an existing problem whose embedding is near-identical (synonym + embedding clustering)."""
    label = canonical("problems", label)
    exact = db.query(Problem).filter(Problem.name == label).first()
    if exact:
        return exact
    existing = db.query(Problem).all()
    if existing:
        emb = get_embedder()
        vs = emb.encode([label] + [p.name for p in existing])
        sims = vs[1:] @ vs[0]
        j = int(sims.argmax())
        thr = 0.82 if emb.backend == "bge" else 0.62
        if sims[j] >= thr:
            return existing[j]
    p = Problem(name=label); db.add(p); db.flush()
    return p


def _persist(db, paper, fields, methods, datasets, problems, chunks):
    db.query(PaperField).filter(PaperField.paper_id == paper.id).delete()
    for name, (val, cid, how) in fields.items():
        db.add(PaperField(paper_id=paper.id, field_name=name, field_value=json.dumps(val), source_chunk_id=cid, method=how))
    for model in (PaperMethod, PaperDataset, PaperProblem, PaperTopic):
        db.query(model).filter(model.paper_id == paper.id).delete()
    for name, role in methods:
        m = _get_or_create(db, Method, name=name)
        m.category = m.category or method_category(name)
        db.add(PaperMethod(paper_id=paper.id, method_id=m.id, role=role))
    for name in datasets:
        d = _get_or_create(db, Dataset, name=name)
        db.add(PaperDataset(paper_id=paper.id, dataset_id=d.id, role="used"))
    seen_problem_ids: set[int] = set()
    seen_topic_ids: set[int] = set()
    for label in problems:
        pr = resolve_problem(db, label)
        if pr.id not in seen_problem_ids:
            db.add(PaperProblem(paper_id=paper.id, problem_id=pr.id))
            seen_problem_ids.add(pr.id)
        t = _get_or_create(db, Topic, label=pr.name)
        if t.id not in seen_topic_ids:
            db.add(PaperTopic(paper_id=paper.id, topic_id=t.id, confidence=1.0))
            seen_topic_ids.add(t.id)
    db.commit()


def get_fields(db: Session, paper_id: int) -> dict:
    rows = db.query(PaperField).filter(PaperField.paper_id == paper_id).all()
    return {r.field_name: {"value": json.loads(r.field_value), "chunk_id": r.source_chunk_id, "method": r.method} for r in rows}


def set_authors(db: Session, paper: Paper, names: list[str]):
    db.query(PaperAuthor).filter(PaperAuthor.paper_id == paper.id).delete()
    for i, n in enumerate(names[:30]):
        a = _get_or_create(db, Author, name=n[:200])
        db.add(PaperAuthor(paper_id=paper.id, author_id=a.id, author_order=i))
    db.commit()
