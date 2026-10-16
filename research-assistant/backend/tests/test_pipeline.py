"""End-to-end test over the synthetic corpus: auth -> upload -> RAG -> compare -> gaps -> graph -> recommend -> review."""
import subprocess, sys, tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def sample_papers():
    with tempfile.TemporaryDirectory(prefix="ra-sample-papers-") as sample_dir:
        samples = Path(sample_dir)
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_sample_papers.py"), str(samples)], check=True)
        yield samples


@pytest.fixture(scope="module")
def client(sample_papers):
    with TestClient(app) as c:
        r = c.post("/auth/register", json={"name": "T", "email": "t@example.com", "password": "password123"})
        assert r.status_code == 200, r.text
        c.headers["Authorization"] = f"Bearer {r.json()['token']}"
        for pdf in sorted(sample_papers.glob("paper_*.pdf")):
            r = c.post("/papers/upload", files={"file": (pdf.name, pdf.read_bytes(), "application/pdf")})
            assert r.status_code == 200, r.text
        yield c


def test_auth_required():
    with TestClient(app) as c:
        assert c.get("/papers").status_code == 401


def test_local_frontend_origins_pass_cors_preflight(client):
    for origin in ("http://localhost:3000", "http://127.0.0.1:3000"):
        r = client.options("/auth/me", headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization",
        })
        assert r.status_code == 200
        assert r.headers["access-control-allow-origin"] == origin


def test_upload_validation(client):
    assert client.post("/papers/upload", files={"file": ("x.pdf", b"not a pdf", "application/pdf")}).status_code == 400


def test_upload_rejects_oversize_before_parsing(client, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "MAX_UPLOAD_MB", 1)
    payload = b"%PDF-" + b"x" * (1024 * 1024) + b"%%EOF"
    r = client.post("/papers/upload", files={"file": ("large.pdf", payload, "application/pdf")})
    assert r.status_code == 400 and "too large" in r.json()["detail"].lower()


def test_search_rate_limit_is_retryable(client, monkeypatch):
    import httpx
    from app.services import paper_search

    request = httpx.Request("GET", "https://api.example/search")
    response = httpx.Response(429, request=request, headers={"Retry-After": "30"})
    error = httpx.HTTPStatusError("rate limited", request=request, response=response)
    monkeypatch.setattr(paper_search, "search", lambda *args: (_ for _ in ()).throw(error))
    r = client.get("/papers/search?q=Parkinson&source=semanticscholar")
    assert r.status_code == 503
    assert r.headers["retry-after"] == "30"


def test_ingestion_and_extraction(client):
    ps = client.get("/papers").json()["papers"]
    assert len(ps) == 16 and all(p["status"] == "ready" for p in ps), [p["error"] for p in ps]
    cnn = next(p for p in ps if "Convolutional" in (p["title"] or ""))
    d = client.get(f"/papers/{cnn['id']}").json()
    f = d["fields"]
    assert d["title"] and d["year"] == 2019
    assert "Speech" not in [x["name"] for x in f["datasets"]["value"]]
    assert any(m["name"] == "CNN" for m in f["methods"]["value"])
    assert f["datasets"]["value"]
    assert f["research_problem"]["value"]["text"]


def test_normalized_problem_aliases_do_not_duplicate_paper_topics(client, monkeypatch):
    from app.db import Paper, PaperProblem, PaperTopic, Problem, SessionLocal, Topic
    from app.services import extraction

    user_id = client.get("/auth/me").json()["id"]
    db = SessionLocal()
    paper = Paper(filename="normalized-topic-test.pdf", title="Normalization test", upload_user_id=user_id, status="ready")
    canonical = Problem(name="canonical normalization test")
    try:
        db.add_all([paper, canonical]); db.commit()
        monkeypatch.setattr(extraction, "resolve_problem", lambda _db, _label: canonical)
        extraction._persist(db, paper, {}, [], [], ["alias one", "alias two"], [])
        assert db.query(PaperProblem).filter_by(paper_id=paper.id, problem_id=canonical.id).count() == 1
        topic = db.query(Topic).filter_by(label=canonical.name).one()
        assert db.query(PaperTopic).filter_by(paper_id=paper.id, topic_id=topic.id).count() == 1
    except Exception:
        db.rollback()
        raise
    finally:
        db.query(PaperProblem).filter_by(paper_id=paper.id).delete()
        db.query(PaperTopic).filter_by(paper_id=paper.id).delete()
        for topic_row in db.query(Topic).filter_by(label=canonical.name).all():
            db.delete(topic_row)
        if paper.id:
            db.delete(paper)
        if canonical.id and db.query(Problem).filter_by(id=canonical.id).count():
            db.delete(canonical)
        db.commit()
        db.close()


def test_rag_with_citations(client):
    r = client.post("/query", json={"question": "Which dataset was used with wav2vec 2.0 and what accuracy was achieved?"}).json()
    assert r["citations"] and "[" in r["answer"]
    assert r["unverified"] == 0
    out = client.post("/query", json={"question": "How does quantum chromodynamics affect protein folding in zebrafish?"}).json()
    assert out["answer"].startswith("Not found")


def test_rag_metadata_filters_intersect_paper_and_section(client):
    papers = client.get("/papers").json()["papers"]
    target = papers[0]
    question = client.get(f"/papers/{target['id']}").json()["abstract"]
    assert question

    scoped = client.post("/query", json={"question": question, "paper_ids": [target["id"]],
                                          "sections": ["abstract"]})
    assert scoped.status_code == 200, scoped.text
    citations = scoped.json()["citations"]
    assert citations
    assert all(c["paper_id"] == target["id"] and c["section"] == "abstract" for c in citations)

    empty = client.post("/query", json={"question": question, "paper_ids": [target["id"]],
                                         "sections": ["references"]})
    assert empty.status_code == 200, empty.text
    assert empty.json()["citations"] == []
    assert empty.json()["answer"].startswith("Not found")


def test_compare(client):
    ids = [p["id"] for p in client.get("/papers").json()["papers"]][:3]
    r = client.post("/compare", json={"paper_ids": ids}).json()
    assert len(r["table"]["rows"]) == 3 and r["narrative"]
    for row in r["table"]["rows"]:
        for dimension in ("dataset", "algorithm", "metric"):
            cell = row[dimension]
            if cell["text"] != "—":
                assert cell["chunk_id"], f"{dimension} comparison cell is missing its evidence pointer: {cell}"
                assert client.get(f"/chunks/{cell['chunk_id']}").status_code == 200


def test_contradiction_detection_requires_same_dataset_metric_and_method(client, monkeypatch):
    from app.config import settings
    from app.services import comparison

    ids = [p["id"] for p in client.get("/papers").json()["papers"][:2]]
    values = {ids[0]: 72.0, ids[1]: 88.0}

    def fake_fields(_db, paper_id):
        return {"result_records": {"value": [{"dataset": "SharedSet", "method": "CNN", "metric": "Accuracy",
                                             "value": values[paper_id], "sentence": "Reported accuracy.",
                                             "chunk_id": paper_id}]}}

    monkeypatch.setattr(comparison, "get_fields", fake_fields)
    monkeypatch.setattr(settings, "CONTRADICTION_TOLERANCE", 5.0)
    conflict = comparison.find_contradictions(None, ids)
    assert len(conflict) == 1 and conflict[0]["delta"] == 16.0

    values[ids[1]] = 88.0

    def mismatched_fields(_db, paper_id):
        result = fake_fields(_db, paper_id)
        result["result_records"]["value"][0]["dataset"] = "DifferentSet" if paper_id == ids[1] else "SharedSet"
        return result

    monkeypatch.setattr(comparison, "get_fields", mismatched_fields)
    assert comparison.find_contradictions(None, ids) == []


def test_delete_removes_user_paper_content_and_references(client, sample_papers):
    from app.db import (Citation, Chunk, PaperDataset, PaperField, PaperMethod, PaperProblem, PaperTopic,
                        SessionLocal)
    from app.services.embeddings import get_embedder
    from app.services.vectorstore import get_store

    other = TestClient(app)
    registration = other.post("/auth/register", json={"name": "Delete", "email": "delete@example.com",
                                                       "password": "password123"})
    assert registration.status_code == 200
    other.headers["Authorization"] = f"Bearer {registration.json()['token']}"
    pdf = sorted(sample_papers.glob("paper_*.pdf"))[0]
    uploaded = other.post("/papers/upload", files={"file": (pdf.name, pdf.read_bytes(), "application/pdf")})
    assert uploaded.status_code == 200
    pid = uploaded.json()["paper_id"]
    db = SessionLocal()
    chunk = db.query(Chunk).filter_by(paper_id=pid).first()
    assert chunk
    assert db.query(PaperField).filter_by(paper_id=pid).count() > 0
    query_vector = get_embedder().encode([chunk.text])[0]
    store = get_store()
    assert store.query(query_vector, 10, paper_ids=[pid])
    db.close()

    response = other.delete(f"/papers/{pid}")
    assert response.status_code == 200
    assert other.get(f"/papers/{pid}").status_code == 404
    assert other.get("/papers").json()["papers"] == []
    assert store.query(query_vector, 10, paper_ids=[pid]) == []

    db = SessionLocal()
    try:
        for model in (Chunk, PaperField, PaperMethod, PaperDataset, PaperProblem, PaperTopic):
            assert db.query(model).filter_by(paper_id=pid).count() == 0, model.__name__
        assert db.query(Citation).filter((Citation.citing_paper_id == pid) | (Citation.cited_paper_id == pid)).count() == 0
    finally:
        db.close()


def test_gap_detection_finds_planted_gap(client):
    r = client.post("/research-gaps", json={"max_gaps": 15}).json()
    assert r["gaps"], r
    for gap in r["gaps"]:
        if gap["tier"] == "evidence_backed":
            assert any(e["type"] in ("limitation", "future_work", "contradiction") for e in gap["evidence"]), gap
        if gap["components"]["contradiction"]:
            linked = [e for e in gap["evidence"] if e["type"] == "contradiction"]
            assert linked, gap
            for evidence in linked:
                assert evidence.get("chunk_id") and client.get(f"/chunks/{evidence['chunk_id']}").status_code == 200
    top = [(g["method"], g["dataset"]) for g in r["gaps"][:5]]
    print("TOP GAPS:", top)
    hit = next((g for g in r["gaps"] if g["method"] == "Transformer" and g["dataset"] == "PC-GITA"), None)
    assert hit is not None, top
    assert hit["tier"] == "evidence_backed"
    specific = [e for e in hit["evidence"] if e["type"] == "future_work"]
    # Only one planted sentence names both entities; generic PC-GITA future work
    # remains useful context without inflating support for Transformer specifically.
    assert len(specific) == 1
    assert all("transformer" in e["text"].lower() and "pc-gita" in e["text"].lower() for e in specific)
    assert any(e["type"] == "related_context" for e in hit["evidence"])
    assert abs(sum(hit["weights"].values()) - 1) < 1e-6
    assert r["matrix"]["problem_dataset"]["cells"]


def test_graph_recommend_review_agent(client):
    ps = [p["id"] for p in client.get("/papers").json()["papers"]]
    g = client.get("/knowledge-graph").json()
    assert {"Paper", "Method", "Dataset"} <= {n["type"] for n in g["nodes"]}
    assert client.get(f"/recommendations?paper_id={ps[0]}").json()["recommendations"]
    assert "Literature Review" in client.post("/literature-review", json={"paper_ids": ps[:5]}).json()["markdown"]
    a = client.post("/agent", json={"request": "what is an under-explored direction in this field?"}).json()
    assert a["route"] == "gap"


def test_isolation_between_users(client):
    from app.db import Chunk, SessionLocal

    other = TestClient(app)
    r = other.post("/auth/register", json={"name": "O", "email": "o@example.com", "password": "password123"})
    other.headers["Authorization"] = f"Bearer {r.json()['token']}"
    pid = client.get("/papers").json()["papers"][0]["id"]
    assert other.get(f"/papers/{pid}").status_code == 404
    assert other.get("/papers").json()["papers"] == []
    db = SessionLocal()
    try:
        chunk_id = db.query(Chunk.id).filter_by(paper_id=pid).first()[0]
    finally:
        db.close()
    assert other.get(f"/chunks/{chunk_id}").status_code == 404
    answer = other.post("/query", json={"question": "What methods were used?", "paper_ids": [pid]})
    assert answer.status_code == 200 and answer.json()["citations"] == []
    assert answer.json()["answer"].startswith("Not found")
    assert other.post("/compare", json={"paper_ids": [pid, pid]}).status_code == 404
    gaps = client.post("/research-gaps", json={"max_gaps": 1}).json()["gaps"]
    assert gaps
    gap_id = gaps[0]["id"]
    assert other.get(f"/research-gaps/{gap_id}").status_code == 404
    assert other.get("/recommendations", params={"gap_id": gap_id}).json()["recommendations"] == []
    assert other.get("/knowledge-graph").json()["nodes"] == []
    scoped = other.post("/research-gaps", json={"paper_ids": [pid]}).json()
    assert scoped["corpus_size"] == 0 and scoped["gaps"] == []
