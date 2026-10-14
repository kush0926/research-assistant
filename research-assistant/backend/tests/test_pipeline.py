"""End-to-end test over the synthetic corpus: auth -> upload -> RAG -> compare -> gaps -> graph -> recommend -> review."""
import subprocess, sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = Path("/tmp/ra_sample_papers")


@pytest.fixture(scope="module")
def client():
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_sample_papers.py"), str(SAMPLES)], check=True)
    with TestClient(app) as c:
        r = c.post("/auth/register", json={"name": "T", "email": "t@example.com", "password": "password123"})
        assert r.status_code == 200, r.text
        c.headers["Authorization"] = f"Bearer {r.json()['token']}"
        for pdf in sorted(SAMPLES.glob("paper_*.pdf")):
            r = c.post("/papers/upload", files={"file": (pdf.name, pdf.read_bytes(), "application/pdf")})
            assert r.status_code == 200, r.text
        yield c


def test_auth_required():
    with TestClient(app) as c:
        assert c.get("/papers").status_code == 401


def test_upload_validation(client):
    assert client.post("/papers/upload", files={"file": ("x.pdf", b"not a pdf", "application/pdf")}).status_code == 400


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


def test_rag_with_citations(client):
    r = client.post("/query", json={"question": "Which dataset was used with wav2vec 2.0 and what accuracy was achieved?"}).json()
    assert r["citations"] and "[" in r["answer"]
    assert r["unverified"] == 0
    out = client.post("/query", json={"question": "How does quantum chromodynamics affect protein folding in zebrafish?"}).json()
    assert out["answer"].startswith("Not found")


def test_compare(client):
    ids = [p["id"] for p in client.get("/papers").json()["papers"]][:3]
    r = client.post("/compare", json={"paper_ids": ids}).json()
    assert len(r["table"]["rows"]) == 3 and r["narrative"]


def test_gap_detection_finds_planted_gap(client):
    r = client.post("/research-gaps", json={"max_gaps": 15}).json()
    assert r["gaps"], r
    top = [(g["method"], g["dataset"]) for g in r["gaps"][:5]]
    print("TOP GAPS:", top)
    hit = next((g for g in r["gaps"] if g["method"] == "Transformer" and g["dataset"] == "PC-GITA"), None)
    assert hit is not None, top
    assert hit["tier"] == "evidence_backed"
    assert sum(1 for e in hit["evidence"] if e["type"] == "future_work") >= 2
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
    other = TestClient(app)
    r = other.post("/auth/register", json={"name": "O", "email": "o@example.com", "password": "password123"})
    other.headers["Authorization"] = f"Bearer {r.json()['token']}"
    pid = client.get("/papers").json()["papers"][0]["id"]
    assert other.get(f"/papers/{pid}").status_code == 404
    assert other.get("/papers").json()["papers"] == []
