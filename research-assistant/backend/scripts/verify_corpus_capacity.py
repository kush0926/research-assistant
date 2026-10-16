"""Exercise a distinct-PDF corpus with BGE/Chroma and an isolated PostgreSQL schema.

Retains the isolated data for a second-process --phase verify check. Credentials
come only from POSTGRES_TEST_URL; neither credentials nor tokens enter reports.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data_store/parkvoice_scale_corpus")
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--phase", choices=("ingest", "verify"), default="ingest")
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import make_url
    if not os.getenv("POSTGRES_TEST_URL"):
        parser.error("Set POSTGRES_TEST_URL to the disposable ra_readiness_test database")
    base = make_url(os.environ["POSTGRES_TEST_URL"])
    if base.get_backend_name() != "postgresql" or base.database != "ra_readiness_test":
        parser.error("POSTGRES_TEST_URL must explicitly select the disposable ra_readiness_test database")
    args.state.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.state.exists():
        state = json.loads(args.state.read_text(encoding="utf-8"))
        report = json.loads(args.report.read_text(encoding="utf-8"))
    else:
        if args.phase != "ingest":
            parser.error("No ingestion state exists")
        schema = "ra_scale_" + uuid.uuid4().hex[:12]
        state = {"schema": schema, "data_directory": str(ROOT / "data_store" / schema), "papers": {}}
        report = {"status": "running", "requested_unique_papers": args.count, "papers": [],
                  "database": "PostgreSQL", "embeddings": "bge", "reranker": "bge", "vector_store": "Chroma",
                  "llm": "none; retrieval/storage capacity is independent of provider quotas",
                  "process_restart_verified": False}
        admin = create_engine(base)
        with admin.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        admin.dispose()
    if not re.fullmatch(r"ra_scale_[0-9a-f]{12}", state["schema"]):
        raise ValueError("Unexpected test schema")
    data = Path(state["data_directory"]).resolve()
    if data.parent != (ROOT / "data_store").resolve() or data.name != state["schema"]:
        raise ValueError("Capacity test data directory is outside its expected location")
    data.mkdir(parents=True, exist_ok=True)
    database_url = base.update_query_dict({"options": "-csearch_path=" + state["schema"]})
    os.environ.update(DATA_DIR=str(data), DATABASE_URL=database_url.render_as_string(hide_password=False),
        LLM_PROVIDER="none", EMBEDDING_BACKEND="bge", RERANKER_BACKEND="bge", VECTOR_BACKEND="chroma",
        GROBID_URL="http://localhost:8070", WARMUP="false", SYNC_INGEST="true", ENABLE_METADATA_LOOKUP="false",
        RATE_LIMIT_PER_MIN="100000", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

    def save():
        args.state.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    save()
    from fastapi.testclient import TestClient
    from app.main import app
    from app.db import Chunk, SessionLocal
    from app.services.embeddings import get_embedder
    from app.services.reranker import get_reranker
    from eval.ie_review import build_ie_review_draft
    try:
        started = time.perf_counter()
        get_embedder().encode(["Parkinson speech classification"])
        get_reranker().rerank("speech", [{"text": "Speech classification.", "score": 1.0}], 1)
        report[args.phase + "_model_startup_seconds"] = round(time.perf_counter() - started, 3)
        with TestClient(app) as client:
            credentials = {"email": "scale@example.test", "password": "Isolated-scale-check-2026"}
            response = client.post("/auth/login", json=credentials)
            if response.status_code == 401 and args.phase == "ingest":
                response = client.post("/auth/register", json={"name": "Capacity test", **credentials})
            response.raise_for_status()
            client.headers["Authorization"] = "Bearer " + response.json()["token"]
            entries = json.loads((args.corpus / "manifest.json").read_text(encoding="utf-8"))[:args.count]
            assert len(entries) == args.count, "The full unique corpus must be ready before this run"
            assert len({re.sub(r"v\d+$", "", e["external_id"]) for e in entries}) == args.count
            assert len({e["sha256"] for e in entries}) == args.count
            if args.phase == "ingest":
                for entry in entries:
                    filename = entry["filename"]
                    if Path(filename).name != filename:
                        raise ValueError("Unsafe manifest filename")
                    if filename in state["papers"]:
                        detail = client.get(f"/papers/{state['papers'][filename]}").json()
                        assert detail["status"] == "ready", "A resumed paper is not ready"
                        continue
                    content = (args.corpus / filename).read_bytes()
                    assert hashlib.sha256(content).hexdigest() == entry["sha256"]
                    started = time.perf_counter()
                    response = client.post("/papers/upload", files={"file": (filename, content, "application/pdf")})
                    response.raise_for_status()
                    pid = response.json()["paper_id"]
                    detail = client.get(f"/papers/{pid}").json()
                    assert detail["status"] == "ready", detail.get("error")
                    state["papers"][filename] = pid
                    row = {"filename": filename, "paper_id": pid, "pages": entry["pages"],
                           "chunks": detail["n_chunks"], "seconds": round(time.perf_counter() - started, 3)}
                    report["papers"].append(row)
                    save()
                    print(f"[{len(state['papers'])}/{args.count}] ready {filename}: {row['seconds']}s", flush=True)
            papers = client.get("/papers").json()["papers"]
            ids = list(state["papers"].values())
            if args.phase == "verify":
                # Refresh extracted fields with the current rules without redoing embeddings.
                for pid in ids:
                    response = client.post(f"/papers/{pid}/analyze")
                    response.raise_for_status()
                report["reanalysis_with_current_rules"] = len(ids)
            assert len(papers) == args.count and {p["id"] for p in papers} == set(ids)
            assert all(p["status"] == "ready" for p in papers)
            with SessionLocal() as db:
                report["stored_chunks"] = db.query(Chunk).filter(Chunk.paper_id.in_(ids)).count()
            queries = []
            for question in ["Which datasets are used for Parkinson speech classification?",
                             "What speech models and representations are evaluated?",
                             "What limitations and future research directions are reported?"]:
                started = time.perf_counter()
                response = client.post("/query", json={"question": question, "paper_ids": ids})
                response.raise_for_status()
                answer = response.json()
                assert answer["citations"], "Supported full-corpus query returned no sources"
                for citation in answer["citations"]:
                    resolved = client.get(f"/chunks/{citation['chunk_id']}")
                    resolved.raise_for_status()
                    assert resolved.json()["paper_id"] == citation["paper_id"]
                queries.append({"question": question, "seconds": round(time.perf_counter() - started, 3),
                                "citations": len(answer["citations"]), "automated_unverified_claims": answer["unverified"]})
            report[args.phase + "_queries"] = queries
            response = client.post("/compare", json={"paper_ids": ids[:5]})
            response.raise_for_status()
            assert len(response.json()["table"]["rows"]) == 5
            report["comparison_rows"] = 5
            response = client.get("/knowledge-graph")
            response.raise_for_status()
            report["graph_nodes"] = len(response.json()["nodes"])
            response = client.get("/recommendations", params={"paper_id": ids[0]})
            response.raise_for_status()
            report["recommendations"] = len(response.json()["recommendations"])
            assert report["recommendations"] > 0
            if args.phase == "ingest":
                response = client.post("/research-gaps", json={"max_gaps": 8})
                response.raise_for_status()
                report["gap_candidates"] = len(response.json()["gaps"])
                for paper in papers:
                    paper["filename"] = next(name for name, pid in state["papers"].items() if pid == paper["id"])
                draft = build_ie_review_draft(papers, lambda pid: client.get(f"/papers/{pid}").json(),
                                             lambda cid: client.get(f"/chunks/{cid}").json())
                draft["corpus_size"] = args.count
                args.report.with_name(args.report.stem + "_extraction.json").write_text(
                    json.dumps(draft, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                report["status"] = "ingestion_and_workflow_passed_restart_check_pending"
            else:
                response = client.post("/research-gaps", json={"max_gaps": 8})
                response.raise_for_status()
                gap_output = response.json()
                args.report.with_name(args.report.stem + "_gaps_current.json").write_text(
                    json.dumps(gap_output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                report["gap_candidates_current"] = len(gap_output["gaps"])
                for paper in papers:
                    paper["filename"] = next(name for name, pid in state["papers"].items() if pid == paper["id"])
                draft = build_ie_review_draft(papers, lambda pid: client.get(f"/papers/{pid}").json(),
                                             lambda cid: client.get(f"/chunks/{cid}").json())
                draft["corpus_size"] = args.count
                args.report.with_name(args.report.stem + "_extraction_current.json").write_text(
                    json.dumps(draft, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                report["process_restart_verified"] = True
                report["status"] = "passed"
            save()
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        save()
        raise
    print(json.dumps({"status": report["status"], "papers": len(state["papers"]), "report": str(args.report)}))


if __name__ == "__main__":
    main()
