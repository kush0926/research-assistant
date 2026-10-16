"""Verify the API across separate processes in an isolated PostgreSQL schema.

Set POSTGRES_TEST_URL to a dedicated test database with permission to create schemas.
The runner drops only its uniquely named schema and uses a temporary vector store.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

BACKEND_DIR = Path(__file__).resolve().parent.parent
EMAIL = "postgres-check@example.test"
PASSWORD = "postgres-check-password"


def _worker(phase: str, state_file: Path, sample_dir: Path) -> int:
    schema = os.environ.get("RA_PG_INTEGRATION_SCHEMA", "")
    url = make_url(os.environ.get("DATABASE_URL", ""))
    if (not re.fullmatch(r"ra_it_[0-9a-f]{12}", schema) or url.get_backend_name() != "postgresql"
            or url.query.get("options") != f"-csearch_path={schema}"):
        raise RuntimeError("Integration workers require the parent's isolated PostgreSQL schema configuration.")

    sys.path.insert(0, str(BACKEND_DIR))
    from fastapi.testclient import TestClient
    from app.main import app
    from app.db import engine

    try:
        with TestClient(app) as client:
            endpoint = "/auth/register" if phase == "ingest" else "/auth/login"
            authenticated = client.post(endpoint, json={
                "name": "PostgreSQL Integration", "email": EMAIL, "password": PASSWORD,
            })
            assert authenticated.status_code == 200, authenticated.text
            headers = {"Authorization": f"Bearer {authenticated.json()['token']}"}
            if phase == "ingest":
                sample_pdfs = sorted(sample_dir.glob("paper_*.pdf"))[:3]
                assert len(sample_pdfs) == 3
                paper_ids = []
                for pdf in sample_pdfs:
                    response = client.post("/papers/upload", headers=headers,
                                           files={"file": (pdf.name, pdf.read_bytes(), "application/pdf")})
                    assert response.status_code == 200, response.text
                    paper_ids.append(response.json()["paper_id"])
                detail = client.get(f"/papers/{paper_ids[0]}", headers=headers)
                assert detail.status_code == 200, detail.text
                state = {"paper_ids": paper_ids, "question": detail.json()["abstract"]}
                state_file.write_text(json.dumps(state), encoding="utf-8")
            else:
                state = json.loads(state_file.read_text(encoding="utf-8"))

            listing = client.get("/papers", headers=headers)
            assert listing.status_code == 200, listing.text
            assert {paper["id"] for paper in listing.json()["papers"]} == set(state["paper_ids"]), listing.text
            assert all(paper["status"] == "ready" for paper in listing.json()["papers"]), listing.text
            comparison = client.post("/compare", headers=headers, json={"paper_ids": state["paper_ids"]})
            assert comparison.status_code == 200 and len(comparison.json()["table"]["rows"]) == 3, comparison.text
            answer = client.post("/query", headers=headers,
                                 json={"question": state["question"], "paper_ids": [state["paper_ids"][0]]})
            assert answer.status_code == 200 and answer.json()["citations"], answer.text
            for citation in answer.json()["citations"]:
                passage = client.get(f"/chunks/{citation['chunk_id']}", headers=headers)
                assert passage.status_code == 200, passage.text
            if phase == "resume":
                paper_id = state["paper_ids"][0]
                deleted = client.delete(f"/papers/{paper_id}", headers=headers)
                assert deleted.status_code == 200, deleted.text
                assert client.get(f"/papers/{paper_id}", headers=headers).status_code == 404
        return 0
    finally:
        engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("ingest", "resume"), help=argparse.SUPPRESS)
    parser.add_argument("--state", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--samples", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.phase:
        if not args.state or not args.samples:
            parser.error("Integration worker state and sample paths are required.")
        return _worker(args.phase, args.state, args.samples)

    test_url = os.environ.get("POSTGRES_TEST_URL", "").strip()
    if not test_url:
        print("POSTGRES_TEST_URL is required; provide a dedicated PostgreSQL test database URL.", file=sys.stderr)
        return 2
    try:
        url = make_url(test_url)
        if url.get_backend_name() != "postgresql":
            raise ValueError()
    except Exception:
        print("POSTGRES_TEST_URL must be a valid PostgreSQL URL.", file=sys.stderr)
        return 2

    schema = f"ra_it_{uuid.uuid4().hex[:12]}"
    admin_engine = create_engine(url, connect_args={"connect_timeout": 10}, pool_pre_ping=True)
    schema_created = False
    try:
        with admin_engine.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        schema_created = True
        child_url = url.set(query={**url.query, "options": f"-csearch_path={schema}", "connect_timeout": "10"})
        with tempfile.TemporaryDirectory(prefix="ra-postgres-check-") as temp:
            temp_dir = Path(temp)
            sample_dir = temp_dir / "sample_papers"
            subprocess.run([sys.executable, str(BACKEND_DIR / "scripts" / "make_sample_papers.py"), str(sample_dir)],
                           check=True, capture_output=True, text=True, timeout=30)
            child_env = {**os.environ,
                         "DATABASE_URL": child_url.render_as_string(hide_password=False),
                         "RA_PG_INTEGRATION_SCHEMA": schema, "DATA_DIR": str(temp_dir / "data"),
                         "EMBEDDING_BACKEND": "hash", "RERANKER_BACKEND": "lexical", "VECTOR_BACKEND": "chroma",
                         "LLM_PROVIDER": "none", "GROBID_URL": "", "WARMUP": "false", "SYNC_INGEST": "true",
                         "ENABLE_METADATA_LOOKUP": "false", "RATE_LIMIT_PER_MIN": "1000"}
            for phase in ("ingest", "resume"):
                completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--phase", phase,
                                            "--state", str(temp_dir / "state.json"), "--samples", str(sample_dir)],
                                           cwd=BACKEND_DIR, env=child_env, capture_output=True, text=True, timeout=180)
                if completed.returncode:
                    print(completed.stdout, file=sys.stderr)
                    print(completed.stderr, file=sys.stderr)
                    return completed.returncode
        print("PostgreSQL API smoke passed: schema creation, registration, 3-paper ingestion, comparison, cited queries, process-restart persistence, and deletion.")
        return 0
    finally:
        if schema_created:
            with admin_engine.begin() as connection:
                connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        admin_engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
