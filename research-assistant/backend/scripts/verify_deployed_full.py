"""Verify the isolated full Docker stack using its existing browser test user.

Credentials remain in memory. Only public corpus files and public generated
outputs are written to the report. Existing matching ready papers are reused.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--postgres-port", type=int, default=55679)
    args = parser.parse_args()
    from dotenv import load_dotenv
    import httpx
    import jwt
    from sqlalchemy import create_engine, text

    load_dotenv(ROOT.parent / ".env", override=True)
    assert args.base_url in ("http://127.0.0.1:8001", "http://localhost:8001")
    old = create_engine(f"postgresql://ra:ra@127.0.0.1:{args.postgres_port}/ra_readiness_test")
    new = create_engine(f"postgresql://ra:ra@127.0.0.1:{args.postgres_port}/ra_groq_full_test")
    report = json.loads(args.output.read_text(encoding="utf-8")) if args.output.is_file() else {}
    report.update(status="running", quality_review="automated_llm_assessment")
    report.setdefault("checks", {})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stage = "reuse_isolated_test_user"

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def pause():
        print("Pacing the generator's free-plan token window: 61s", flush=True)
        time.sleep(61)

    try:
        with old.connect() as conn:
            users = conn.execute(text("SELECT id,name,email,password_hash,created_at FROM users "
                "WHERE name='Browser verification' AND email LIKE 'browser-check-%@example.test'")).mappings().all()
        assert len(users) == 1, "Expected the single existing isolated browser test user"
        user = dict(users[0])
        with new.begin() as conn:
            existing = conn.execute(text("SELECT id,email FROM users WHERE id=:id OR email=:email"), user).mappings().all()
            if existing:
                assert len(existing) == 1 and existing[0]["id"] == user["id"] and existing[0]["email"] == user["email"]
            else:
                conn.execute(text("INSERT INTO users (id,name,email,password_hash,created_at) "
                    "VALUES (:id,:name,:email,:password_hash,:created_at)"), user)
                conn.execute(text("SELECT setval(pg_get_serial_sequence('users','id'),(SELECT max(id) FROM users))"))
        token = jwt.encode({"sub": str(user["id"]), "exp": datetime.now(timezone.utc) + timedelta(hours=12)},
                           os.getenv("JWT_SECRET", "change-me-in-production"), algorithm="HS256")
        with httpx.Client(base_url=args.base_url, headers={"Authorization": f"Bearer {token}"}, timeout=240) as client:
            health = client.get("/health").raise_for_status().json()
            assert health["llm"] == "groq" and health["embeddings"] == "bge" and health["reranker"] == "cross-encoder"
            assert health["vector_db"] == "ChromaStore" and health["grobid"]
            assert client.get("/auth/me").raise_for_status().json()["name"] == "Browser verification"
            report["health"] = health
            ids, details = [], []
            stage = "public_pdf_ingestion"
            with new.connect() as conn:
                existing_papers = conn.execute(text("SELECT id,filename,status FROM papers WHERE upload_user_id=:id"),
                                               {"id": user["id"]}).mappings().all()
            filenames = ["arxiv_2005_14647v2.pdf", "arxiv_2009_01231v4.pdf", "arxiv_2105_14704v1.pdf"]
            for filename in filenames:
                matching = [p for p in existing_papers if p["filename"] == filename]
                assert len(matching) <= 1
                if matching:
                    pid = matching[0]["id"]
                else:
                    if ids:
                        pause()
                    path = ROOT / "data_store/parkvoice_arxiv_corpus" / filename
                    upload = client.post("/papers/upload", files={"file": (filename, path.read_bytes(), "application/pdf")})
                    pid = upload.raise_for_status().json()["paper_id"]
                started = time.monotonic()
                while True:
                    detail = client.get(f"/papers/{pid}").raise_for_status().json()
                    if detail["status"] in ("ready", "failed", "error"):
                        break
                    assert time.monotonic() - started < 240, "Ingestion did not finish"
                    time.sleep(2)
                assert detail["status"] == "ready", "Public PDF ingestion failed"
                assert detail["fields"]["research_problem"]["method"] == "llm", "Live extraction fell back"
                detail["test_source_filename"] = filename
                ids.append(pid)
                details.append(detail)
                report["papers"] = details
                save()
                print(f"Ready: {filename}; {len(ids)}/3", flush=True)
            report["checks"]["ingestion"] = {"status": "passed", "ready_papers": len(ids)}
            save()

            stage = "source_verified_query"
            if stage not in report["checks"]:
                pause()
                started = time.perf_counter()
                answer = client.post("/query", json={"question": "Which speech representations or models are used to classify Parkinson disease? Answer in at most 120 words.", "paper_ids": ids}).raise_for_status().json()
                assert answer.get("generation_mode", answer["mode"]) == "llm", "Generation fell back"
                assert answer["citations"] and answer["claims"] and all(c["verified"] for c in answer["claims"])
                assert answer["unverified"] == 0
                # Every displayed marker belongs to a substantive claim, never an orphan fragment.
                assert not re.search(r"(?m)^\s*(?:\[\d+\]\s*)+$", answer["answer"])
                for citation in answer["citations"]:
                    source = client.get(f"/chunks/{citation['chunk_id']}").raise_for_status().json()
                    assert source["paper_id"] == citation["paper_id"] and source["paper_id"] in ids
                    citation["resolved_passage"] = source
                report["checks"][stage] = {"status": "passed", "seconds": round(time.perf_counter()-started, 3), **answer}
                save()
            stage = "comparison"
            if stage not in report["checks"]:
                pause()
                compared = client.post("/compare", json={"paper_ids": ids}).raise_for_status().json()
                assert compared["generation_mode"] == "llm", "Comparison generation fell back"
                assert compared.get("verification") and compared["verification"]["status"] in ("passed", "fallback")
                assert len(compared["table"]["rows"]) == len(ids)
                count = 0
                for row in compared["table"]["rows"]:
                    for dimension in compared["table"]["dimensions"]:
                        cell = row[dimension]
                        for cid in cell.get("chunk_ids", []):
                            source = client.get(f"/chunks/{cid}").raise_for_status().json()
                            assert source["paper_id"] == row["paper_id"]
                            count += 1
                report["checks"][stage] = {"status": "passed", "resolved_cell_sources": count, **compared}
                save()
            stage = "literature_review"
            if stage not in report["checks"]:
                pause()
                review = client.post("/literature-review", json={"paper_ids": ids}).raise_for_status().json()
                assert review.get("generation_mode", review["mode"]) == "llm", "Review generation fell back"
                assert review.get("verification") and review["verification"]["status"] in ("passed", "fallback")
                assert all(f"[P{i}]" in review["markdown"] for i in range(1, len(ids)+1))
                report["checks"][stage] = {"status": "passed", **review}
                save()
            stage = "graph_and_recommendations"
            graph = client.get("/knowledge-graph").raise_for_status().json()
            recommendations = client.get("/recommendations", params={"paper_id": ids[0]}).raise_for_status().json()
            assert graph["nodes"] and graph["edges"] and recommendations["recommendations"]
            report["checks"][stage] = {"status": "passed", "nodes": len(graph["nodes"]), "edges": len(graph["edges"]),
                                      "recommendations": len(recommendations["recommendations"])}
            report["status"] = "passed"
    except Exception as error:
        # Never render SQL parameter dumps or credentials on failure.
        report.update(status="failed", failed_stage=stage, error_type=type(error).__name__)
        if hasattr(error, "response"):
            report["http_status"] = error.response.status_code
        print(json.dumps({"status": "failed", "stage": stage, "error_type": type(error).__name__}), flush=True)
    finally:
        save()
        old.dispose()
        new.dispose()
    print(json.dumps({"status": report["status"], "report": str(args.output)}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
