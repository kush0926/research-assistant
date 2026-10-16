"""Live Groq smoke on two public PDFs, isolated from application data.

Loads the key from the chosen .env without logging it. Requests are conservatively
paced for the free plan; pacing time is recorded separately from provider latency.
This produces inputs for automated LLM assessment, not human quality scores.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT.parent / ".env")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data_store/parkvoice_arxiv_corpus")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding", choices=("bge", "hash"), default="bge")
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(args.env_file, override=True)
    if not os.getenv("GROQ_API_KEY"):
        parser.error("GROQ_API_KEY is missing; set it locally in --env-file")
    tmp = Path(tempfile.mkdtemp(prefix="ra-groq-live-"))
    os.environ.update(DATA_DIR=str(tmp), DATABASE_URL=f"sqlite:///{tmp / 'live.db'}",
                      LLM_PROVIDER="groq", SYNC_INGEST="true", WARMUP="false",
                      EMBEDDING_BACKEND=args.embedding,
                      RERANKER_BACKEND="bge" if args.embedding == "bge" else "lexical",
                      VECTOR_BACKEND="chroma", ENABLE_METADATA_LOOKUP="false",
                      RATE_LIMIT_PER_MIN="1000")
    # Caller may select a live GROBID service explicitly; otherwise use the local parser.
    os.environ["GROBID_URL"] = os.getenv("GROQ_TEST_GROBID_URL", "")
    from fastapi.testclient import TestClient
    import httpx
    from app.main import app
    from app.db import SessionLocal
    from app.config import settings
    from app.services import rag
    from app.services.embeddings import get_embedder
    from app.services.reranker import get_reranker
    from app.services.llm import llm
    from eval.rag_ablation import answer_basic_rag, review_record

    report = {"status": "running", "provider": llm.provider, "model": llm.model,
              "key_loaded": llm.available, "embedding": args.embedding,
              "reranker": settings.RERANKER_BACKEND, "grobid": bool(settings.GROBID_URL),
              "calls": [], "checks": {}, "quality_review": "automated_llm_assessment",
              "relevance_labels": "Smoke-test source choices; not independently reviewed benchmark labels"}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    original_complete = llm.complete
    reservations = []
    stage = "startup"

    def paced_complete(system, user, max_tokens=1200, temperature=0.0):
        # Conservative estimate; the account's actual limits remain authoritative.
        reasoning_headroom = 1024 if llm.model.startswith("openai/gpt-oss-") else 0
        estimate = int((len(system) + len(user)) / 3) + max_tokens + reasoning_headroom
        if estimate > 7500:
            raise RuntimeError("Test prompt exceeds the conservative free-tier request budget")
        wait_total = 0.0
        while reservations:
            now = time.monotonic()
            reservations[:] = [(t, n) for t, n in reservations if now - t < 61]
            if sum(n for _, n in reservations) + estimate <= 7500 or not reservations:
                break
            delay = max(0.1, 61 - (now - reservations[0][0]))
            print(f"Pacing {stage}: waiting {delay:.1f}s for the free-tier token window", flush=True)
            time.sleep(delay)
            wait_total += delay
        reservations.append((time.monotonic(), estimate))
        started = time.perf_counter()
        call = {"stage": stage, "estimated_token_reservation": estimate,
                "pacing_seconds": round(wait_total, 3)}
        try:
            result = original_complete(system, user, max_tokens, temperature)
            call.update(status="passed", output_chars=len(result))
            return result
        except Exception as error:
            call.update(status="failed", error_type=type(error).__name__)
            if isinstance(error, httpx.HTTPStatusError):
                call["http_status"] = error.response.status_code
            raise
        finally:
            call["provider_seconds"] = round(time.perf_counter() - started, 3)
            report["calls"].append(call)
            save()

    llm.complete = paced_complete
    try:
        started = time.perf_counter()
        get_embedder().encode(["Parkinson speech classification"])
        get_reranker().rerank("speech", [{"text": "Speech classification methods.", "score": 1.0}], 1)
        report["model_startup_seconds"] = round(time.perf_counter() - started, 3)
        files = [args.corpus / "arxiv_2009_01231v4.pdf", args.corpus / "arxiv_2105_14704v1.pdf"]
        if not all(path.is_file() for path in files):
            raise FileNotFoundError("Download the ParkVoice corpus before running this check")
        with TestClient(app) as client:
            response = client.post("/auth/register", json={"name": "Groq integration",
                "email": "groq-live@example.test", "password": "isolated-test-password"})
            response.raise_for_status()
            client.headers["Authorization"] = f"Bearer {response.json()['token']}"
            uid = response.json()["user"]["id"]
            ids, filenames = [], {}
            stage = "extraction"
            for path in files:
                response = client.post("/papers/upload", files={"file": (path.name, path.read_bytes(), "application/pdf")})
                response.raise_for_status()
                pid = response.json()["paper_id"]
                detail = client.get(f"/papers/{pid}").json()
                assert detail["status"] == "ready", detail.get("error")
                assert detail["fields"]["research_problem"]["method"] == "llm", "Extraction fell back"
                ids.append(pid)
                filenames[pid] = path.name
            report["checks"]["extraction"] = {"status": "passed", "papers": len(ids)}
            stage = "comparison"
            response = client.post("/compare", json={"paper_ids": ids})
            response.raise_for_status()
            compared = response.json()
            assert len(compared["table"]["rows"]) == 2
            assert any(c["stage"] == stage and c["status"] == "passed" for c in report["calls"])
            report["checks"][stage] = {"status": "passed", "rows": 2, "narrative": compared["narrative"]}
            stage = "review"
            response = client.post("/literature-review", json={"paper_ids": ids})
            response.raise_for_status()
            review = response.json()
            assert review.get("generation_mode", review["mode"]) == "llm", "Review generation fell back"
            assert "[P1]" in review["markdown"] and "[P2]" in review["markdown"]
            report["checks"][stage] = {"status": "passed", **review}
            question = "Which speech representations or models are used to classify Parkinson disease? Answer in at most 120 words."
            with SessionLocal() as db:
                stage = "basic_rag"
                basic = answer_basic_rag(db, question, ids, llm.complete)
                assert basic["citations"] and not basic["invalid_citation_markers"]
                report["checks"][stage] = {"status": "passed", "citations": len(basic["citations"])}
                stage = "proposed_rag"
                started = time.perf_counter()
                proposed = rag.answer_question(db, uid, question, ids)
                elapsed = time.perf_counter() - started
                assert proposed.get("generation_mode", proposed["mode"]) == "llm", "No live answer generation"
                assert all(claim["verified"] for claim in proposed["claims"]), "Delivered answer has unchecked claims"
                assert proposed["citations"], "No cited answer sources"
                for citation in proposed["citations"]:
                    source = client.get(f"/chunks/{citation['chunk_id']}")
                    source.raise_for_status()
                    assert source.json()["paper_id"] == citation["paper_id"]
                report["checks"][stage] = {"status": "passed", "mode": proposed["mode"],
                    "generation_mode": proposed.get("generation_mode", proposed["mode"]),
                    "omitted_unverified_claims": proposed.get("omitted_unverified_claims", 0),
                    "citations": len(proposed["citations"]), "automated_unverified_claims": proposed["unverified"],
                    "seconds_including_pacing": round(elapsed, 3), "answer": proposed["answer"]}
                proposed["contexts"] = rag.retrieve(db, question, ids)
                row = review_record(question, list(filenames.values()),
                                    {"baseline_3_basic_rag": basic, "proposed_rag": proposed}, filenames)
                report["rag_ablation_review"] = {"status": "live_outputs_ready_for_automated_assessment", "questions": [row]}
            assert not [call for call in report["calls"] if call["status"] != "passed"], "A provider call failed"
            report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        llm.complete = original_complete
        save()
        print(json.dumps({"status": report["status"], "output": str(args.output),
                          "provider_calls": len(report["calls"])}, indent=2), flush=True)


if __name__ == "__main__":
    main()
