"""Representative 1,000-record storage/retrieval load, using a 100-real-paper corpus.

Replicas are explicitly tagged, retain realistic text/vector shapes, and are never
used as independent papers for research-gap quality assessment. Embeddings are
reused; this does not measure 1,000 unique-PDF ingestion or semantic diversity.
Run expand and verify as separate processes, then remove only tagged replicas.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPLICA_SOURCE = "capacity_load_replica"


def memory_snapshot():
    if os.name != "nt":
        return {"scope": "application process only", "available": False}
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ["PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage"]]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    function = ctypes.windll.psapi.GetProcessMemoryInfo
    function.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not function(handle, ctypes.byref(counters), counters.cb):
        return {"scope": "application process only", "available": False}
    return {"scope": "application process only; excludes PostgreSQL and GROBID containers",
            "working_set_mb": round(counters.WorkingSetSize / 1024 ** 2, 1),
            "peak_working_set_mb": round(counters.PeakWorkingSetSize / 1024 ** 2, 1)}


def remap_chunks(value, mapping):
    if isinstance(value, dict):
        return {key: mapping.get(item, item) if key == "chunk_id" and (item is None or isinstance(item, int))
                else remap_chunks(item, mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [remap_chunks(item, mapping) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--phase", choices=("expand", "verify"), required=True)
    parser.add_argument("--target", type=int, default=1000)
    args = parser.parse_args()
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import make_url
    if not os.getenv("POSTGRES_TEST_URL"):
        parser.error("POSTGRES_TEST_URL is required")
    base = make_url(os.environ["POSTGRES_TEST_URL"])
    if base.get_backend_name() != "postgresql" or base.database != "ra_readiness_test":
        parser.error("Use only the disposable ra_readiness_test database")
    state = json.loads(args.state.read_text(encoding="utf-8"))
    schema, data = state["schema"], Path(state["data_directory"]).resolve()
    if not re.fullmatch(r"ra_scale_[0-9a-f]{12}", schema) or data.parent != (ROOT / "data_store").resolve() or data.name != schema:
        raise ValueError("Unexpected isolated capacity schema or directory")
    database = base.update_query_dict({"options": "-csearch_path=" + schema})
    os.environ.update(DATA_DIR=str(data), DATABASE_URL=database.render_as_string(hide_password=False),
        LLM_PROVIDER="none", EMBEDDING_BACKEND="bge", RERANKER_BACKEND="bge", VECTOR_BACKEND="chroma",
        GROBID_URL="http://localhost:8070", WARMUP="false", SYNC_INGEST="true", ENABLE_METADATA_LOOKUP="false",
        RATE_LIMIT_PER_MIN="100000", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    from fastapi.testclient import TestClient
    from app.main import app
    from app.db import (Chunk, EmbeddingRef, Paper, PaperAuthor, PaperDataset, PaperField, PaperMethod,
                        PaperProblem, PaperTopic, SessionLocal)
    from app.services.embeddings import get_embedder
    from app.services.reranker import get_reranker
    from app.services.vectorstore import get_store
    source_ids = list(state["papers"].values())
    if len(source_ids) != 100 or args.target < len(source_ids):
        raise ValueError("This run requires the verified 100-paper source corpus")
    report = json.loads(args.report.read_text(encoding="utf-8")) if args.report.exists() else {
        "status": "running", "source_distinct_papers": len(source_ids), "target_paper_records": args.target,
        "embedding_reuse": True, "workload": "100 distinct real sources plus explicitly tagged load replicas",
        "scope": "Representative storage, SQL filtering and BGE/Chroma retrieval volume; not a diverse 1000-source quality or ingestion benchmark"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    save = lambda: args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        store = get_store()
        with TestClient(app) as client:
            response = client.post("/auth/login", json={"email": "scale@example.test", "password": "Isolated-scale-check-2026"})
            response.raise_for_status()
            client.headers["Authorization"] = "Bearer " + response.json()["token"]
            with SessionLocal() as db:
                sources = db.query(Paper).filter(Paper.id.in_(source_ids)).order_by(Paper.id).all()
                assert len(sources) == 100 and len({p.upload_user_id for p in sources}) == 1
                user_id = sources[0].upload_user_id
                if args.phase == "expand":
                    started = time.perf_counter()
                    chunks_by_paper, fields_by_paper = {}, {}
                    for pid in source_ids:
                        chunks_by_paper[pid] = db.query(Chunk).filter_by(paper_id=pid).order_by(Chunk.chunk_index).all()
                        fields_by_paper[pid] = db.query(PaperField).filter_by(paper_id=pid).all()
                    links = {}
                    for model, column_names in [(PaperMethod, ("method_id", "role")), (PaperDataset, ("dataset_id", "role")),
                                               (PaperProblem, ("problem_id",)), (PaperTopic, ("topic_id", "confidence")),
                                               (PaperAuthor, ("author_id", "author_order"))]:
                        rows = db.query(model).filter(model.paper_id.in_(source_ids)).all()
                        links[model] = (column_names, {pid: [row for row in rows if row.paper_id == pid] for pid in source_ids})
                    existing = db.query(Paper).filter(Paper.upload_user_id == user_id, Paper.source == REPLICA_SOURCE).count()
                    vector_cache = {}
                    for index in range(existing, args.target - len(source_ids)):
                        source = sources[index % len(sources)]
                        old_chunks = chunks_by_paper[source.id]
                        if source.id not in vector_cache:
                            vectors = store.col.get(ids=[f"c{c.id}" for c in old_chunks], include=["embeddings"])
                            vector_cache[source.id] = dict(zip(vectors["ids"], vectors["embeddings"]))
                        replica = Paper(title=f"{source.title} [capacity replica {index + 1}]", year=source.year,
                            venue=source.venue, abstract=source.abstract, source=REPLICA_SOURCE,
                            external_id=f"capacity-{index + 1}", status="ready", upload_user_id=user_id)
                        db.add(replica); db.flush()
                        new_chunks = [Chunk(paper_id=replica.id, section=c.section, chunk_index=c.chunk_index,
                                            text=c.text, page_number=c.page_number, suspicious=c.suspicious) for c in old_chunks]
                        db.add_all(new_chunks); db.flush()
                        mapping = {old.id: new.id for old, new in zip(old_chunks, new_chunks)}
                        for field in fields_by_paper[source.id]:
                            value = remap_chunks(json.loads(field.field_value), mapping)
                            db.add(PaperField(paper_id=replica.id, field_name=field.field_name, field_value=json.dumps(value),
                                source_chunk_id=mapping.get(field.source_chunk_id), method=field.method))
                        for model, (names, grouped) in links.items():
                            db.add_all([model(paper_id=replica.id, **{name: getattr(link, name) for name in names})
                                        for link in grouped[source.id]])
                        new_ids = [f"c{c.id}" for c in new_chunks]
                        store.add(new_ids, [vector_cache[source.id][f"c{c.id}"] for c in old_chunks],
                                  [{"paper_id": replica.id, "chunk_id": c.id, "section": c.section, "page": c.page_number} for c in new_chunks])
                        db.add_all([EmbeddingRef(chunk_id=c.id, vector_db_id=f"c{c.id}") for c in new_chunks])
                        db.commit()
                        if (index + 1) % 25 == 0:
                            report["paper_records_created"] = len(source_ids) + index + 1
                            save()
                            print(f"Volume: {report['paper_records_created']}/{args.target} paper records", flush=True)
                    report["expansion_seconds"] = round(time.perf_counter() - started, 3)
                all_ids = [row[0] for row in db.query(Paper.id).filter(Paper.upload_user_id == user_id).all()]
                assert len(all_ids) == args.target
                report["stored_paper_records"] = len(all_ids)
                report["stored_chunks"] = db.query(Chunk).filter(Chunk.paper_id.in_(all_ids)).count()
                assert store.col.count() == report["stored_chunks"]
            started = time.perf_counter()
            get_embedder().encode(["Parkinson speech classification"])
            get_reranker().rerank("speech", [{"text": "Speech classification.", "score": 1.0}], 1)
            report[args.phase + "_model_startup_seconds"] = round(time.perf_counter() - started, 3)
            queries = []
            for question, scope in [("Which datasets are used for Parkinson speech classification?", all_ids),
                                    ("What speech models are evaluated?", all_ids),
                                    ("Which datasets and speech models are evaluated?", source_ids[:5])]:
                started = time.perf_counter()
                response = client.post("/query", json={"question": question, "paper_ids": scope})
                response.raise_for_status()
                result = response.json()
                assert result["citations"] and all(c["paper_id"] in scope for c in result["citations"])
                assert all(c["verified"] for c in result["claims"])
                for citation in result["citations"]:
                    resolved = client.get(f"/chunks/{citation['chunk_id']}")
                    resolved.raise_for_status()
                    assert resolved.json()["paper_id"] == citation["paper_id"]
                queries.append({"scope_size": len(scope), "seconds": round(time.perf_counter() - started, 3),
                                "citations": len(result["citations"]), "unverified": result["unverified"]})
            report[args.phase + "_queries"] = queries
            report[args.phase + "_memory"] = memory_snapshot()
            if args.phase == "verify":
                # Remove only explicitly tagged replicas, restoring the original real corpus.
                with SessionLocal() as db:
                    replicas = [row[0] for row in db.query(Paper.id).filter(Paper.upload_user_id == user_id,
                                                                 Paper.source == REPLICA_SOURCE).all()]
                    assert not set(replicas) & set(source_ids)
                    store.col.delete(where={"paper_id": {"$in": replicas}})
                    db.query(Paper).filter(Paper.upload_user_id == user_id, Paper.source == REPLICA_SOURCE).delete(synchronize_session=False)
                    db.commit()
                    assert db.query(Paper).filter(Paper.upload_user_id == user_id).count() == len(source_ids)
                    expected_chunks = db.query(Chunk).filter(Chunk.paper_id.in_(source_ids)).count()
                    assert store.col.count() == expected_chunks
                    report["replicas_removed"] = len(replicas)
                    report["retained_real_papers"] = len(source_ids)
                    report["retained_chunks"] = expected_chunks
                report["fresh_process_verified"] = True
                report["status"] = "passed"
            else:
                report["status"] = "expanded_and_queried_restart_check_pending"
            save()
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        save()
        raise


if __name__ == "__main__":
    main()
