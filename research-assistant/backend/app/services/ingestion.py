"""Ingestion orchestration: parse -> chunk -> embed -> store (Postgres + vector DB) -> extract -> citations."""
import logging
import re
import time
from datetime import datetime

from sqlalchemy.orm import Session

from ..db import Chunk, Citation, EmbeddingRef, Paper, PaperField, SessionLocal
from .embeddings import get_embedder
from .extraction import analyze_paper, get_fields, set_authors
from .pdf_processing import PDFError, process_pdf
from .vectorstore import get_store

log = logging.getLogger("ingest")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()


def ingest_paper(paper_id: int):
    """Runs in a background task (own DB session)."""
    db: Session = SessionLocal()
    try:
        paper = db.get(Paper, paper_id)
        if not paper or not paper.file_path:
            return
        t0 = time.time()
        data = process_pdf(paper.file_path)
        t_parse = time.time()
        meta = data["meta"]
        paper.title = paper.title or meta.get("title") or paper.filename
        paper.year = paper.year or meta.get("year")
        paper.doi = paper.doi or meta.get("doi")
        paper.venue = paper.venue or meta.get("venue")
        paper.abstract = paper.abstract or meta.get("abstract", "")
        db.query(Chunk).filter(Chunk.paper_id == paper.id).delete()
        get_store().delete_paper(paper.id)
        rows = []
        for c in data["chunks"]:
            row = Chunk(paper_id=paper.id, section=c["section"], chunk_index=c["chunk_index"], text=c["text"],
                        page_number=c["page"], suspicious=c.get("suspicious", False))
            db.add(row); rows.append(row)
        db.flush()
        emb = get_embedder()
        vecs = emb.encode([r.text for r in rows])
        get_store().add([f"c{r.id}" for r in rows], vecs,
                        [{"paper_id": paper.id, "chunk_id": r.id, "section": r.section, "page": r.page_number} for r in rows])
        for r in rows:
            db.add(EmbeddingRef(chunk_id=r.id, vector_db_id=f"c{r.id}"))
        db.commit()
        t_embed = time.time()
        if meta.get("authors"):
            set_authors(db, paper, meta["authors"])
        analyze_paper(db, paper, data.get("references_text", ""))
        paper.status, paper.error, paper.processed_at = "ready", None, datetime.utcnow()
        db.commit()
        rebuild_citations(db, paper.upload_user_id)
        log.info("paper %s ready: parse %.1fs, embed+store %.1fs, extract %.1fs (%d chunks)", paper_id,
                 t_parse - t0, t_embed - t_parse, time.time() - t_embed, len(rows))
    except (PDFError, Exception) as e:  # noqa
        log.exception("ingest failed for %s", paper_id)
        db.rollback()
        p = db.get(Paper, paper_id)
        if p:
            p.status, p.error = "failed", str(e)[:500]
            db.commit()
    finally:
        db.close()


def reanalyze(db: Session, paper: Paper) -> dict:
    refs = get_fields(db, paper.id).get("references", {}).get("value") or []
    return analyze_paper(db, paper, "\n".join(refs))


def rebuild_citations(db: Session, user_id: int):
    """Match each paper's reference list against titles of other papers in the same library."""
    papers = db.query(Paper).filter(Paper.upload_user_id == user_id, Paper.status.in_(["ready", "metadata_only"])).all()
    ids = [p.id for p in papers]
    db.query(Citation).filter(Citation.citing_paper_id.in_(ids)).delete(synchronize_session=False)
    for p in papers:
        refs = db.query(PaperField).filter_by(paper_id=p.id, field_name="references").first()
        if not refs:
            continue
        import json
        blob = _norm(" ".join(json.loads(refs.field_value)))
        for q in papers:
            t = _norm(q.title)
            if q.id != p.id and len(t.split()) >= 4 and t in blob:
                db.add(Citation(citing_paper_id=p.id, cited_paper_id=q.id))
    db.commit()
