"""FastAPI application — endpoints follow Part 22 of the blueprint (+ auth, library, graph, review, agent)."""
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .config import UPLOAD_DIR, settings
from .db import (Author, Chunk, Paper, PaperAuthor, ResearchGap, GapEvidence, User, get_db, init_db)
from .security import create_token, current_user, hash_password, rate_limit, safe_filename, verify_password
from .services import agent, comparison, gaps, graph, litreview, paper_search, rag, recommend
from .services.extraction import get_fields, set_authors
from .services.ingestion import ingest_paper, reanalyze, rebuild_citations
from .services.llm import llm
from .services.pdf_processing import PDFError, validate_pdf_bytes
from .services.vectorstore import get_store

logging.basicConfig(level=logging.INFO)


def _warmup():
    """Load embedding model, reranker and vector store once, so the first upload/question is not the slow one."""
    log = logging.getLogger("warmup")
    try:
        from .services.embeddings import get_embedder
        from .services.reranker import get_reranker
        get_store(); get_embedder().encode(["warm up"]); get_reranker()
        log.info("Warm-up complete: models loaded and ready")
    except Exception:
        log.exception("Warm-up failed (the first request will load models lazily)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    if settings.WARMUP and not settings.SYNC_INGEST:
        import threading
        threading.Thread(target=_warmup, daemon=True).start()
    yield


app = FastAPI(title="AI Research Assistant", version="1.0.0", lifespan=lifespan,
              description="Evidence-grounded literature analysis and research-gap detection")
app.add_middleware(CORSMiddleware, allow_origins=settings.CORS_ORIGINS, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


# ------------------------------ helpers ------------------------------
def own_paper(db: Session, user: User, paper_id: int) -> Paper:
    p = db.get(Paper, paper_id)
    if not p or p.upload_user_id != user.id:
        raise HTTPException(404, "Paper not found")
    return p


def paper_summary(db: Session, p: Paper) -> dict:
    return {"id": p.id, "title": p.title or p.filename, "year": p.year, "venue": p.venue, "doi": p.doi,
            "status": p.status, "error": p.error, "source": p.source, "citation_count": p.citation_count,
            "n_chunks": db.query(Chunk).filter_by(paper_id=p.id).count(),
            "authors": [a.name for a in db.query(Author).join(PaperAuthor, PaperAuthor.author_id == Author.id)
                        .filter(PaperAuthor.paper_id == p.id).order_by(PaperAuthor.author_order)]}


# ------------------------------ system ------------------------------
@app.get("/health")
def health():
    from .services.embeddings import get_embedder
    from .services.reranker import get_reranker
    return {"status": "ok", "llm": llm.provider if llm.available else "none (extractive/offline mode)",
            "embeddings": get_embedder().backend, "reranker": "cross-encoder" if get_reranker().model else "lexical",
            "vector_db": type(get_store()).__name__, "grobid": bool(settings.GROBID_URL)}


# ------------------------------ auth ------------------------------
class RegisterIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    email: str = Field(min_length=5, max_length=200, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(min_length=8, max_length=200)


class LoginIn(BaseModel):
    email: str
    password: str


@app.post("/auth/register")
def register(body: RegisterIn, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == body.email.lower()).first():
        raise HTTPException(409, "Email already registered")
    u = User(name=body.name, email=body.email.lower(), password_hash=hash_password(body.password))
    db.add(u); db.commit()
    return {"token": create_token(u.id), "user": {"id": u.id, "name": u.name, "email": u.email}}


@app.post("/auth/login")
def login(body: LoginIn, db: Session = Depends(get_db)):
    u = db.query(User).filter(User.email == body.email.lower()).first()
    if not u or not verify_password(body.password, u.password_hash):
        raise HTTPException(401, "Invalid email or password")
    return {"token": create_token(u.id), "user": {"id": u.id, "name": u.name, "email": u.email}}


@app.get("/auth/me")
def me(user: User = Depends(current_user)):
    return {"id": user.id, "name": user.name, "email": user.email}


# ------------------------------ papers ------------------------------
@app.post("/papers/upload")
async def upload_paper(background: BackgroundTasks, file: UploadFile = File(...),
                       user: User = Depends(current_user), db: Session = Depends(get_db)):
    data = await file.read()
    try:
        validate_pdf_bytes(data, file.content_type)
    except PDFError as e:
        raise HTTPException(400, str(e))
    name = safe_filename(file.filename)
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{name}"
    path.write_bytes(data)
    p = Paper(filename=name, file_path=str(path), upload_user_id=user.id, status="processing", source="upload")
    db.add(p); db.commit()
    if settings.SYNC_INGEST:
        ingest_paper(p.id)
    else:
        background.add_task(ingest_paper, p.id)
    return {"paper_id": p.id, "status": "processing"}


@app.get("/papers/search")
def search_papers(q: str = Query(min_length=2), source: str = "semanticscholar", limit: int = Query(10, le=30),
                  user: User = Depends(current_user)):
    try:
        return {"results": paper_search.search(q, source, limit)}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"{source} search failed: {e}")


class ImportIn(BaseModel):
    source: str
    external_id: str | None = None
    title: str
    abstract: str = ""
    year: int | None = None
    authors: list[str] = []
    pdf_url: str | None = None
    doi: str | None = None
    citation_count: int | None = None
    url: str | None = None


@app.post("/papers/import")
def import_paper(body: ImportIn, background: BackgroundTasks, user: User = Depends(current_user),
                 db: Session = Depends(get_db)):
    """Open-access PDFs are downloaded and fully processed. Otherwise only metadata/abstract is stored (copyright policy)."""
    p = Paper(title=body.title, abstract=body.abstract, year=body.year, doi=body.doi, source=body.source,
              external_id=body.external_id, citation_count=body.citation_count or 0, source_url=body.url or body.pdf_url,
              upload_user_id=user.id, status="metadata_only")
    db.add(p); db.commit()
    set_authors(db, p, body.authors)
    if body.pdf_url:
        try:
            data = paper_search.download_pdf(body.pdf_url)
            validate_pdf_bytes(data)
            path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_filename(body.title)[:60]}.pdf"
            path.write_bytes(data)
            p.file_path, p.filename, p.status = str(path), path.name, "processing"
            db.commit()
            if settings.SYNC_INGEST:
                ingest_paper(p.id)
            else:
                background.add_task(ingest_paper, p.id)
        except Exception as e:
            p.error = f"PDF not imported ({e}); abstract/metadata only"; db.commit()
    return paper_summary(db, p)


@app.get("/papers")
def list_papers(user: User = Depends(current_user), db: Session = Depends(get_db)):
    ps = db.query(Paper).filter(Paper.upload_user_id == user.id).order_by(Paper.id.desc()).all()
    return {"papers": [paper_summary(db, p) for p in ps]}


@app.get("/papers/{paper_id}")
def get_paper(paper_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    p = own_paper(db, user, paper_id)
    return {**paper_summary(db, p), "abstract": p.abstract,
            "fields": get_fields(db, p.id)}


@app.post("/papers/{paper_id}/analyze")
def analyze(paper_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    p = own_paper(db, user, paper_id)
    if p.status != "ready":
        raise HTTPException(409, f"Paper is {p.status}")
    return {"paper_id": p.id, "extracted_fields": reanalyze(db, p)}


@app.delete("/papers/{paper_id}")
def delete_paper(paper_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    p = own_paper(db, user, paper_id)
    get_store().delete_paper(p.id)
    import os
    if p.file_path and os.path.exists(p.file_path):
        os.remove(p.file_path)
    db.delete(p); db.commit()
    rebuild_citations(db, user.id)
    return {"deleted": paper_id}


@app.get("/chunks/{chunk_id}")
def get_chunk(chunk_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    c = db.get(Chunk, chunk_id)
    if not c:
        raise HTTPException(404, "Not found")
    p = own_paper(db, user, c.paper_id)
    return {"chunk_id": c.id, "paper_id": p.id, "paper_title": p.title, "section": c.section, "page": c.page_number,
            "text": c.text, "flagged_suspicious": c.suspicious}


# ------------------------------ RAG ------------------------------
class QueryIn(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    paper_ids: list[int] | None = None
    sections: list[str] | None = None


@app.post("/query")
def query(body: QueryIn, user: User = Depends(rate_limit), db: Session = Depends(get_db)):
    return rag.answer_question(db, user.id, body.question, body.paper_ids, body.sections)


# ------------------------------ comparison ------------------------------
class CompareIn(BaseModel):
    paper_ids: list[int] = Field(min_length=2, max_length=5)


@app.post("/compare")
def compare(body: CompareIn, user: User = Depends(rate_limit), db: Session = Depends(get_db)):
    papers = [own_paper(db, user, i) for i in body.paper_ids]
    if any(p.status != "ready" for p in papers):
        raise HTTPException(409, "All papers must be processed (status 'ready')")
    return comparison.compare(db, papers)


# ------------------------------ research gaps ------------------------------
class GapIn(BaseModel):
    topic: str | None = None
    paper_ids: list[int] | None = None
    max_gaps: int = Field(12, ge=1, le=40)
    weights: dict[str, float] | None = None


@app.post("/research-gaps")
def research_gaps(body: GapIn, user: User = Depends(rate_limit), db: Session = Depends(get_db)):
    return gaps.detect_gaps(db, user.id, body.paper_ids, body.topic, body.weights, body.max_gaps)


@app.get("/research-gaps/{gap_id}")
def get_gap(gap_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    g = db.get(ResearchGap, gap_id)
    if not g or g.user_id != user.id:
        raise HTTPException(404, "Gap not found")
    ev = db.query(GapEvidence).filter_by(gap_id=g.id).all()
    return {"id": g.id, "description": g.description, "score": g.gap_score, "tier": g.tier, "type": g.gap_type,
            "evidence": [{"type": e.evidence_type, "paper_id": e.source_paper_id, "chunk_id": e.source_chunk_id, "note": e.note} for e in ev]}


# ------------------------------ discovery ------------------------------
@app.get("/recommendations")
def recommendations(paper_id: int | None = None, topic: str | None = None, gap_id: int | None = None,
                    user: User = Depends(current_user), db: Session = Depends(get_db)):
    return {"recommendations": recommend.recommend(db, user.id, paper_id, topic, gap_id)}


@app.get("/knowledge-graph")
def knowledge_graph(paper_id: int | None = None, topic: str | None = None, hops: int = Query(2, ge=1, le=3),
                    user: User = Depends(current_user), db: Session = Depends(get_db)):
    return graph.graph_view(db, user.id, paper_id, topic, hops)


class ReviewIn(BaseModel):
    paper_ids: list[int] = Field(min_length=1, max_length=30)


@app.post("/literature-review")
def literature_review(body: ReviewIn, user: User = Depends(rate_limit), db: Session = Depends(get_db)):
    papers = [own_paper(db, user, i) for i in body.paper_ids]
    return litreview.generate(db, [p for p in papers if p.status == "ready"], user.id)


class AgentIn(BaseModel):
    request: str = Field(min_length=3, max_length=1000)
    paper_ids: list[int] | None = None


@app.post("/agent")
def run_agent(body: AgentIn, user: User = Depends(rate_limit), db: Session = Depends(get_db)):
    return agent.run(db, user.id, body.request, body.paper_ids)
