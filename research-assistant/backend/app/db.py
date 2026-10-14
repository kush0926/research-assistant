"""PostgreSQL / SQLite schema — mirrors Part 21 of the blueprint (+ problems tables for the gap matrix)."""
from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text,
                        create_engine)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from .config import settings

_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(settings.DATABASE_URL, connect_args=_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String(120), nullable=False)
    email = Column(String(200), unique=True, nullable=False, index=True)
    password_hash = Column(String(300), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Paper(Base):
    __tablename__ = "papers"
    id = Column(Integer, primary_key=True)
    title = Column(Text, default="")
    year = Column(Integer)
    venue = Column(String(300))
    doi = Column(String(200))
    abstract = Column(Text, default="")
    source_url = Column(Text)
    source = Column(String(40), default="upload")  # upload | arxiv | semanticscholar | openalex
    external_id = Column(String(200))
    citation_count = Column(Integer, default=0)
    filename = Column(String(300))
    file_path = Column(Text)
    status = Column(String(30), default="processing")  # processing | ready | failed | metadata_only
    error = Column(Text)
    upload_user_id = Column(Integer, ForeignKey("users.id"), index=True)
    processed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    chunks = relationship("Chunk", cascade="all, delete-orphan", back_populates="paper")


class Author(Base):
    __tablename__ = "authors"
    id = Column(Integer, primary_key=True)
    name = Column(String(200), unique=True, index=True)


class PaperAuthor(Base):
    __tablename__ = "paper_authors"
    paper_id = Column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True)
    author_id = Column(Integer, ForeignKey("authors.id"), primary_key=True)
    author_order = Column(Integer, default=0)


class Topic(Base):
    __tablename__ = "topics"
    id = Column(Integer, primary_key=True)
    label = Column(String(200), unique=True)


class PaperTopic(Base):
    __tablename__ = "paper_topics"
    paper_id = Column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True)
    topic_id = Column(Integer, ForeignKey("topics.id"), primary_key=True)
    confidence = Column(Float, default=1.0)


class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(Integer, primary_key=True)
    name = Column(String(200), unique=True, index=True)
    description = Column(Text, default="")


class Method(Base):
    __tablename__ = "methods"
    id = Column(Integer, primary_key=True)
    name = Column(String(200), unique=True, index=True)
    category = Column(String(80), default="")


class Problem(Base):
    __tablename__ = "problems"
    id = Column(Integer, primary_key=True)
    name = Column(String(250), unique=True, index=True)


class PaperMethod(Base):
    __tablename__ = "paper_methods"
    paper_id = Column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True)
    method_id = Column(Integer, ForeignKey("methods.id"), primary_key=True)
    role = Column(String(20), default="used")  # proposed | used


class PaperDataset(Base):
    __tablename__ = "paper_datasets"
    paper_id = Column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), primary_key=True)
    role = Column(String(20), default="used")


class PaperProblem(Base):
    __tablename__ = "paper_problems"
    paper_id = Column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True)
    problem_id = Column(Integer, ForeignKey("problems.id"), primary_key=True)


class Citation(Base):
    __tablename__ = "citations"
    citing_paper_id = Column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True)
    cited_paper_id = Column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True)


class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(Integer, primary_key=True)
    paper_id = Column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    section = Column(String(60), default="body")
    chunk_index = Column(Integer, default=0)
    text = Column(Text, nullable=False)
    page_number = Column(Integer, default=1)
    suspicious = Column(Boolean, default=False)
    paper = relationship("Paper", back_populates="chunks")


class EmbeddingRef(Base):
    __tablename__ = "embedding_refs"
    chunk_id = Column(Integer, ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True)
    vector_db_id = Column(String(100), nullable=False)


class PaperField(Base):
    __tablename__ = "paper_fields"
    id = Column(Integer, primary_key=True)
    paper_id = Column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    field_name = Column(String(60), index=True)
    field_value = Column(Text)  # JSON-encoded
    source_chunk_id = Column(Integer, ForeignKey("chunks.id", ondelete="SET NULL"))
    method = Column(String(20), default="rule")  # rule | ner | llm


class ResearchGap(Base):
    __tablename__ = "research_gaps"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    run_id = Column(String(40), index=True)
    description = Column(Text)
    gap_type = Column(String(30))  # combination | evidence_cluster | contradiction
    tier = Column(String(30))      # evidence_backed | under_explored
    problem_id = Column(Integer, ForeignKey("problems.id"))
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    method_id = Column(Integer, ForeignKey("methods.id"))
    gap_score = Column(Float)
    novelty = Column(Float)
    sparsity = Column(Float)
    evidence_score = Column(Float)
    components = Column(Text)  # JSON of all eight components + weights
    created_at = Column(DateTime, default=datetime.utcnow)


class GapEvidence(Base):
    __tablename__ = "gap_evidence"
    id = Column(Integer, primary_key=True)
    gap_id = Column(Integer, ForeignKey("research_gaps.id", ondelete="CASCADE"), index=True)
    evidence_type = Column(String(30))  # limitation | future_work | contradiction | sparsity
    source_paper_id = Column(Integer, ForeignKey("papers.id", ondelete="SET NULL"))
    source_chunk_id = Column(Integer, ForeignKey("chunks.id", ondelete="SET NULL"))
    note = Column(Text)


def init_db():
    Base.metadata.create_all(engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
