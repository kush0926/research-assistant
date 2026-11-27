"""Central configuration. Everything is overridable through environment variables (.env)."""
import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data_store"))
UPLOAD_DIR = DATA_DIR / "uploads"
for _d in (DATA_DIR, UPLOAD_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


class Settings:
    # --- database (PostgreSQL in docker-compose; SQLite fallback for zero-setup runs) ---
    DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'app.db'}")

    # --- auth / security ---
    JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
    JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "720"))
    CORS_ORIGINS = [o.strip() for o in os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")]
    MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
    MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "80"))
    # Keep PDF parsing bounded so a malformed or pathological document cannot
    # hold an ingestion worker for several minutes.
    PARSE_TIMEOUT_S = int(os.getenv("PARSE_TIMEOUT_S", "60"))
    PARSE_MEM_MB = int(os.getenv("PARSE_MEM_MB", "2048"))
    RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "30"))

    # --- LLM: anthropic | openai | groq | gemini | none (offline heuristics) ---
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "none").lower()
    LLM_MODEL = os.getenv("LLM_MODEL", "")
    LLM_VERIFIER_MODEL = os.getenv("LLM_VERIFIER_MODEL", "")  # blank uses the generation model
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")  # also Ollama/Groq/etc.
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

    # --- embeddings / vector DB / reranker ---
    EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "auto")  # auto | bge | hash
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    RERANKER_BACKEND = os.getenv("RERANKER_BACKEND", "auto")  # auto | bge | lexical
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "auto")  # auto | chroma | numpy
    RETRIEVE_K = int(os.getenv("RETRIEVE_K", "10"))
    RERANK_N = int(os.getenv("RERANK_N", "6"))

    # --- document processing ---
    GROBID_URL = os.getenv("GROBID_URL", "")  # e.g. http://grobid:8070 ; empty -> PyMuPDF parser
    CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "260"))  # ~350 tokens
    CHUNK_OVERLAP = float(os.getenv("CHUNK_OVERLAP", "0.15"))
    ENABLE_METADATA_LOOKUP = _bool("ENABLE_METADATA_LOOKUP", True)  # Crossref cross-check
    SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    WARMUP = _bool("WARMUP", True)  # preload models at startup in a background thread
    SYNC_INGEST = _bool("SYNC_INGEST", False)  # tests: ingest inline instead of background task

    # --- gap scoring (Part 15). Hand-set, documented weights, equal by default. ---
    GAP_WEIGHTS = json.loads(os.getenv(
        "GAP_WEIGHTS",
        json.dumps({k: 0.125 for k in
                    ["novelty", "sparsity", "importance", "feasibility",
                     "temporal", "evidence", "opportunity", "contradiction"]})))
    SPARSITY_ONLY_PENALTY = float(os.getenv("SPARSITY_ONLY_PENALTY", "0.75"))
    CONTRADICTION_TOLERANCE = float(os.getenv("CONTRADICTION_TOLERANCE", "5.0"))  # metric points


settings = Settings()
