import os, sys, tempfile
from pathlib import Path

_tmp = tempfile.mkdtemp()
os.environ.update({"DATA_DIR": _tmp, "SYNC_INGEST": "true", "EMBEDDING_BACKEND": "hash", "RERANKER_BACKEND": "lexical",
                   "VECTOR_BACKEND": os.environ.get("TEST_VECTOR", "numpy"), "ENABLE_METADATA_LOOKUP": "false", "LLM_PROVIDER": "none",
                   "DATABASE_URL": f"sqlite:///{_tmp}/test.db", "RATE_LIMIT_PER_MIN": "1000"})
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
