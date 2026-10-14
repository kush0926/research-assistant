"""Embedding generation (BGE via sentence-transformers) with an offline hashing fallback."""
import hashlib
import logging
import re
import threading

import numpy as np

from ..config import settings

log = logging.getLogger("embeddings")
_STOP = set("a an the of to in on for and or is are was were be by with as at from that this it its we our "
            "which these those can using use used based also not but than then into over such".split())
DIM_HASH = 384


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9][a-z0-9\-']+|[a-z0-9]", text.lower()) if t not in _STOP]


class Embedder:
    def __init__(self):
        self.backend = "hash"
        self.model = None
        self._lock = threading.Lock()
        if settings.EMBEDDING_BACKEND in ("auto", "bge"):
            try:
                from sentence_transformers import SentenceTransformer  # noqa
                log.info("Loading embedding model %s (first run downloads it; this can take a few minutes)...", settings.EMBEDDING_MODEL)
                self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
                self.backend = "bge"
                log.info("Embeddings: %s", settings.EMBEDDING_MODEL)
            except Exception as e:  # not installed / offline
                if settings.EMBEDDING_BACKEND == "bge":
                    raise
                log.warning("BGE unavailable (%s) -> hashing embeddings fallback", e)

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension() if self.model else DIM_HASH

    def _hash_vec(self, text: str) -> np.ndarray:
        v = np.zeros(DIM_HASH, dtype=np.float32)
        toks = tokenize(text)
        feats = toks + [a + "_" + b for a, b in zip(toks, toks[1:])]
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            v[h % DIM_HASH] += 1.0 if "_" not in f else 0.7 * (-1 if (h >> 20) & 1 else 1) + 0.7
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self.model is not None:
            prefix = "Represent this sentence for searching relevant passages: " if is_query else ""
            with self._lock:
                return np.asarray(self.model.encode([prefix + t for t in texts], normalize_embeddings=True,
                                                    batch_size=32, show_progress_bar=False), dtype=np.float32)
        return np.stack([self._hash_vec(t) for t in texts])


_embedder: Embedder | None = None
_embedder_lock = threading.Lock()


def get_embedder() -> Embedder:
    """Thread-safe lazy singleton: concurrent uploads must not each load their own copy of the model."""
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                _embedder = Embedder()
    return _embedder
