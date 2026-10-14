"""Cross-encoder reranking (bge-reranker-base) with a lexical fallback (BM25-ish overlap + vector score blend)."""
import logging
import math
import threading

from ..config import settings
from .embeddings import tokenize

log = logging.getLogger("reranker")


class Reranker:
    def __init__(self):
        self.model = None
        if settings.RERANKER_BACKEND in ("auto", "bge"):
            try:
                from sentence_transformers import CrossEncoder
                log.info("Loading reranker %s (first run downloads it)...", settings.RERANKER_MODEL)
                self.model = CrossEncoder(settings.RERANKER_MODEL)
            except Exception as e:
                if settings.RERANKER_BACKEND == "bge":
                    raise
                log.warning("Cross-encoder unavailable (%s) -> lexical reranker", e)

    def rerank(self, query: str, candidates: list[dict], top_n: int) -> list[dict]:
        """candidates: dicts with 'text' and 'score' (vector similarity). Adds 'rerank_score'."""
        if not candidates:
            return []
        if self.model is not None:
            scores = self.model.predict([(query, c["text"]) for c in candidates])
            for c, s in zip(candidates, scores):
                c["rerank_score"] = float(1 / (1 + math.exp(-float(s))))
        else:
            q = set(tokenize(query))
            for c in candidates:
                toks = tokenize(c["text"])
                tset = set(toks)
                overlap = len(q & tset) / (len(q) or 1)
                tf = sum(toks.count(t) for t in q) / (len(toks) or 1)
                c["rerank_score"] = 0.6 * overlap + 0.2 * min(1.0, tf * 15) + 0.2 * max(0.0, c.get("score", 0))
        return sorted(candidates, key=lambda c: -c["rerank_score"])[:top_n]


_rr = None
_rr_lock = threading.Lock()


def get_reranker() -> Reranker:
    global _rr
    if _rr is None:
        with _rr_lock:
            if _rr is None:
                _rr = Reranker()
    return _rr
