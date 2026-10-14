"""Vector DB: Chroma (persistent, metadata filtering) with a pure-numpy persisted fallback."""
import json
import logging
import threading

import numpy as np

from ..config import DATA_DIR, settings

log = logging.getLogger("vectorstore")


class NumpyStore:
    """Tiny exact-cosine store with the same interface as the Chroma wrapper."""

    def __init__(self):
        self.path = DATA_DIR / "numpy_store.json"
        self._lock = threading.RLock()
        self.ids: list[str] = []
        self.vecs: list[list[float]] = []
        self.meta: list[dict] = []
        if self.path.exists():
            d = json.loads(self.path.read_text())
            self.ids, self.vecs, self.meta = d["ids"], d["vecs"], d["meta"]

    def _save(self):
        self.path.write_text(json.dumps({"ids": self.ids, "vecs": self.vecs, "meta": self.meta}))

    def add(self, ids, vectors, metadatas):
        with self._lock:
            index = {i: j for j, i in enumerate(self.ids)}
            for i, v, m in zip(ids, vectors, metadatas):
                if i in index:
                    j = index[i]
                    self.vecs[j], self.meta[j] = list(map(float, v)), m
                else:
                    index[i] = len(self.ids)
                    self.ids.append(i); self.vecs.append(list(map(float, v))); self.meta.append(m)
            self._save()

    def query(self, vector, k, paper_ids=None, sections=None):
        if not self.ids:
            return []
        M = np.asarray(self.vecs, dtype=np.float32)
        sims = M @ np.asarray(vector, dtype=np.float32)
        order = np.argsort(-sims)
        out = []
        for j in order:
            m = self.meta[j]
            if paper_ids and m["paper_id"] not in paper_ids:
                continue
            if sections and m.get("section") not in sections:
                continue
            out.append((self.ids[j], float(sims[j]), m))
            if len(out) >= k:
                break
        return out

    def delete_paper(self, paper_id: int):
        with self._lock:
            keep = [i for i, m in enumerate(self.meta) if m["paper_id"] != paper_id]
            self.ids = [self.ids[i] for i in keep]; self.vecs = [self.vecs[i] for i in keep]
            self.meta = [self.meta[i] for i in keep]
            self._save()


class ChromaStore:
    def __init__(self):
        import chromadb
        self.client = chromadb.PersistentClient(path=str(DATA_DIR / "chroma"))
        self.col = self.client.get_or_create_collection("chunks", metadata={"hnsw:space": "cosine"})

    def add(self, ids, vectors, metadatas):
        self.col.upsert(ids=list(ids), embeddings=[list(map(float, v)) for v in vectors], metadatas=metadatas)

    def query(self, vector, k, paper_ids=None, sections=None):
        conds = []
        if paper_ids:
            conds.append({"paper_id": {"$in": list(paper_ids)}})
        if sections:
            conds.append({"section": {"$in": list(sections)}})
        where = None if not conds else (conds[0] if len(conds) == 1 else {"$and": conds})
        if self.col.count() == 0:
            return []
        r = self.col.query(query_embeddings=[list(map(float, vector))], n_results=min(k, self.col.count()), where=where)
        return [(i, 1.0 - d, m) for i, d, m in zip(r["ids"][0], r["distances"][0], r["metadatas"][0])]

    def delete_paper(self, paper_id: int):
        self.col.delete(where={"paper_id": paper_id})


_store = None
_store_lock = threading.Lock()


def get_store():
    """Thread-safe lazy singleton."""
    global _store
    with _store_lock:
        if _store is None:
            if settings.VECTOR_BACKEND in ("auto", "chroma"):
                try:
                    _store = ChromaStore()
                    log.info("Vector DB: Chroma")
                except Exception as e:
                    if settings.VECTOR_BACKEND == "chroma":
                        raise
                    log.warning("Chroma unavailable (%s) -> numpy store", e)
            if _store is None:
                _store = NumpyStore()
    return _store
