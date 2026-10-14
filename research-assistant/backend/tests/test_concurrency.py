"""Regression: concurrent first use must create exactly ONE embedder / reranker / vector store
(previously, parallel uploads each loaded their own copy of the model)."""
import threading
import time

from app.services import embeddings, reranker, vectorstore


def _hammer(fn, n=12):
    out = []
    ts = [threading.Thread(target=lambda: out.append(fn())) for _ in range(n)]
    [t.start() for t in ts]; [t.join() for t in ts]
    return out


def test_single_embedder_under_concurrency(monkeypatch):
    created = []
    orig_init = embeddings.Embedder.__init__

    def slow_init(self):
        created.append(1); time.sleep(0.2); orig_init(self)

    monkeypatch.setattr(embeddings.Embedder, "__init__", slow_init)
    monkeypatch.setattr(embeddings, "_embedder", None)
    objs = _hammer(embeddings.get_embedder)
    assert len(created) == 1 and len({id(o) for o in objs}) == 1


def test_single_reranker_and_store_under_concurrency(monkeypatch):
    created = []
    orig = reranker.Reranker.__init__

    def slow(self):
        created.append(1); time.sleep(0.2); orig(self)

    monkeypatch.setattr(reranker.Reranker, "__init__", slow)
    monkeypatch.setattr(reranker, "_rr", None)
    monkeypatch.setattr(vectorstore, "_store", None)
    assert len({id(o) for o in _hammer(reranker.get_reranker)}) == 1 and len(created) == 1
    assert len({id(o) for o in _hammer(vectorstore.get_store)}) == 1
