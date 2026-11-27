"""The UI matrix must expose the same scoped counts used by gap detection."""
from collections import Counter
from types import SimpleNamespace

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db import (Base, Dataset, Method, Paper, PaperDataset, PaperMethod,
                    PaperProblem, Problem, User)
from app.services import gaps


def test_matrix_view_preserves_projections_and_exposes_exact_sparse_counts():
    problems = Counter({"task A": 4, "task B": 1})
    datasets = Counter({"dataset X": 4, "dataset Y": 1})
    methods = Counter({"CNN": 2, "Transformer": 2})
    pd = Counter({("task A", "dataset X"): 3, ("task A", "dataset Y"): 1,
                  ("task B", "dataset X"): 1})
    pm = Counter({("task A", "CNN"): 2, ("task A", "Transformer"): 1,
                  ("task B", "Transformer"): 1})
    triples = Counter({("task A", "dataset X", "CNN"): 2,
                       ("task A", "dataset Y", "Transformer"): 1,
                       ("task B", "dataset X", "Transformer"): 1,
                       ("task A", "dataset X", "Transformer"): 0})

    matrix = gaps._matrix_view(pd, pm, datasets, methods, problems, triples)

    assert matrix["problem_dataset"] == {
        "rows": ["task A", "task B"], "cols": ["dataset X", "dataset Y"], "cells": [[3, 1], [1, 0]]}
    assert matrix["problem_method"] == {
        "rows": ["task A", "task B"], "cols": ["CNN", "Transformer"], "cells": [[2, 1], [0, 1]]}
    assert matrix["problem_dataset_method"] == {
        "problems": ["task A", "task B"], "datasets": ["dataset X", "dataset Y"],
        "methods": ["CNN", "Transformer"], "counts": [
            {"problem": "task A", "dataset": "dataset X", "method": "CNN", "count": 2},
            {"problem": "task A", "dataset": "dataset Y", "method": "Transformer", "count": 1},
            {"problem": "task B", "dataset": "dataset X", "method": "Transformer", "count": 1}]}


def test_detector_matrix_axes_are_not_truncated_with_legacy_projections():
    problems = Counter({f"task {i}": 10 - i for i in range(7)})
    datasets = Counter({f"dataset {i}": 10 - i for i in range(9)})
    methods = Counter({f"method {i}": 10 - i for i in range(9)})
    triples = Counter({("task 6", "dataset 8", "method 8"): 1})

    matrix = gaps._matrix_view(Counter(), Counter(), datasets, methods, problems, triples)

    assert len(matrix["problem_dataset"]["rows"]) == 6
    assert len(matrix["problem_dataset"]["cols"]) == 8
    assert len(matrix["problem_method"]["cols"]) == 8
    full = matrix["problem_dataset_method"]
    assert full["problems"] == list(problems)
    assert full["datasets"] == list(datasets)
    assert full["methods"] == list(methods)
    assert full["counts"] == [
        {"problem": "task 6", "dataset": "dataset 8", "method": "method 8", "count": 1}]


def test_matrix_view_handles_missing_axes_without_inventing_counts():
    matrix = gaps._matrix_view(Counter(), Counter(), Counter(), Counter(), Counter(), Counter())
    assert matrix["problem_dataset_method"] == {"problems": [], "datasets": [], "methods": [], "counts": []}

    matrix = gaps._matrix_view(Counter(), Counter(), Counter({"dataset X": 3}), Counter(),
                               Counter({"task A": 3}), Counter())
    assert matrix["problem_dataset_method"] == {
        "problems": ["task A"], "datasets": ["dataset X"], "methods": [], "counts": []}


def test_detector_matrix_matches_candidates_and_respects_problem_and_corpus_scope(monkeypatch):
    class SimilarEmbedder:
        backend, dim = "bge", 2

        def encode(self, texts):
            return np.tile(np.array([1.0, 0.0]), (len(texts), 1))

    monkeypatch.setattr(gaps, "get_embedder", lambda: SimilarEmbedder())
    monkeypatch.setattr(gaps, "find_contradictions", lambda *_: [])
    monkeypatch.setattr(gaps, "llm", SimpleNamespace(available=False))
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    try:
        with Session(engine) as db:
            user = User(name="Matrix test", email="matrix@example.test", password_hash="unused")
            task_a, task_b = Problem(name="task A"), Problem(name="task B")
            dataset_x, dataset_y = Dataset(name="dataset X"), Dataset(name="dataset Y")
            cnn, transformer = Method(name="CNN", category="model"), Method(name="Transformer", category="model")
            feature = Method(name="MFCC", category="feature")
            db.add_all([user, task_a, task_b, dataset_x, dataset_y, cnn, transformer, feature,
                        Problem(name="unused task"), Dataset(name="unused dataset"),
                        Method(name="unused method", category="model")])
            db.flush()
            paper_ids = []
            for index, (problem, dataset, method) in enumerate([
                    (task_a, dataset_x, cnn), (task_a, dataset_x, cnn),
                    (task_a, dataset_y, transformer), (task_b, dataset_x, transformer),
                    (task_a, dataset_x, feature)]):
                paper = Paper(title=f"Study {index}", abstract="Classification study.",
                              upload_user_id=user.id, status="ready", year=2025)
                db.add(paper); db.flush()
                paper_ids.append(paper.id)
                db.add_all([PaperProblem(paper_id=paper.id, problem_id=problem.id),
                            PaperDataset(paper_id=paper.id, dataset_id=dataset.id),
                            PaperMethod(paper_id=paper.id, method_id=method.id)])
            db.commit()

            result = gaps.detect_gaps(db, user.id, max_gaps=30)
            matrix = result["matrix"]["problem_dataset_method"]
            counts = {(c["problem"], c["dataset"], c["method"]): c["count"] for c in matrix["counts"]}
            assert matrix["problems"] == ["task A", "task B"]
            assert matrix["datasets"] == ["dataset X", "dataset Y"]
            assert matrix["methods"] == ["CNN", "Transformer"]
            assert counts == {("task A", "dataset X", "CNN"): 2,
                              ("task A", "dataset Y", "Transformer"): 1,
                              ("task B", "dataset X", "Transformer"): 1}
            candidates = [g for g in result["gaps"] if g["type"] == "combination"]
            assert any((g["problem"], g["dataset"], g["method"]) ==
                       ("task A", "dataset X", "Transformer") for g in candidates)
            for candidate in candidates:
                key = (candidate["problem"], candidate["dataset"], candidate["method"])
                assert counts.get(key, 0) == candidate["cell_count"]

            scoped = gaps.detect_gaps(db, user.id, paper_ids=paper_ids[:3], max_gaps=30)
            scoped_matrix = scoped["matrix"]["problem_dataset_method"]
            assert scoped["corpus_size"] == 3
            assert scoped_matrix["problems"] == ["task A"]
            assert scoped_matrix["counts"] == [c for c in matrix["counts"] if c["problem"] == "task A"]
    finally:
        engine.dispose()
