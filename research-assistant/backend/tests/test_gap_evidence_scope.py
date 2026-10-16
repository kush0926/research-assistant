from app.services.gaps import _matches_candidate_evidence, _related_contradictions
import pytest


def test_limitation_evidence_respects_problem_scope_when_labels_exist():
    item = {"paper_id": 1, "p": set(), "d": {"PC-GITA"}, "m": set()}
    paper_problems = {1: {"Parkinson's detection"}}

    assert not _matches_candidate_evidence("Alzheimer's detection", "PC-GITA", "Transformer",
                                           item, 0.9, 0.6, paper_problems)
    assert _matches_candidate_evidence("Parkinson's detection", "PC-GITA", "Transformer",
                                       item, 0.1, 0.6, paper_problems)


def test_limitation_evidence_keeps_semantic_fallback_for_unlabeled_papers():
    item = {"paper_id": 2, "p": set(), "d": set(), "m": set()}

    assert _matches_candidate_evidence("Parkinson's detection", "PC-GITA", "Transformer",
                                       item, 0.7, 0.6, {})
    assert not _matches_candidate_evidence("Parkinson's detection", "PC-GITA", "Transformer",
                                           item, 0.5, 0.6, {})


def test_explicit_statement_problem_takes_precedence_over_paper_topics():
    item = {"paper_id": 1, "p": {"Alzheimer's detection"}, "d": {"PC-GITA"}, "m": set()}
    paper_problems = {1: {"Parkinson's detection", "Alzheimer's detection"}}

    assert not _matches_candidate_evidence("Parkinson's detection", "PC-GITA", "Transformer",
                                           item, 0.9, 0.6, paper_problems)
    assert _matches_candidate_evidence("Alzheimer's detection", "PC-GITA", "Transformer",
                                       item, 0.1, 0.6, paper_problems)


def test_contradiction_evidence_requires_dataset_method_and_matching_problem():
    conflict = {"dataset": "PC-GITA", "method": "Transformer",
                "papers": [{"paper_id": 1}, {"paper_id": 2}]}
    paper_problems = {1: {"Parkinson's detection"}, 2: {"Parkinson's detection"}}

    assert _related_contradictions("Parkinson's detection", "PC-GITA", "Transformer",
                                   [conflict], paper_problems) == [conflict]
    assert _related_contradictions("Alzheimer's detection", "PC-GITA", "Transformer",
                                   [conflict], paper_problems) == []
    assert _related_contradictions("Parkinson's detection", "Other dataset", "Transformer",
                                   [conflict], paper_problems) == []
    assert _related_contradictions("Parkinson's detection", "PC-GITA", "CNN",
                                   [conflict], paper_problems) == []


def test_contradiction_evidence_allows_unknown_problem_labels():
    conflict = {"dataset": "PC-GITA", "method": "Transformer", "papers": [{"paper_id": 1}, {"paper_id": 2}]}

    assert _related_contradictions("Parkinson's detection", "PC-GITA", "Transformer",
                                   [conflict], {}) == [conflict]


@pytest.mark.parametrize("sentence,specific", [
    ("Future work should collect recordings from more subjects.", False),
    ("Future work should use Transformer on more recordings.", False),
    ("Future work should collect more PC-GITA recordings.", False),
    ("Future work should evaluate Transformer on PC-GITA.", True),
])
def test_semantically_related_future_work_requires_both_combination_entities(monkeypatch, sentence, specific):
    import json
    import numpy as np
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from app.db import (Base, Chunk, Dataset, Method, Paper, PaperDataset, PaperField,
                        PaperMethod, PaperProblem, Problem, User)
    from app.services import gaps

    class SimilarEmbedder:
        backend, dim = "bge", 2

        def encode(self, texts):
            return np.tile(np.array([1.0, 0.0]), (len(texts), 1))

    monkeypatch.setattr(gaps, "get_embedder", lambda: SimilarEmbedder())
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    try:
        with Session(engine) as db:
            user = User(name="Generic scope", email="generic@example.test", password_hash="unused")
            problem = Problem(name="task A")
            pc, other = Dataset(name="PC-GITA"), Dataset(name="OtherCorpus")
            cnn, transformer = Method(name="CNN", category="model"), Method(name="Transformer", category="model")
            db.add_all([user, problem, pc, other, cnn, transformer]); db.flush()
            for index, (dataset, method) in enumerate([(pc, cnn), (other, transformer), (other, transformer)]):
                paper = Paper(title=f"Study {index}", abstract="Task A classification.",
                              upload_user_id=user.id, status="ready", year=2025)
                db.add(paper); db.flush()
                db.add_all([PaperProblem(paper_id=paper.id, problem_id=problem.id),
                            PaperDataset(paper_id=paper.id, dataset_id=dataset.id),
                            PaperMethod(paper_id=paper.id, method_id=method.id)])
                if index == 0:
                    chunk = Chunk(paper_id=paper.id, section="future_work", chunk_index=0, text=sentence)
                    db.add(chunk); db.flush()
                    db.add(PaperField(paper_id=paper.id, field_name="future_work",
                                      field_value=json.dumps([{"text": sentence, "chunk_id": chunk.id}])))
            db.commit()
            result = gaps.detect_gaps(db, user.id, max_gaps=30)
            candidate = next(g for g in result["gaps"] if g["type"] == "combination"
                             and g["dataset"] == "PC-GITA" and g["method"] == "Transformer")
            assert candidate["components"]["evidence"] == (0.2 if specific else 0)
            assert candidate["tier"] == ("evidence_backed" if specific else "under_explored")
            assert {e["type"] for e in candidate["evidence"]} == ({"future_work", "sparsity"} if specific else {"related_context", "sparsity"})
            assert "extracted records in this uploaded corpus" in candidate["description"]
    finally:
        engine.dispose()


def test_gap_detector_does_not_promote_candidates_with_evidence_from_another_problem():
    import json

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from app.db import (Base, Chunk, Dataset, Method, Paper, PaperDataset, PaperField,
                        PaperMethod, PaperProblem, Problem, User)
    from app.services.gaps import detect_gaps

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    try:
        with Session(engine) as db:
            user = User(name="Scope review", email="scope@example.test", password_hash="unused")
            task_a, task_b = Problem(name="task A"), Problem(name="task B")
            dataset = Dataset(name="PC-GITA")
            cnn, transformer = Method(name="CNN", category="model"), Method(name="Transformer", category="model")
            db.add_all([user, task_a, task_b, dataset, cnn, transformer])
            db.flush()
            for index, (problem, method) in enumerate([(task_a, cnn), (task_b, transformer), (task_b, transformer)]):
                paper = Paper(title=f"Study {index} of {problem.name}", abstract=f"Evaluate {problem.name}.",
                              upload_user_id=user.id, status="ready", year=2025)
                db.add(paper)
                db.flush()
                db.add_all([PaperProblem(paper_id=paper.id, problem_id=problem.id),
                            PaperDataset(paper_id=paper.id, dataset_id=dataset.id),
                            PaperMethod(paper_id=paper.id, method_id=method.id)])
                if problem == task_b:
                    sentence = "Future work should evaluate Transformer on PC-GITA with more subjects."
                    result_sentence = f"Accuracy is {70 + 10 * index}%."
                    chunk = Chunk(paper_id=paper.id, section="future_work", chunk_index=0,
                                  text=f"{sentence} {result_sentence}")
                    db.add(chunk)
                    db.flush()
                    db.add(PaperField(paper_id=paper.id, field_name="future_work",
                                      field_value=json.dumps([{"text": sentence, "chunk_id": chunk.id}])))
                    db.add(PaperField(paper_id=paper.id, field_name="result_records", field_value=json.dumps([
                        {"dataset": dataset.name, "method": method.name, "metric": "Accuracy",
                         "value": 70 + 10 * index, "sentence": result_sentence,
                         "chunk_id": chunk.id}])))
            db.commit()

            result = detect_gaps(db, user.id, max_gaps=30)
            assert result["contradictions"], "The fixture must exercise the contradiction association path."
            candidate = next(gap for gap in result["gaps"]
                             if gap["type"] == "combination" and gap["problem"] == task_a.name
                             and gap["dataset"] == dataset.name and gap["method"] == transformer.name)
            assert candidate["components"]["evidence"] == 0
            assert candidate["components"]["contradiction"] == 0
            assert candidate["tier"] == "under_explored"
            assert {item["type"] for item in candidate["evidence"]} == {"sparsity"}
    finally:
        engine.dispose()
