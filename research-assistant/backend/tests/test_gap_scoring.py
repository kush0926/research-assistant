"""Gap ranking must respond to corpus support even when embeddings are identical."""
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.db import (Base, Chunk, Dataset, Method, Paper, PaperDataset, PaperField,
                    PaperMethod, PaperProblem, Problem, ResearchGap, User)
from app.services import gaps


@pytest.fixture
def corpus(monkeypatch):
    class IdenticalEmbedder:
        backend, dim = "bge", 2

        def encode(self, texts, **_kwargs):
            return np.tile(np.array([1.0, 0.0]), (len(texts), 1))

    monkeypatch.setattr(gaps, "get_embedder", lambda: IdenticalEmbedder())
    monkeypatch.setattr(gaps, "llm", SimpleNamespace(available=False))
    monkeypatch.setattr(gaps, "find_contradictions", lambda *_args: [])
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        user = User(name="Scoring tests", email="scoring@example.test", password_hash="unused")
        db.add(user)
        db.flush()
        entities = {}

        def entity(model, name):
            key = model, name
            if key not in entities:
                entities[key] = model(name=name, **({"category": "model"} if model is Method else {}))
                db.add(entities[key])
                db.flush()
            return entities[key]

        def add_paper(dataset=None, method=None, problem="task A", year=2025, citations=0):
            paper = Paper(title=f"Study {db.query(Paper).count() + 1}", abstract="Classification study.",
                          upload_user_id=user.id, status="ready", year=year, citation_count=citations)
            db.add(paper)
            db.flush()
            for model, link, column, name in [
                    (Problem, PaperProblem, "problem_id", problem),
                    (Dataset, PaperDataset, "dataset_id", dataset),
                    (Method, PaperMethod, "method_id", method)]:
                if name is not None:
                    db.add(link(paper_id=paper.id, **{column: entity(model, name).id}))
            db.commit()
            return paper

        def set_future_work(paper, repeats=1):
            db.query(PaperField).filter(PaperField.paper_id == paper.id,
                                        PaperField.field_name == "future_work").delete()
            sentence = "Future work should evaluate Transformer on PC-GITA."
            chunk = Chunk(paper_id=paper.id, section="future_work", chunk_index=0, text=sentence)
            db.add(chunk)
            db.flush()
            db.add(PaperField(paper_id=paper.id, field_name="future_work", field_value=json.dumps(
                [{"text": sentence, "chunk_id": chunk.id}] * repeats)))
            db.commit()

        yield SimpleNamespace(db=db, user=user, add_paper=add_paper, set_future_work=set_future_work)
    engine.dispose()


@pytest.fixture
def api(corpus, monkeypatch):
    from fastapi.testclient import TestClient
    from app import main

    monkeypatch.setitem(main.app.dependency_overrides, main.get_db, lambda: corpus.db)
    monkeypatch.setitem(main.app.dependency_overrides, main.current_user, lambda: corpus.user)
    monkeypatch.setitem(main.app.dependency_overrides, main.rate_limit, lambda: corpus.user)
    client = TestClient(main.app)
    yield client
    client.close()


def detect(corpus, **kwargs):
    return gaps.detect_gaps(corpus.db, corpus.user.id, max_gaps=100, **kwargs)


def combination(result, dataset, method="Transformer"):
    return next(g for g in result["gaps"] if g["type"] == "combination"
                and g["dataset"] == dataset and g["method"] == method)


def conflict(dataset, papers, delta):
    return {"dataset": dataset, "method": "Transformer", "metric": "Accuracy", "delta": delta,
            "papers": [{"paper_id": p.id, "value": 60 + index * delta / max(1, len(papers) - 1),
                        "sentence": f"Reported accuracy in study {p.id}.", "chunk_id": None}
                       for index, p in enumerate(papers)], "note": "Study settings may differ."}


@pytest.mark.parametrize("weights", [None, {"sparsity": 1}])
def test_empty_cells_with_stronger_structural_support_rank_higher(corpus, weights):
    for _ in range(3):
        corpus.add_paper("strong dataset", "CNN")
    corpus.add_paper("weak dataset", "CNN")
    for _ in range(2):
        corpus.add_paper("other dataset", "Transformer")

    result = detect(corpus, weights=weights)
    strong = combination(result, "strong dataset")
    weak = combination(result, "weak dataset")

    assert strong["cell_count"] == weak["cell_count"] == 0
    assert strong["components"]["novelty"] == weak["components"]["novelty"]
    assert 0 < weak["components"]["sparsity"] < strong["components"]["sparsity"] < 1
    assert strong["score"] > weak["score"]


def test_opportunity_can_be_supported_in_a_small_single_cluster_corpus(corpus):
    corpus.add_paper("PC-GITA", "CNN")
    corpus.add_paper("other dataset", "Transformer")
    corpus.add_paper("other dataset", "Transformer")

    target = combination(detect(corpus), "PC-GITA")

    assert target["components"]["opportunity"] > 0


def test_structural_support_prefers_methods_used_on_the_same_problem(corpus):
    corpus.add_paper("PC-GITA", "CNN")
    for _ in range(2):
        corpus.add_paper("other dataset", "Transformer")
        corpus.add_paper("other dataset", "SVM", problem="task B")

    result = detect(corpus, weights={"sparsity": 1})
    within_problem = combination(result, "PC-GITA", "Transformer")
    transfer = combination(result, "PC-GITA", "SVM")

    assert within_problem["cell_count"] == transfer["cell_count"] == 0
    assert 0 < transfer["components"]["sparsity"] < within_problem["components"]["sparsity"]
    assert within_problem["score"] > transfer["score"]


def test_an_additional_transfer_source_does_not_lower_opportunity(corpus, monkeypatch):
    class MixedRelevanceEmbedder:
        backend, dim = "bge", 2

        def encode(self, texts, **_kwargs):
            return np.array([[0.0, 1.0] if text.startswith("Unrelated transfer.") else [1.0, 0.0]
                             for text in texts])

    corpus.add_paper("PC-GITA", "CNN")
    for _ in range(3):
        corpus.add_paper("other dataset", "Transformer")
    monkeypatch.setattr(gaps, "get_embedder", lambda: MixedRelevanceEmbedder())
    before = combination(detect(corpus, weights={"opportunity": 1}), "PC-GITA")

    unrelated = corpus.add_paper("remote dataset", "Transformer", problem="task B")
    unrelated.title = "Unrelated transfer"
    corpus.db.commit()
    after = combination(detect(corpus, weights={"opportunity": 1}), "PC-GITA")

    assert after["components"]["opportunity"] > before["components"]["opportunity"]
    assert after["score"] > before["score"]
    assert before["support"]["paper_count"] == 4
    assert after["support"]["paper_count"] == 5


def test_repeated_evidence_counts_independent_papers_without_saturating_at_five(corpus):
    papers = [corpus.add_paper("PC-GITA", "CNN")]
    papers.extend(corpus.add_paper("other dataset", "Transformer") for _ in range(7))
    corpus.set_future_work(papers[0])
    first = combination(detect(corpus), "PC-GITA")

    corpus.set_future_work(papers[0], repeats=20)
    repeated = combination(detect(corpus), "PC-GITA")
    assert first["components"]["evidence"] == repeated["components"]["evidence"] == 0.2
    assert first["score"] == repeated["score"]
    assert repeated["support"]["evidence_paper_count"] == 1
    assert repeated["score_breakdown"]["penalty"] == 1

    for paper in papers[1:5]:
        corpus.set_future_work(paper)
    five_result = detect(corpus)
    five = combination(five_result, "PC-GITA")
    corpus.set_future_work(papers[5])
    six_result = detect(corpus)
    six = combination(six_result, "PC-GITA")

    assert 0.2 < five["components"]["evidence"] < six["components"]["evidence"] < 1
    assert five["score"] < six["score"]
    assert five["support"]["evidence_paper_count"] == 5
    assert six["support"]["evidence_paper_count"] == 6
    five_theme = next(g for g in five_result["gaps"] if g["type"] == "evidence_cluster")
    six_theme = next(g for g in six_result["gaps"] if g["type"] == "evidence_cluster")
    assert five_theme["components"]["evidence"] < six_theme["components"]["evidence"] < 1
    assert five_theme["score"] < six_theme["score"]


def test_contradiction_scores_respond_to_magnitude_and_independent_sources(corpus, monkeypatch):
    papers = [corpus.add_paper("PC-GITA", "Transformer") for _ in range(4)]

    def score(delta, sources):
        ct = conflict("PC-GITA", sources, delta)
        monkeypatch.setattr(gaps, "find_contradictions", lambda *_args: [ct])
        return next(g for g in detect(corpus, weights={"contradiction": 1})["gaps"]
                    if g["type"] == "contradiction")

    mild = score(6, papers[:2])
    large = score(40, papers[:2])
    corroborated = score(40, papers)
    repeated_source = score(40, papers[:2] + [papers[0]] * 5)

    assert 0 < mild["score"] < large["score"] < corroborated["score"] <= 1
    assert repeated_source["score"] == large["score"]
    assert large["support"]["paper_count"] == 2
    assert corroborated["support"]["paper_count"] == 4


def test_contradiction_candidate_components_use_source_metadata(corpus, monkeypatch):
    papers = [corpus.add_paper("PC-GITA", "Transformer", year=None, citations=0) for _ in range(3)]
    ct = conflict("PC-GITA", papers[:2], 20)
    monkeypatch.setattr(gaps, "find_contradictions", lambda *_args: [ct])
    before = next(g for g in detect(corpus)["gaps"] if g["type"] == "contradiction")

    papers[0].year, papers[0].citation_count = 2026, 300
    papers[1].year, papers[1].citation_count = 2026, 200
    corpus.db.commit()
    after = next(g for g in detect(corpus)["gaps"] if g["type"] == "contradiction")

    assert before["support"]["dated_paper_count"] == 0
    assert after["support"]["dated_paper_count"] == 2
    assert after["components"]["importance"] > before["components"]["importance"]
    assert after["components"]["temporal"] > before["components"]["temporal"]


def test_missing_metadata_still_has_finite_explainable_persisted_scores(corpus):
    corpus.add_paper("PC-GITA", "CNN", year=None, citations=None)
    corpus.add_paper("other dataset", "Transformer", year=None, citations=None)
    corpus.add_paper("other dataset", "Transformer", year=None, citations=None)

    result = detect(corpus)

    assert result["gaps"]
    for gap in result["gaps"]:
        assert math.isfinite(gap["score"]) and 0 <= gap["score"] <= 1
        assert all(math.isfinite(value) and 0 <= value <= 1 for value in gap["components"].values())
        breakdown = gap["score_breakdown"]
        assert set(breakdown["contributions"]) == set(gaps.COMPONENTS)
        assert sum(breakdown["contributions"].values()) == pytest.approx(breakdown["weighted_sum"], abs=0.001)
        assert gap["score"] == pytest.approx(breakdown["weighted_sum"] * breakdown["penalty"], abs=0.0002)
        assert breakdown["penalty"] == gaps.settings.SPARSITY_ONLY_PENALTY
        assert gap["support"]["dated_paper_count"] == 0
        row = corpus.db.get(ResearchGap, gap["id"])
        assert row.gap_score == gap["score"]
        stored = json.loads(row.components)
        assert stored["score_breakdown"] == gap["score_breakdown"]
        assert stored["support"] == gap["support"]


@pytest.mark.parametrize("weights", [
    {}, {"sparsity": 0}, {"sparsity": -1, "evidence": 2}, {"sparsity": float("nan")},
    {"sparsity": float("inf")}, {"sparsity": float("-inf")}, {"sparsity": "bad"},
    {"sparsity": 1, "typo": 2},
])
def test_invalid_weights_are_rejected(weights):
    with pytest.raises(ValueError):
        gaps._normalize_weights(weights)


def test_valid_partial_weights_are_normalized_without_filling_missing_signal():
    weights = gaps._normalize_weights({"sparsity": 2, "evidence": 1})

    assert set(weights) == set(gaps.COMPONENTS)
    assert sum(weights.values()) == pytest.approx(1)
    assert weights["sparsity"] == pytest.approx(2 / 3)
    assert weights["evidence"] == pytest.approx(1 / 3)
    assert weights["novelty"] == 0


def test_large_finite_weights_do_not_overflow_normalization():
    weights = gaps._normalize_weights({"sparsity": 1e308, "evidence": 1e308})

    assert weights["sparsity"] == weights["evidence"] == 0.5
    assert all(math.isfinite(value) for value in weights.values())
    assert sum(weights.values()) == pytest.approx(1)


def test_true_ties_are_honest_and_canonically_sorted(corpus, monkeypatch):
    corpus.add_paper("Z dataset", "base method")
    corpus.add_paper("A dataset", "base method")
    corpus.add_paper("other dataset", "target method")
    corpus.add_paper("other dataset", "target method")

    def tied(result):
        return [g for g in result["gaps"] if g["type"] == "combination"
                and g["method"] == "target method" and g["dataset"] in {"A dataset", "Z dataset"}]

    first = tied(detect(corpus))
    original_select = gaps.select_corpus
    monkeypatch.setattr(gaps, "select_corpus", lambda *args, **kwargs: list(reversed(original_select(*args, **kwargs))))
    second = tied(detect(corpus))

    assert [g["dataset"] for g in first] == [g["dataset"] for g in second] == ["A dataset", "Z dataset"]
    assert first[0]["score"] == first[1]["score"]
    assert first[0]["components"] == first[1]["components"]
    assert [g["score"] for g in first] == [g["score"] for g in second]


def test_ranking_uses_precision_before_display_rounding(corpus, monkeypatch):
    papers = [corpus.add_paper("PC-GITA", "Transformer") for _ in range(3)]
    conflicts = [conflict("A dataset", papers[:2], 40), conflict("Z dataset", papers[:2], 40.00001)]
    monkeypatch.setattr(gaps, "find_contradictions", lambda *_args: conflicts)

    ranked = [g for g in detect(corpus, weights={"contradiction": 1})["gaps"] if g["type"] == "contradiction"]

    assert [g["dataset"] for g in ranked] == ["Z dataset", "A dataset"]
    assert ranked[0]["score"] == ranked[1]["score"]


@pytest.mark.parametrize("weights", [{}, {"sparsity": 0}, {"sparsity": -1}, {"unknown": 1}])
def test_gap_request_rejects_invalid_weights_with_a_validation_response(api, weights):
    response = api.post("/research-gaps", json={"weights": weights})

    assert response.status_code == 422
    assert any("weights" in error["loc"] for error in response.json()["detail"])


@pytest.mark.parametrize("value", ['"NaN"', '"Infinity"', "1e309"])
def test_nonfinite_request_weights_return_validation_errors(api, value):
    response = api.post("/research-gaps", content='{"weights":{"sparsity":' + value + '}}',
                        headers={"Content-Type": "application/json"})

    assert response.status_code == 422


def test_gap_detail_preserves_scoring_explanation_from_detection(corpus, api):
    corpus.add_paper("PC-GITA", "CNN")
    corpus.add_paper("other dataset", "Transformer")
    corpus.add_paper("other dataset", "Transformer")
    response = api.post("/research-gaps", json={"weights": {"sparsity": 2, "evidence": 1}})
    assert response.status_code == 200
    result = response.json()
    candidate = combination(result, "PC-GITA")

    saved = api.get(f"/research-gaps/{candidate['id']}")

    assert saved.status_code == 200
    for key in ("score", "components", "weights", "score_breakdown", "support", "scoring_version"):
        assert saved.json()[key] == candidate[key]
    assert sum(candidate["weights"].values()) == pytest.approx(1)
