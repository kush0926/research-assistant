from types import SimpleNamespace

from app.services import comparison, litreview, synthesis


def table():
    return {"rows": [{"paper_id": i, "title": f"Study {i}", "year": 2025,
                      **{d: {"text": "SVM" if d == "algorithm" else "—"} for d in comparison.DIMENSIONS}}
                     for i in (1, 2)], "dimensions": comparison.DIMENSIONS}


def test_review_tags_and_headings_are_normalized_before_verification(monkeypatch):
    captured = []
    monkeypatch.setattr(synthesis.rag, "verify_answer", lambda text, sources:
                        captured.append((text, sources)) or [{"verified": True}])
    result = synthesis.verify_synthesis("## Methods\nSVM is used [P1].", [{"methods": ["SVM"]}], True)
    assert result["status"] == "passed"
    assert "[1]" in captured[0][0] and "[P1]" not in captured[0][0] and "## Methods" not in captured[0][0]
    assert result["scope"] == "exact_structured_inputs"


def test_invalid_paper_marker_rejects_even_an_affirmative_verdict(monkeypatch):
    monkeypatch.setattr(synthesis.rag, "verify_answer", lambda *args: [{"verified": True}])
    assert synthesis.verify_synthesis("SVM [2].", [{"methods": ["SVM"]}])["status"] == "fallback"


def test_uncited_claim_cannot_pass_with_a_cited_claim(monkeypatch):
    monkeypatch.setattr(synthesis.rag, "verify_answer", lambda *args: [{"verified": True}, {"verified": False}])
    assert synthesis.verify_synthesis("SVM [1]. It is the best.", [{}])["unverified_claims"] == 1


def test_heading_only_output_cannot_pass(monkeypatch):
    monkeypatch.setattr(synthesis.rag, "verify_answer", lambda *args: [])
    assert synthesis.verify_synthesis("## Methods", [{}])["status"] == "fallback"


def test_comparison_rejects_unsupported_generated_prose(monkeypatch):
    monkeypatch.setattr(comparison, "llm", SimpleNamespace(available=True, complete=lambda *args: "All models outperform humans [1]."))
    monkeypatch.setattr(comparison, "verify_synthesis", lambda *args: {"status": "fallback"})
    result = comparison.narrative_result(table(), [])
    assert result["generation_mode"] == "llm" and result["mode"] == "template"
    assert "outperform humans" not in result["narrative"] and "Shared methods: SVM" in result["narrative"]


def test_comparison_returns_verified_generation(monkeypatch):
    monkeypatch.setattr(comparison, "llm", SimpleNamespace(available=True, complete=lambda *args: "Both use SVM [1][2]."))
    monkeypatch.setattr(comparison, "verify_synthesis", lambda *args: {"status": "passed"})
    result = comparison.narrative_result(table(), [])
    assert result["mode"] == "llm" and result["narrative"] == "Both use SVM [1][2]."


class NoGaps:
    def query(self, *args): return self
    def filter_by(self, **kwargs): return self
    def order_by(self, *args): return self
    def limit(self, *args): return self
    def all(self): return []


def review_setup(monkeypatch, accepted):
    facts = {"tag": "P1", "title": "Study", "year": 2025, "methods": ["SVM"], "datasets": [],
             "problem_labels": [], "best_result": "", "limitation": "", "future_work": ""}
    monkeypatch.setattr(litreview, "_facts", lambda *args: facts)
    monkeypatch.setattr(litreview, "find_contradictions", lambda *args: [])
    monkeypatch.setattr(litreview, "llm", SimpleNamespace(available=True, complete=lambda *args: "## Methods\n- Uses SVM [P1]."))
    monkeypatch.setattr(litreview, "verify_synthesis", lambda *args, **kwargs: {"status": "passed" if accepted else "fallback"})


def test_review_uses_factual_template_when_generation_fails_verification(monkeypatch):
    review_setup(monkeypatch, False)
    result = litreview.generate(NoGaps(), [SimpleNamespace(id=1)], 1)
    assert result["mode"] == "template" and result["generation_mode"] == "llm"
    assert result["verification"]["status"] == "fallback" and "# Literature Review Draft" in result["markdown"]


def test_review_keeps_verified_generation_and_resolved_references(monkeypatch):
    review_setup(monkeypatch, True)
    result = litreview.generate(NoGaps(), [SimpleNamespace(id=1)], 1)
    assert result["mode"] == "llm" and "Uses SVM [P1]" in result["markdown"]
    assert "[P1] Study (2025)" in result["markdown"]
