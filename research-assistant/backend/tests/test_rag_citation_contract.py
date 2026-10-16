import pytest

from app.config import settings
from app.services import rag

TEXT = "The study used speech recordings and support vector machines to classify Parkinson disease."


def setup_rag(monkeypatch, replies):
    monkeypatch.setattr(settings, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(rag.llm, "provider", "groq")
    monkeypatch.setattr(rag, "owned_ready_ids", lambda *args: [11])
    monkeypatch.setattr(rag, "retrieve", lambda *args: [{
        "chunk_id": 42, "paper_id": 11, "paper_title": "Speech study", "page": 2,
        "section": "method", "text": TEXT, "score": 1.0, "rerank_score": 1.0}])
    monkeypatch.setattr(rag, "_supported", lambda *args: True)
    calls = []
    remaining = iter(replies)

    def complete(system, prompt, max_tokens):
        calls.append(prompt)
        return next(remaining)

    monkeypatch.setattr(rag.llm, "complete", complete)
    class Verifier:
        def complete_json(self, *args):
            return {"verdicts": [{"id": 0, "supported": True, "quotes": [{"marker": 1, "text": TEXT}]}]}
    monkeypatch.setattr(rag, "get_verifier", lambda: Verifier())
    return calls


@pytest.mark.parametrize("first", [TEXT, TEXT + " [99]."])
def test_missing_or_invalid_citations_get_one_repair(monkeypatch, first):
    calls = setup_rag(monkeypatch, [first, TEXT + " [1]."])
    result = rag.answer_question(None, 1, "What speech models were used?")

    assert len(calls) == 2
    assert "missing or invalid citation" in calls[1]
    assert result["mode"] == "llm"
    assert result["citations"][0]["chunk_id"] == 42
    assert result["unverified"] == 0


def test_failed_citation_repair_uses_cited_extractive_fallback(monkeypatch):
    calls = setup_rag(monkeypatch, [TEXT, TEXT + " [0]."])
    result = rag.answer_question(None, 1, "What speech models were used?")

    assert len(calls) == 2
    assert result["mode"] == "extractive"
    assert result["citations"][0]["chunk_id"] == 42
    assert "[0]" not in result["answer"]


def test_valid_cited_answer_does_not_use_extra_generation_request(monkeypatch):
    calls = setup_rag(monkeypatch, [TEXT + " [1]."])
    result = rag.answer_question(None, 1, "What speech models were used?")

    assert len(calls) == 1
    assert result["mode"] == "llm"


def test_unverified_draft_claim_is_withheld_from_delivered_answer(monkeypatch):
    unsupported = "Quantum particles cure Parkinson disease."
    setup_rag(monkeypatch, [TEXT + " [1]. " + unsupported])
    result = rag.answer_question(None, 1, "What speech models were used?")
    assert unsupported not in result["answer"]
    assert result["omitted_unverified_claims"] == 1
    assert result["withheld_claims"][0]["text"] == unsupported
    assert unsupported in result["draft_answer"]
    assert all(c["verified"] for c in result["claims"])


def test_unsupported_cited_answer_uses_source_excerpts(monkeypatch):
    setup_rag(monkeypatch, ["A quantum particle model cured Parkinson disease [1]."])
    class Verifier:
        def complete_json(self, *args):
            return {"verdicts": [{"id": 0, "supported": False, "quotes": []}]}
    monkeypatch.setattr(rag, "get_verifier", lambda: Verifier())
    result = rag.answer_question(None, 1, "What speech models were used?")
    assert result["mode"] == "extractive"
    assert "quantum" not in result["answer"]
    assert result["generation_mode"] == "llm"
    assert result["citations"][0]["chunk_id"] == 42
    assert result["omitted_unverified_claims"] == 1
