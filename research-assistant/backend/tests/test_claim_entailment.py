import pytest

from app.config import settings
from app.services import rag

SOURCE = "The XGBoost model was not tested on the PC-GITA dataset and no accuracy was reported."
CLAIM = "The XGBoost model was tested on the PC-GITA dataset and 92 percent accuracy was reported [1]."


@pytest.mark.parametrize("supported,supports", [
    (False, [{"marker": 1, "sentence_id": 1}]),
    ("false", [{"marker": 1, "sentence_id": 1}]),
    (True, [{"marker": 1, "sentence_id": 99}]),
    (True, [{"marker": 9, "sentence_id": 1}]),
    (True, [{"marker": 1, "sentence_id": True}]),
    (True, []),
])
def test_high_overlap_does_not_bypass_entailment_or_quote_checks(monkeypatch, supported, supports):
    monkeypatch.setattr(rag.llm, "provider", "groq")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "test-key")
    calls = []
    class Verifier:
        def complete_json(self, system, payload, max_tokens):
            calls.append(payload)
            return {"verdicts": [{"id": 0, "supported": supported, "supports": supports}]}
    monkeypatch.setattr(rag, "get_verifier", lambda: Verifier())
    claims = rag.verify_answer(CLAIM, [{"text": SOURCE}])
    assert calls
    assert not claims[0]["verified"]


def test_exact_extractive_claim_can_be_verified_offline(monkeypatch):
    monkeypatch.setattr(rag.llm, "provider", "none")
    assert rag.verify_answer(SOURCE.rstrip('.') + " [1].", [{"text": SOURCE}])[0]["verified"]
    assert not rag.verify_answer(CLAIM, [{"text": SOURCE}])[0]["verified"]


def test_duplicate_verifier_ids_are_ambiguous_and_not_accepted(monkeypatch):
    monkeypatch.setattr(rag.llm, "provider", "groq")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "test-key")
    class Verifier:
        def complete_json(self, *args):
            verdict = {"id": 0, "supported": True, "supports": [{"marker": 1, "sentence_id": 1}]}
            return {"verdicts": [verdict, verdict]}
    monkeypatch.setattr(rag, "get_verifier", lambda: Verifier())
    paraphrase = "XGBoost was not evaluated and the study reported no accuracy [1]."
    assert not rag.verify_answer(paraphrase, [{"text": SOURCE}])[0]["verified"]


def test_selected_source_sentence_is_returned_as_original_support_quote(monkeypatch):
    monkeypatch.setattr(rag.llm, "provider", "groq")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "test-key")
    class Verifier:
        def complete_json(self, *args):
            return {"verdicts": [{"id": 0, "supported": True, "supports": [{"marker": 1, "sentence_id": 1}]}]}
    monkeypatch.setattr(rag, "get_verifier", lambda: Verifier())
    claim = "The study did not evaluate XGBoost on PC-GITA and reported no accuracy [1]."
    result = rag.verify_answer(claim, [{"text": SOURCE}])[0]
    assert result["verified"]
    assert result["supporting_quotes"][0]["text"] == SOURCE
