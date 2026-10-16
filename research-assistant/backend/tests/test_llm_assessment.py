import pytest
from eval.llm_verification import validate_claim_assessment, checked_entities


def test_judge_cannot_omit_claims_or_use_nonverbatim_quotes_to_get_full_support():
    claims = [{"id": 0, "text": "A supported claim [1].", "citation_markers": [1]},
              {"id": 1, "text": "An omitted claim.", "citation_markers": []}]
    passages = [{"text": "The source contains exact evidence for the model."}]
    result = {"claims": [{"id": 0, "supported": True,
                          "quotes": [{"marker": 1, "text": "The source ... model."}]}]}
    validated = validate_claim_assessment(claims, passages, result)
    assert validated["supported_claim_fraction"] == 0
    assert validated["unresolved_claims"] == 2


def test_source_support_and_citation_coverage_are_separate():
    quote = "The source contains exact evidence for the model."
    claims = [{"id": 0, "text": "An uncited supported claim.", "citation_markers": []}]
    result = {"claims": [{"id": 0, "supported": True, "quotes": [{"marker": 1, "text": quote}]}]}
    validated = validate_claim_assessment(claims, [{"text": quote}], result)
    assert validated["supported_claim_fraction"] == 1
    assert validated["citation_bound_support_fraction"] == 0


def test_entity_reference_rejects_invented_source_quote():
    result = {"methods": [{"name": "LSTM", "quote": "We evaluated LSTM."}], "datasets": []}
    accepted, rejected = checked_entities(result, "We evaluated a different classifier.")
    assert not accepted["methods"]
    assert rejected[0]["reason"] == "quote_not_in_supplied_source"


@pytest.mark.parametrize("sentence_id,expected", [(1, 1.0), (99, 0.0)])
def test_source_sentence_pointers_must_resolve(sentence_id, expected):
    claims = [{"id": 0, "text": "Supported claim [1].", "citation_markers": [1]}]
    result = {"claims": [{"id": 0, "supported": True, "supports": [{"marker": 1, "sentence_id": sentence_id}]}]}
    validated = validate_claim_assessment(claims, [{"text": "The source contains evidence for the claim."}], result)
    assert validated["supported_claim_fraction"] == expected
