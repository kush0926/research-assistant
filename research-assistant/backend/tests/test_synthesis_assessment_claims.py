from eval.assess_current_outputs import synthesis_claims


def test_review_citations_after_period_stay_with_original_claim():
    claims = synthesis_claims("## Methods\nSVM is used. [P1]\n\nLSTM is used. [P2]", True, 2)
    assert len(claims) == 2
    assert claims[0]["citation_markers"] == [1] and claims[1]["citation_markers"] == [2]
    assert all("is used" in claim["text"] for claim in claims)


def test_review_reference_list_and_orphan_marker_are_not_factual_claims():
    claims = synthesis_claims("SVM is used [P1].\n\n[P1]\n\n## References\n[P1] Study", True, 1)
    assert len(claims) == 1 and claims[0]["citation_markers"] == [1]
