from app.services.rag import _citation_contract_met, source_sentence_units, split_answer_claims, verify_answer
from app.services import rag


def test_citations_after_period_stay_with_previous_bullet():
    claims = split_answer_claims("- The first study used PASE. [1]\n- The second study used MFCCs. [2][3]")
    assert claims == ["The first study used PASE [1].", "The second study used MFCCs [2][3]."]


def test_abbreviation_and_year_do_not_split_one_claim():
    claims = split_answer_claims("The encoder (Pascual et al. 2019) was evaluated [1].")
    assert claims == ["The encoder (Pascual et al. 2019) was evaluated [1]."]


def test_orphan_markers_do_not_become_claims():
    assert not _citation_contract_met("[1]", 2)
    assert split_answer_claims("[1]\n- A valid statement [2].") == ["A valid statement [2]."]


def test_adjacent_bullets_do_not_borrow_each_others_source(monkeypatch):
    monkeypatch.setattr(rag.llm, "provider", "none")
    top = [{"text": "The first study used PASE."}, {"text": "The second study used MFCCs."}]
    claims = verify_answer("- The first study used PASE. [1]\n- The second study used MFCCs. [2]", top)
    assert [c["markers"] for c in claims] == [[1], [2]]
    assert all(c["verified"] for c in claims)


def test_source_sentence_ids_preserve_original_words_and_citation_order():
    source = "Pascual et al. 2019 studied speech [3]. Another study followed."
    units = source_sentence_units(source)
    assert len(units) == 2
    assert units[0]["text"] == "Pascual et al. 2019 studied speech [3]."
    assert all(unit["text"] in source for unit in units)
