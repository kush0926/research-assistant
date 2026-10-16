import pytest

from eval.ie_review import build_ie_review_draft, load_ie_gold_document


def test_ie_review_draft_keeps_labels_blank_and_attaches_source_passages():
    papers = [
        {"id": 1, "filename": "one.pdf", "title": "Paper One", "status": "ready"},
        {"id": 2, "filename": "two.pdf", "status": "failed", "error": "parse error"},
    ]
    details = {1: {"title": "Paper One", "fields": {
        "methods": {"value": [{"name": "CNN", "role": "proposed", "chunk_id": 11}]},
        "datasets": {"value": [{"name": "PC-GITA", "chunk_id": 12}]},
    }}}
    chunks = {11: {"chunk_id": 11, "text": "We propose a CNN model."},
              12: {"chunk_id": 12, "text": "The PC-GITA dataset was used."}}

    draft = build_ie_review_draft(papers, lambda pid: details[pid], lambda cid: chunks.get(cid))

    assert draft["status"] == "draft_annotations_awaiting_double_review_not_gold_labels"
    assert draft["records"][0]["filename"] == "one.pdf"
    assert draft["records"][0]["methods"] is None and draft["records"][0]["datasets"] is None
    prediction = draft["records"][0]["system_predictions"]
    assert prediction["methods"][0]["evidence"] == {"chunk_id": 11, "text": "We propose a CNN model.", "resolved": True}
    assert draft["skipped"] == [{"paper_id": 2, "filename": "two.pdf", "reason": "parse error"}]


def test_ie_gold_loader_rejects_unreviewed_worksheet():
    draft = {"status": "draft_annotations_awaiting_double_review_not_gold_labels", "records": []}
    with pytest.raises(ValueError, match="not gold"):
        load_ie_gold_document(draft)


def test_ie_gold_loader_requires_reviewed_records_and_accepts_complete_gold():
    draft_gold = {"status": "reviewed_gold", "records": [
        {"filename": "one.pdf", "methods": ["CNN"], "datasets": ["PC-GITA"]}
    ]}
    with pytest.raises(ValueError, match="not reviewed"):
        load_ie_gold_document(draft_gold)

    gold = {"status": "reviewed_gold", "records": [
        {"filename": "one.pdf", "methods": ["CNN"], "datasets": ["PC-GITA"], "review_status": "reviewed"}
    ]}
    assert load_ie_gold_document(gold) == gold["records"]
