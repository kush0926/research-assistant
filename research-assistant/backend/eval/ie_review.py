"""Human annotation worksheet helpers for information-extraction evaluation."""


def build_ie_review_draft(papers: list[dict], get_paper_detail, get_chunk) -> dict:
    records = []
    skipped = []
    chunk_cache = {}

    def chunk_evidence(chunk_id):
        if not chunk_id:
            return None
        if chunk_id not in chunk_cache:
            response = get_chunk(chunk_id)
            chunk_cache[chunk_id] = response if response and response.get("chunk_id") == chunk_id else None
        chunk = chunk_cache[chunk_id]
        if not chunk:
            return {"chunk_id": chunk_id, "text": None, "resolved": False}
        return {"chunk_id": chunk_id, "text": chunk.get("text", "")[:1200], "resolved": True}

    for paper in papers:
        if paper.get("status") != "ready":
            skipped.append({"paper_id": paper.get("id"), "filename": paper.get("filename"),
                            "reason": paper.get("error") or f"status={paper.get('status')}"})
            continue
        detail = get_paper_detail(paper["id"])
        fields = detail.get("fields", {})
        predictions = {}
        for kind in ("methods", "datasets"):
            predictions[kind] = []
            for item in fields.get(kind, {}).get("value", []) or []:
                predictions[kind].append({
                    "name": item.get("name"),
                    "role": item.get("role"),
                    "category": item.get("category"),
                    "confidence": item.get("confidence"),
                    "evidence": chunk_evidence(item.get("chunk_id")),
                })
        records.append({
            "filename": paper.get("filename"),
            "paper_id": paper["id"],
            "title": detail.get("title") or paper.get("title"),
            "system_predictions": predictions,
            # Leave reference labels blank so annotators can label independently.
            "methods": None,
            "datasets": None,
            "review_status": "unreviewed",
            "reviewer_1": None,
            "reviewer_2": None,
            "adjudicator": None,
            "notes": "",
        })
    return {
        "status": "draft_annotations_awaiting_double_review_not_gold_labels",
        "instructions": [
            "Annotate the paper independently before looking at system_predictions.",
            "Enter reference methods and datasets as arrays of canonical names; use [] only after confirming none are present.",
            "After both reviewers complete a record, resolve disagreements, set review_status to reviewed, and record the adjudicator.",
            "Before evaluation, set the top-level status to reviewed_gold and pass this file with --ie-gold.",
        ],
        "records": records,
        "skipped": skipped,
    }


def load_ie_gold_document(document) -> list[dict]:
    """Accept legacy gold lists or a fully reviewed worksheet; reject draft worksheets."""
    if isinstance(document, dict):
        if document.get("status") != "reviewed_gold":
            raise ValueError("IE annotations are not gold: top-level status must be 'reviewed_gold'")
        records = document.get("records")
        if not isinstance(records, list):
            raise ValueError("Reviewed IE gold must contain a records list")
        require_review_status = True
    elif isinstance(document, list):
        records = document
        require_review_status = False
    else:
        raise ValueError("IE gold must be a list or reviewed worksheet object")

    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Every IE gold record must be an object")
        if (require_review_status and record.get("review_status") != "reviewed") or (
            not require_review_status and record.get("review_status") not in (None, "reviewed")
        ):
            raise ValueError(f"IE annotation for {record.get('filename', '<unknown>')} is not reviewed")
        for field in ("methods", "datasets"):
            if not isinstance(record.get(field), list):
                raise ValueError(f"IE gold record for {record.get('filename', '<unknown>')} needs a {field} list")
    return records
