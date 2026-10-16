from eval import rag_ablation


def test_basic_rag_uses_unreranked_context_and_resolves_citation_markers(monkeypatch):
    contexts = [
        {"chunk_id": 7, "paper_id": 2, "paper_title": "Paper B", "section": "results", "page": 3,
         "text": "The reported model achieved 91 percent accuracy.", "score": 0.8},
        {"chunk_id": 8, "paper_id": 3, "paper_title": "Paper C", "section": "method", "page": 2,
         "text": "The study used voice recordings.", "score": 0.7},
    ]
    calls = {}

    def retrieve(_db, question, paper_ids, *, k, n):
        calls.update(question=question, paper_ids=paper_ids, k=k, n=n)
        return contexts[:n]

    def generate(system, prompt, max_tokens):
        calls.update(system=system, prompt=prompt, max_tokens=max_tokens)
        return "The model reached 91 percent accuracy [1]. Invalid source [9]."

    monkeypatch.setattr(rag_ablation, "retrieve_without_reranking", retrieve)
    result = rag_ablation.answer_basic_rag(None, "What accuracy?", [2, 3], generate, k=10, n=2)

    assert calls["k"] == 10 and calls["n"] == 2
    assert "Cite every factual claim" not in calls["system"]
    assert result["citations"] == [{"marker": 1, "paper_id": 2, "chunk_id": 7, "paper_title": "Paper B",
                                    "section": "results", "page": 3}]
    assert result["invalid_citation_markers"] == [9]


def test_review_record_keeps_quality_labels_unset_and_computes_context_overlap():
    output = {"answer": "A cited answer [1].", "contexts": [{"paper_id": 4, "chunk_id": 11, "text": "Evidence."}],
              "citations": [{"marker": 1, "paper_id": 4, "chunk_id": 11}], "invalid_citation_markers": []}
    row = rag_ablation.review_record("Question", ["paper_04.pdf"], {"basic": output}, {4: "paper_04.pdf"})

    method = row["methods"]["basic"]
    assert method["context_precision"] == 1.0 and method["context_recall"] == 1.0
    assert method["human_review"]["faithfulness_1_to_5"] is None
    assert method["human_review"]["citation_correctness_1_to_5"] is None
