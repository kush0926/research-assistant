# AI-Powered Research Assistant — literature review, paper analysis, evidence-backed research gaps

Implementation of the project blueprint: ingest papers → structured extraction → cited RAG → multi-paper comparison →
**hybrid structural + textual research-gap detection with an explicit, adjustable GapScore** → knowledge graph → recommendations → literature-review draft.

## Quick start

**Option A — Docker (Postgres + backend + frontend)**
```bash
cp .env.example .env        # set JWT_SECRET; optionally add an LLM key
docker compose up --build   # UI http://localhost:3000   API docs http://localhost:8000/docs
# optional GROBID parser:  docker compose --profile grobid up   (and set GROBID_URL=http://grobid:8070 in .env)
```

**Option B — local, no Docker**
```bash
cd backend
pip install -r requirements.txt          # full stack (BGE, bge-reranker, Chroma). Use requirements-lite.txt for a tiny install
uvicorn app.main:app --reload --env-file ../.env  # SQLite by default
cd ../frontend && npm install && npm run dev   # http://localhost:3000
```
Try it immediately with the bundled synthetic corpus: `python backend/scripts/make_sample_papers.py sample_papers`, then upload those PDFs.

## Works with or without an LLM key
| Component | Full mode | Automatic fallback (no key / offline) |
|---|---|---|
| Embeddings | `BAAI/bge-base-en-v1.5` (sentence-transformers) | hashing embeddings |
| Reranker | `bge-reranker-base` cross-encoder | lexical BM25-style |
| Vector DB | Chroma (persistent, metadata filter) | numpy store |
| PDF parser | GROBID (if `GROBID_URL`) | PyMuPDF in a sandboxed subprocess |
| LLM (Anthropic / OpenAI-compatible / Groq / Gemini) | structured extraction, grounded answers, narratives | rule/dictionary extraction, extractive cited answers |
Check what is active: `GET /health`. **For your demo and report use the full mode** (BGE + reranker + an LLM key); the fallbacks exist so nothing breaks, and their quality is lower.

## Blueprint → code map
| Blueprint | Where |
|---|---|
| Parts 9, 11 pipeline: validate, sandboxed parse, clean, sections, chunk, metadata | `backend/app/services/pdf_processing.py`, `pdf_worker.py` |
| Part 12 hybrid extraction (rule + dictionary NER + LLM, evidence pointer per field) | `services/extraction.py`, `data/domain_dict.json` |
| Part 11 RAG: retrieve → filter → rerank → grounded generation → citation resolve → verification | `services/rag.py`, `reranker.py`, `vectorstore.py` |
| Part 13 comparison + contradiction detection | `services/comparison.py` |
| **Parts 14–15 gap detection + GapScore** | `services/gaps.py` |
| Parts 16, 17 knowledge graph, hybrid + gap-aware recommendation | `services/graph.py`, `recommend.py` |
| Part 18 rule-based coordinator (bounded, not an open planner) | `services/agent.py` |
| Part 21 schema, Part 22 API | `app/db.py`, `app/main.py` |
| Part 23 security: JWT, per-user isolation, upload validation, sandbox, SSRF guard, prompt-injection sanitising, rate limits | `app/security.py`, `pdf_processing.py`, `paper_search.py` |
| Parts 27–28 evaluation + baselines + ablations | `backend/eval/run_eval.py` |

## How the gap score works (viva-ready)
A candidate is a (problem, dataset, method) cell that is empty (or "explored early, then dropped") **while its neighbours are populated**.
`GapScore = Σ wᵢ·componentᵢ` over novelty, sparsity, importance, feasibility, temporal, evidence, opportunity, contradiction (Part 15).
Weights are hand-set, equal by default, normalised, and editable in the UI; they are **not learned**. A candidate with no textual evidence and no
contradiction is kept but multiplied by `SPARSITY_ONLY_PENALTY` (0.75) and labelled "under-explored" (lower confidence). Contradictions only count
when dataset and method match, with compatible problem labels. Every gap links to the exact source sentences.

## Verification status — 2026-10-14

The user requested automated LLM verification instead of manual review. Groq GPT-OSS 20B generates outputs and GPT-OSS 120B checks source support. Assessments are labeled automated; they are not human gold or independently measured research accuracy.

- **122 backend tests pass with NumPy and 122 with Chroma.** Coverage includes ownership, PDF limits, citation repair, source sentence verification, withheld unsupported claims, grounded entities, result semantics, gap evidence scope, and comparison source links.
- **100 distinct real PDFs passed the PostgreSQL/GROBID/BGE/Chroma workflow**, including extraction, cited questions, comparison, graph, recommendations, gaps, and retrieval after a new process. They produced 3,364 chunks. Mean ingestion was 19.05s; timings include CPU/model-startup variation.
- **1,000 representative paper records / 33,640 chunks passed a separate volume check and fresh-process retrieval.** This uses the 100 real papers plus 900 explicitly tagged replicas with reused embeddings. It does not establish quality or ingestion performance for 1,000 distinct PDFs. Replicas were removed after verification.
- **The isolated Docker test interface uses full models:** Groq, BGE embeddings, BGE cross-encoder, Chroma, PostgreSQL, and GROBID. UI: [localhost:3001](http://localhost:3001); API docs: [localhost:8001/docs](http://localhost:8001/docs). The full Groq overlay uses a separate database and vector volume to avoid mixing hashing and BGE dimensions.
- **Generated Q&A is checked against original numbered source sentences.** Claims without accepted support are withheld; if needed the answer uses cited source excerpts. A source link and an LLM verdict remain automated evidence, not a guarantee of correctness.
- **Extraction and gap errors found in real papers were corrected:** generic dataset names, unquoted LLM entities, features shown as models, relative percentages interpreted as absolute results, ambiguous result attribution, and broadly related text promoting specific gap combinations. Comparison cells expose every supporting passage. Comparison and review prose now also require a separate LLM check against their structured inputs; a failed check returns the factual template summary.
- **Automated assessment is recorded in `backend/eval/results/`.** The original 20-paper assessment exposed low dictionary/reference agreement and citation issues. Current synthesis assessment measures fidelity to the exact supplied structured facts; source-scoped entity reference agreement is not full-paper accuracy.
- **Retrieval benefit remains unproven:** on the four-query pilot, embedding-only and reranked retrieval had identical final parser scores (Precision@5 .300, Recall@5 .792, MRR 1.000, NDCG@5 .845). A four-query sample cannot establish general quality.

### Reproduce and inspect

See [evaluation guide](backend/eval/README.md) and [blueprint traceability](backend/eval/BLUEPRINT_TRACEABILITY.md) for commands, report paths, and limitations. The original 20-paper corpus is pinned by `backend/eval/parkvoice_manifest.json`; the 100-paper downloader records source identifiers and hashes in its manifest.

### Deferred scope

Frontend visual redesign and a dedicated temporal-trend chart remain later work, as requested. Backend trend data is available. OCR for scanned PDFs, paywalled full text, and learned gap-score weights remain outside the blueprint scope. Semantic Scholar may return upstream rate limits, and Groq free-plan quotas can delay or force fallback generation.
