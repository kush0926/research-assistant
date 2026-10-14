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
uvicorn app.main:app --reload            # SQLite by default
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
| LLM (Anthropic / OpenAI-compatible / Gemini) | structured extraction, grounded answers, narratives | rule/dictionary extraction, extractive cited answers |
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
when they share the candidate's dataset or method. Every gap links to the exact source sentences.

## Evaluation
```bash
python backend/eval/run_eval.py --corpus <pdf_dir> --gold gold_gaps.json --ie-gold ie_gold.json --queries retrieval_queries.json
```
Reports gap P/R/F1@K + MRR for sparsity-only vs evidence-only vs hybrid; retrieval Recall/MRR/NDCG for TF-IDF vs embedding vs embedding+rerank; IE P/R/F1.

## Honest status — read before your viva
- **Verified here:** 8 end-to-end tests pass (auth, upload validation, ingestion, extraction, cited RAG + refusal, comparison, gap detection, graph, recommendations, review, agent, per-user isolation) on both the numpy and the real Chroma store; live-server async ingestion; frontend type-checks and builds.
- **Not exercised in my environment (no network/GPU/keys):** BGE embeddings, the bge cross-encoder, GROBID, the three LLM providers, PostgreSQL via Docker, external paper-search APIs, and the UI in a browser. The code paths are written to spec but **run them once and fix any environment-specific issue before the demo**.
- **The bundled sample corpus is synthetic** (clearly stamped on each PDF) with one *planted* gap. Results on it only prove the harness works. **Do not report them as findings.** Build the real 15–25 paper benchmark with manually derived gaps (blueprint Part 4-G).
- The domain dictionary covers voice/Parkinson's plus common ML terms; extend `data/domain_dict.json` for your domain. Entity normalisation and extraction accuracy bound gap quality, so measure IE first.
- Scanned PDFs (OCR), paywalled full-text, and learned scoring weights are out of scope, as in the blueprint.
