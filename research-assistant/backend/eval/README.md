# Evaluation and reproduction

## Current scope

The user requested LLM verification instead of manual review. GPT-OSS 20B generates through Groq; GPT-OSS 120B independently assesses source support. Supporting sentence IDs are resolved back to original text. Results are automated assessments, not human gold, and do not prove research accuracy or global novelty.

Keep credentials in the application-root `.env`. Never copy them into reports. All commands below run from the application root with `backend/.venv/Scripts/python.exe`; activate that environment or substitute its Python executable on other platforms.

## Recorded evidence (2026-10-14)

| Report | Scope |
|---|---|
| `results/pytest_numpy_final_2026-10-14.log` | 122 tests passed using NumPy. |
| `results/pytest_chroma_final_2026-10-14.log` | 122 tests passed using Chroma. |
| `results/capacity_100_2026-10-14.json` | 100 distinct PDFs, GROBID/BGE/cross-encoder/PostgreSQL/Chroma, fresh-process persistence, current-rule reanalysis, queries, comparison, graph, recommendations and gaps. |
| `results/volume_1000_2026-10-14.json` | 1,000 representative records and 33,640 chunks; 900 tagged replicas use existing vectors. Fresh-process verification and replica cleanup passed. |
| `results/docker_full_groq_workflow_2026-10-14.json` | Actual deployed full-stack ingestion of three public PDFs, verified Q&A, comparison passage resolution, live review, graph and recommendations. |
| `results/llm_assessment_2026-10-14.json` | Automated diagnosis of original Basic/proposed RAG, eight old gap candidates, and 20-paper source-scoped extraction references. |
| `results/current_llm_assessment_2026-10-14.json` | Current comparison/review fidelity to their exact inputs and qualification of current gap claims. |
| `results/warm_runtime_benchmark_2026-10-14.json` | Clean warm three-paper CPU diagnostic, excluding model loading and provider quota waits. |

Historical attempts remain available; consult the report status and scope before using a number. The 100-paper corpus has 100 distinct source IDs and hashes. The 1,000-record check is a capacity diagnostic, not 1,000-source research evaluation. Its process memory excludes PostgreSQL, Docker and GROBID.

## Backend checks

```powershell
$env:TEST_VECTOR = "numpy"
backend/.venv/Scripts/python.exe -m pytest backend/tests -q
$env:TEST_VECTOR = "chroma"
backend/.venv/Scripts/python.exe -m pytest backend/tests -q
```

The suite uses temporary data, hashing embeddings, lexical ranking and mocked/no LLM calls. Separate real-model reports establish integration behavior. On this Windows host, TestClient needs socket access unavailable in the restricted execution sandbox.

## Public corpora and 100-paper integration

```powershell
backend/.venv/Scripts/python.exe backend/eval/fetch_parkvoice_corpus.py
backend/.venv/Scripts/python.exe backend/eval/fetch_scale_corpus.py
```

The first downloader uses the pinned 20-paper manifest. The scale downloader uses a saved arXiv API response, reuses those files, checks PDF limits/text, and records source identifiers and hashes in `backend/data_store/parkvoice_scale_corpus/manifest.json`. arXiv requests are sequential and spaced by at least 3.1 seconds.

Start PostgreSQL and GROBID, obtain the dynamically published PostgreSQL port, then set the explicit test URL:

```powershell
docker compose -f docker-compose.yml -f backend/eval/docker.readiness.override.yml --profile grobid up -d db grobid
docker compose -f docker-compose.yml -f backend/eval/docker.readiness.override.yml port db 5432
$env:POSTGRES_TEST_URL = "postgresql://ra:ra@127.0.0.1:55679/ra_readiness_test"
backend/.venv/Scripts/python.exe backend/scripts/verify_corpus_capacity.py --state backend/eval/results/capacity_100_state_2026-10-14.json --report backend/eval/results/capacity_100_2026-10-14.json
backend/.venv/Scripts/python.exe backend/scripts/verify_corpus_capacity.py --phase verify --state backend/eval/results/capacity_100_state_2026-10-14.json --report backend/eval/results/capacity_100_2026-10-14.json
```

Use the actual published port; 55679 is this run's port. The runner only permits the named test database and creates its own random schema and Chroma directory. Reuse `--phase verify` to reanalyze current rules without re-embedding/re-uploading. Do not run ingest against an existing state file to repeat a completed check.

The volume runner uses `verify_vector_volume.py`; inspect its `--help` for state arguments. It adds only marked replicas in its isolated test schema/index and removes those replicas during verification, retaining the real 100 papers.

## Full Docker test interface

The readiness overlay alone selects hash/lexical/offline fallbacks. Add the Groq overlay for the full test configuration:

```powershell
docker compose -f docker-compose.yml -f backend/eval/docker.readiness.override.yml -f backend/eval/docker.groq.override.yml --profile grobid up -d --build backend frontend
```

Before starting this overlay, create `ra_groq_full_test` in the existing test PostgreSQL service. It uses `groq_full_appdata`, so 384-dimensional hash vectors are not mixed with 768-dimensional BGE vectors. The overlay mounts only the two public BGE model directories from `backend/data_store/docker_hf_cache` at `/opt/huggingface` read-only, with Hugging Face/Transformers offline mode. Populate that cache from official model downloads before enabling it. The copied directory must contain `hub/models--BAAI--bge-base-en-v1.5` and `hub/models--BAAI--bge-reranker-base`, including blobs, refs and snapshots. Do not copy private tokens into it.

- UI: http://localhost:3001
- API docs: http://localhost:8001/docs
- `GET /health` must report Groq, BGE, cross-encoder, ChromaStore and GROBID.

The deployed verification helper reuses the specifically named isolated browser test user from `ra_readiness_test`; it does not create UI accounts. It preserves that user's ID in the separate full-model DB so the existing browser session remains valid. Credentials remain in memory, and only three public corpus PDFs are uploaded through the API:

```powershell
backend/.venv/Scripts/python.exe backend/scripts/verify_deployed_full.py --output backend/eval/results/docker_full_groq_workflow_2026-10-14.json
```

It reuses ready papers and completed output checks when resuming. Its deliberate generator pacing is reported separately from normal UI behavior. Free-tier request limits can otherwise cause deterministic fallbacks; inspect `mode` and `generation_mode` before claiming live generation.

## Automated LLM assessment

```powershell
backend/.venv/Scripts/python.exe backend/eval/llm_verification.py --output backend/eval/results/llm_assessment_2026-10-14.json
backend/.venv/Scripts/python.exe backend/eval/assess_current_outputs.py --workflow backend/eval/results/docker_full_groq_workflow_2026-10-14.json --gaps backend/eval/results/capacity_100_2026-10-14_gaps_current.json --output backend/eval/results/current_llm_assessment_2026-10-14.json
```

The first command assesses the original pilot outputs and creates independent entity references from selected source passages (up to 10,500 characters per paper), without showing predictions to the reference extractor. A second role check excludes features, preprocessing, metrics, software and cited-only methods. The original dictionary/reference agreement was low (F1 about .227); this is a diagnostic of incomplete coverage and differing entity scope, not full-paper accuracy. Do not convert it into a final accuracy claim.

The second command verifies current comparison/review claims against their exact structured generation inputs. It checks original claim coverage and source sentence IDs, and separately assesses current gap wording. Input fidelity does not establish that extraction from a paper was correct. The runtime RAG verifier checks retrieved original source passages, withholds unsupported claims, and reports omitted claims and fallback provenance. Comparison and review generation now also require this source-sentence protocol against their exact structured inputs. If any generated factual claim is unverified, the complete synthesis falls back to its deterministic fact summary; `mode`, `generation_mode`, and `verification` distinguish delivery from generation. The final live sample generated both comparison and review with Groq, then delivered factual templates after the verifier rejected two and three draft claims respectively. The earlier unguarded synthesis and its failed claims remain in the `*_before_synthesis_guard_2026-10-14.json` reports.

Immutable prompt/response caches are stored under ignored `backend/data_store/llm_assessment_cache`. Verifier requests are conservatively paced, retried on rate limits, and use a separate model from generation. LLM self-assessment can share errors with generation; these results remain automated evidence.

## Final automated outcome

`results/readiness_summary_2026-10-14.json` collects the completed checks. All 122 tests pass with each vector store. The final source-sentence assessment accepted all five claims in the delivered comparison template and all thirteen claims in the delivered review template against their structured inputs. This is input fidelity, not full-paper accuracy. The eight current gap candidates are correctly retained as `under_explored`: the verifier accepted their local, qualified wording, found no specific paired textual support, and established no global novelty. The existing browser session also passed live Groq Q&A, three source links, an opened original passage, and graph rendering; see `results/browser_smoke_2026-10-14.json` and its screenshots.

## Retrieval, baselines and optional reference formats

`run_eval.py` compares TF-IDF, embeddings, and embeddings plus reranking. It can write IE and Basic/proposed RAG worksheets and run the workflow. Run `--help` for all options. The four-query pilot is small; final GROBID/BGE embedding-only and reranked metrics were identical: Precision@5 .300, Recall@5 .792, MRR 1.000 and NDCG@5 .845. Reranking benefit is not demonstrated.

Legacy `--gold`, `--ie-gold` and human-rating worksheets remain available for a future independently annotated benchmark. Their validation rules still distinguish reviewed labels from empty templates. They are not required for this user's LLM-only verification request. Do not relabel generated references as reviewed human gold to bypass those rules.

## Practical limits

Warm CPU timing met the small diagnostic target, but cold startup, longer papers, concurrent CPU work, provider latency and free quotas increase latency. The 100-paper mean was 19.05s with a maximum of 66.36s; 20/22 ten-to-fifteen-page papers were below 30s. No general latency or research-accuracy guarantee is established.

Frontend visual redesign and a dedicated temporal chart are deferred. Scanned-PDF OCR, paywalled downloads and learned gap-score weights remain outside the blueprint scope. External discovery is implemented, but Semantic Scholar's upstream 429 may require retrying later.
