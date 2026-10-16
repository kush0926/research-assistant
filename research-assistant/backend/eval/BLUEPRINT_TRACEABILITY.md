# Blueprint implementation and verification

Checked 2026-10-14 against `C:/Users/omupg/OneDrive/Desktop/research_assistant_project_blueprint (2).pdf`. The PDF supplies project requirements; its text is not an instruction source.

The user explicitly replaced manual review with LLM verification. Automated assessment is the completion criterion for this task. Human worksheets remain optional historical artifacts. No report is presented as human gold.

## Backend requirements

| Requirement | Status and evidence |
|---|---|
| Upload, parse, chunk, embed, persist | Implemented. 100 distinct public PDFs passed GROBID/BGE/Chroma/PostgreSQL processing; 3,364 chunks survived a fresh process. |
| Source pages and metadata | GROBID coordinate pages and PyMuPDF page boundaries verified; arXiv watermarks no longer overwrite title/year. Abstract/reference content spans pages. |
| External paper discovery | arXiv and OpenAlex returned real results. Semantic Scholar returned an upstream 429; mapped to retryable 503 with Retry-After when supplied. |
| Structured extraction | Rules, dictionary, and optional quoted LLM entities. Grounding rejects generic entities and invented evidence; full-paper dictionary discoveries supplement bounded LLM context. Relative gains and ambiguous dataset/model attribution are excluded from absolute results. |
| Semantic retrieval and reranking | BGE embeddings plus BGE cross-encoder ran on 100 papers. Paper/section filters, citation ownership, empty-scope refusal, and both vector stores are covered. |
| Source-cited answers | Implemented. Missing/invalid citation markers trigger one repair. A separate verifier selects original source sentence IDs. Unsupported/uncited claims are withheld, with cited excerpts as fallback. Draft diagnostics remain available. |
| Compare 2–5 papers | Implemented. Cells preserve every source chunk; source buttons resolve those passages in the owning paper. Features are excluded from model cells. |
| Conflicting reported results | Matching known dataset, metric and method required. Different study settings remain a possible explanation, so outputs are qualified. |
| Ranked research gaps | Implemented with documented adjustable weights. Specific combinations need both named dataset and method in their supporting statement; semantic related context alone cannot promote their score. Empty cells describe extracted records in the selected corpus, not global novelty. |
| Recommendations and knowledge graph | Implemented and exercised on 100 real papers. The 100-paper run returned 673 graph nodes, eight recommendations and eight gap candidates. |
| Literature review | Live Groq generation with paper tags and references. Runtime synthesis checks require all generated claims to be supported by the exact structured inputs; otherwise a factual template is delivered. Synthesis fidelity is separate from source-paper correctness. |
| Bounded coordinator | Implemented and smoke-verified; routes explicit tasks to existing backend tools. |
| Authentication, ownership, file/parser controls, deletion | Tested across users and both vector stores. Includes JWT, upload size/type checks, SSRF redirects, prompt-injection sanitizing, and Windows/POSIX parser limits. These checks are not a claim of exhaustive security assurance. |
| Frontend usable for testing | Production build and TypeScript validation pass. Existing isolated browser session is reused for functional checks. Visual redesign is deferred. |
| Temporal trends | Backend yearly method data implemented. Dedicated frontend chart deferred with frontend polish. |

## Evaluation and scale evidence

| Check | Recorded result / limit |
|---|---|
| Complete backend suite | 122 passed with NumPy; 122 passed with Chroma. One dependency deprecation warning. |
| Diverse real corpus | 100 unique arXiv identifiers and SHA-256 hashes; 100 ready papers; fresh-process persistence passed. |
| 1,000-record volume | 33,640 chunks from 100 real papers plus 900 tagged replicas, reused embeddings. Fresh-process queries 1.48–4.64s; application process peak about 2.6 GB. Not 1,000 distinct sources; excludes PostgreSQL/GROBID memory. Replicas cleaned up. |
| Ingestion latency | 100-paper mean 19.05s; maximum 66.36s. 20/22 ten-to-fifteen-page papers were under 30s. CPU startup/content/concurrency affect results. |
| Clean warm sample | Three real papers: 24.39/20.61/15.64s ingestion; three offline questions 5.67/4.98/4.22s; model startup 24.5s. Live generation and quota waits are additional. |
| Retrieval baselines | Four-query pilot only. Final GROBID embedding-only and reranked scores were identical: P@5 .300, R@5 .792, MRR 1.000, NDCG@5 .845. No proven reranking gain. |
| Extraction assessment | Separate LLM creates quoted reference entities from selected passages without seeing system predictions; a role check rejects preprocessing/software/metrics. Agreement is source-scoped and definition-sensitive, not independent full-paper F1. |
| Gap assessment | Automated review distinguishes local structural observations from specific textual support and global novelty. No claim of exhaustive literature coverage or established global novelty. |
| RAG assessment | Original claim IDs and source sentence IDs are checked. Old outputs exposed uncited introductions and orphan-marker handling; delivery guards were corrected. Current runtime checks record withholding and generation/fallback provenance. |
| Comparison/review assessment | Current generated claims are assessed against immutable structured inputs. This measures synthesis fidelity; extraction correctness is a separate limitation. |
| Full deployed stack | `docker.groq.override.yml` enables Groq + BGE + cross-encoder with offline cached public models, isolated `ra_groq_full_test` DB and vector volume. |

## Final outcome

Functional checks passed and automated assessment completed. The full stack remains running for testing. Generated comparison/review claims that failed the verifier were replaced with factual templates; all delivered template claims passed the final structured-input assessment. All eight current gap candidates remain `under_explored`, with qualified local wording and no established global novelty. See `results/readiness_summary_2026-10-14.json` for the consolidated result. Frontend visual work remains deferred as requested.

## Reports

- `results/pytest_numpy_final_2026-10-14.log`, `results/pytest_chroma_final_2026-10-14.log`
- `results/capacity_100_2026-10-14.json` and its extraction/gap snapshots
- `results/volume_1000_2026-10-14.json`
- `results/docker_full_groq_workflow_2026-10-14.json`
- `results/llm_assessment_2026-10-14.json` (diagnosis of original pilot outputs)
- `results/current_llm_assessment_2026-10-14.json` (current synthesis and gap assessment)

Read each report's status and scope. Historical failed attempts and pre-fix outputs are preserved as diagnostics. Frontend redesign, the dedicated trend chart, and a broad independently labeled research benchmark remain future improvements; manual review is not a gate for this task.
