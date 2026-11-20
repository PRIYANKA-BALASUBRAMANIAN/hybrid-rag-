# 📚 Multilingual Multimodal AI Knowledge Assistant

**Hybrid RAG over your own documents — PDFs, DOCX, TXT and images — with multilingual retrieval (English / Tamil / Hindi), evidence-grounded answers, and page-level citations.**

This is not "a chatbot". The assistant answers **only from the user's uploaded knowledge base**, shows **which document and page** every claim came from, and explicitly refuses when the answer is not in the files.

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Problem statement](#2-problem-statement)
3. [Motivation](#3-motivation)
4. [Features](#4-features)
5. [Architecture diagram](#5-architecture-diagram)
6. [Complete workflow](#6-complete-workflow)
7. [Technology stack](#7-technology-stack)
8. [Why each technology was selected](#8-why-each-technology-was-selected)
9. [Why BGE-M3](#9-why-bge-m3)
10. [Why Qdrant](#10-why-qdrant)
11. [Why RAG](#11-why-rag)
12. [Why hybrid retrieval](#12-why-hybrid-retrieval)
13. [Why reranking](#13-why-reranking)
14. [Multilingual support](#14-multilingual-support)
15. [Multimodal processing](#15-multimodal-processing)
16. [OCR](#16-ocr)
17. [LLM integration](#17-llm-integration)
18. [Installation](#18-installation)
19. [Environment variables](#19-environment-variables)
20. [Running locally](#20-running-locally)
21. [Running with Docker](#21-running-with-docker)
22. [API documentation](#22-api-documentation)
23. [Testing](#23-testing)
24. [Evaluation](#24-evaluation)
25. [Limitations](#25-limitations)
26. [Future improvements](#26-future-improvements)

---

## 1. Project overview

Users upload documents through a Streamlit sidebar. Every file goes through one pipeline:

```
extract → normalise → chunk → embed (BGE-M3) → index (Qdrant)
```

Questions asked in chat are answered by a FastAPI backend that runs:

```
hybrid retrieval → rerank → relevance gate → LLM (grounded prompt) → cited answer
```

Every answer carries expandable source cards: document name, page number, chunk id, relevance score, and the actual excerpt used as evidence.

## 2. Problem statement

Organisations and individuals accumulate knowledge locked inside PDFs, reports, scanned pages, screenshots of dashboards and documents in multiple languages. Generic LLM chatbots cannot answer questions about this private content: they were never trained on it, they hallucinate confidently when asked about it, and they cannot cite where an answer came from. Existing enterprise search tools handle keywords but not meaning, and almost none handle Indic scripts or image-based content.

**Goal:** a self-hostable assistant that answers natural-language questions — including cross-language ones — strictly from uploaded documents, with verifiable citations and honest refusal when evidence is missing.

## 3. Motivation

- **Hallucination is the #1 blocker** for real-world LLM adoption. Grounding answers in retrievable evidence plus refusing low-confidence answers directly attacks that problem.
- **Multilingual India context**: English-only retrieval fails for Tamil/Hindi users. One embedding space across languages makes cross-language Q&A possible ("English document ← Tamil question").
- **Documents are multimodal**: charts, scans and photos carry facts no text extractor can see. OCR + vision description turn them into searchable knowledge.
- As a placement project, every component uses established, interview-explainable technology — nothing is a black box.

## 4. Features

| Area | What you get |
|---|---|
| Ingestion | PDF (page-level), DOCX (headings + tables), TXT/MD, JPG/PNG images |
| Scanned PDFs | Image-heavy pages auto-detected and OCR'd (PaddleOCR) |
| Tables | Detected in PDFs/DOCX and stored as readable `a \| b \| c` rows |
| Chunking | Paragraph/sentence-aware, configurable size & overlap, never splits across pages |
| Embeddings | BGE-M3 dense (1024-d) **and** sparse lexical weights from one model |
| Storage | Qdrant collection with dense + sparse named vectors, metadata payload |
| Retrieval | Hybrid = dense semantic + sparse keyword fused by Reciprocal Rank Fusion |
| Reranking | BGE-reranker-v2-m3 cross-encoder picks top evidence chunks |
| Anti-hallucination | Score threshold gate; explicit refusal message; injection-hardened prompts |
| Citations | Document name + page number + chunk id + excerpt under each answer |
| Chat | History, follow-ups ("what dataset did *they* use?"), clear-conversation |
| Multilingual | EN/TA/HI documents and questions, incl. cross-language matches |
| Ops | FastAPI backend, Streamlit UI, Docker/compose, pytest suite (50 tests), evaluation script |

## 5. Architecture diagram

```
                        ┌────────────────────────────┐
                        │      STREAMLIT FRONTEND     │
                        │  chat · uploads · settings  │
                        └─────────────┬──────────────┘
                                      │ HTTP (httpx)
                        ┌─────────────▼──────────────┐
                        │       FASTAPI BACKEND       │
                        │  /api/documents  /api/chat  │
                        └──────┬───────────────┬─────┘
              ingestion        │               │        question
        ┌──────────────────────▼───┐      ┌────▼──────────────────┐
        │   DOCUMENT PROCESSORS    │      │   HYBRID RETRIEVER    │
        │  PyMuPDF · python-docx   │      │  dense + sparse (RRF) │
        │  txt · PaddleOCR · vision│      └────┬──────────────────┘
        └──────────────┬───────────┘           │ rerank (bge-reranker-v2-m3)
                       ▼                       ▼
                 TEXT CHUNKER            RELEVANCE GATE
                       │                       ▼
                       ▼                 GROUNDED PROMPT
                BGE-M3 EMBEDDER  ──►  QDRANT  ──►  LLM PROVIDER
             (dense 1024 + sparse)  (vectors+    (OpenAI-compatible /
                                     payload)    echo dev stub)
                                             │
                                    ANSWER + SOURCES
                                    (doc · page · excerpt)
```

Full module map: [ARCHITECTURE.md](ARCHITECTURE.md).

## 6. Complete workflow

**Ingestion** (`app/services/ingestion_service.py`)
1. Upload validated (extension, size) → routed to the right processor.
2. Processor extracts page-wise content; scanned PDF pages flagged (`needs_ocr`) and OCR'd when available.
3. Text normalised (line endings, zero-width chars, NFC), language tagged per chunk.
4. `TextChunker` packs paragraphs into ~600-token chunks (80-token overlap), preserving page numbers.
5. Chunks embedded via BGE-M3 (sqlite-cached by content hash).
6. Points upserted into Qdrant with deterministic ids → re-upload = overwrite, duplicate upload = detected via content hash and skipped.

**Question answering** (`app/rag/pipeline.py`)
1. Question embedded once (dense + sparse).
2. Qdrant fuses dense top-10 and sparse top-10 via server-side RRF → ~20 candidates.
3. Cross-encoder reranks candidates; top-4 kept.
4. If best rerank score < threshold → answer is the refusal message (+ closest matches shown). The LLM is **not called**.
5. Otherwise a fenced, injection-resistant prompt (context between markers, strict rules) goes to the configured LLM.
6. Answer returned with deduplicated source references and stage timings.

## 7. Technology stack

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.11+ (developed on 3.13) |
| Frontend | Streamlit | ≥ 1.38 |
| Backend | FastAPI + Uvicorn | ≥ 0.115 |
| PDF / DOCX | PyMuPDF / python-docx | ≥ 1.24 / ≥ 1.1.2 |
| Images | Pillow | ≥ 10.4 |
| OCR | PaddleOCR (optional install) | ≥ 3.0 |
| Embeddings | BAAI/**bge-m3** via FlagEmbedding | ≥ 1.3 |
| Reranker | BAAI/**bge-reranker-v2-m3** | — |
| Vector DB | Qdrant (local/server/memory modes) | client ≥ 1.12 |
| LLM | Any OpenAI-compatible API (OpenAI/Groq/OpenRouter/Ollama…) | — |
| Tests | pytest | ≥ 8.3 |
| Packaging | Docker + docker-compose | — |

## 8. Why each technology was selected

- **FastAPI** — typed request/response models (pydantic), automatic interactive docs, dependency injection for singletons (one loaded model per process). Industry-standard for ML serving.
- **Streamlit** — a full chat UI with file upload in ~250 lines; ideal for demos where the product is the pipeline, not the CSS.
- **PyMuPDF** — fastest reliable pure-Python-installable PDF parser; gives per-page text, image detection (scanned-page heuristic), table finder, and can render pages to PNG for OCR.
- **python-docx** — preserves heading structure and tables instead of dumping plain text.
- **PaddleOCR** — strong multilingual OCR with active development; installed separately because its wheels lag new Python versions, and the app degrades gracefully without it.
- **BGE-M3 / bge-reranker-v2-m3** — see §9 and §13.
- **Qdrant** — see §10.
- **pytest** — the de-facto Python standard; offline test doubles keep CI fast and free.

## 9. Why BGE-M3

1. **One model, 100+ languages** — English/Tamil/Hindi land in the same vector space, so a Tamil query retrieves relevant English passages and vice-versa (verified pattern in our gated integration tests).
2. **Dense AND sparse in one forward pass** — its learned lexical weights give keyword-style retrieval without maintaining a second BM25 stack; this powers true hybrid search.
3. **Quality** — strong published retrieval benchmarks (MIRACL multi-lingual, MS MARCO); 8192-token context comfortably covers chunk sizes.
4. **Practicality** — Apache-2.0 licence, 1024-dim vectors, runs acceptably on CPU for portfolio-scale corpora, first-class support in FlagEmbedding.

## 10. Why Qdrant

1. **Named dense + sparse vectors per point** — maps 1:1 to BGE-M3's dual output; hybrid fusion happens *inside* the database (RRF), not glued together in application code.
2. **Payload filtering** — delete one document's chunks by filter; metadata (doc id, source type, language, page) is first-class.
3. **Zero-infra local mode** — `QdrantClient(path=...)` persists to disk with no Docker; identical code talks to a Qdrant server in production (`QDRANT_MODE=server`). Great for students and for demos.
4. **Deterministic point IDs** (uuid5 of content hash + chunk index) make re-ingestion idempotent — duplicates overwrite instead of accumulating.

## 11. Why RAG

LLMs only know their training data. RAG (Retrieval-Augmented Generation) injects *retrieved, current, private* evidence into the prompt so the model:
- answers questions about content it has never seen,
- cites sources (trust + auditability),
- stays small/cheap — we retrieve instead of fine-tuning,
- can be corrected instantly by fixing documents, not retraining.

Fine-tuning was rejected deliberately: it cannot cite pages, it bakes in stale knowledge, and it needs data/resources a student doesn't have.

## 12. Why hybrid retrieval

| Strategy | Finds | Misses |
|---|---|---|
| Dense (semantic) | paraphrases, concepts, cross-language matches | rare identifiers, exact numbers ("FY24-Q2") |
| Sparse (keyword) | exact terms, codes, names, table cells | synonyms, translated phrasing |
| **Hybrid (RRF)** | both | — |

RRF (Reciprocal Rank Fusion) combines the two ranked lists using ranks, not raw scores — cosine similarity (0–1) and lexical weights (unbounded) are otherwise incomparable. Qdrant computes this server-side in one query.

## 13. Why reranking

Retrieval optimises **recall** (get the right chunk somewhere in the top 20). A bi-encoder compares the question to chunks *independently compressed* — fast but coarse. A **cross-encoder** (bge-reranker-v2-m3) reads question+chunk *together* and judges true relevance much more precisely. Feeding an LLM four sharp evidence chunks beats feeding it twenty mediocre ones: fewer tokens, less distraction, fewer hallucinations. Its sigmoid scores also power the refusal threshold.

## 14. Multilingual support

- Documents/questions supported: **English, Tamil, Hindi** (any language BGE-M3 handles benefits automatically).
- No forced translation anywhere: queries and chunks stay in their own language inside one shared embedding space.
- Per-chunk language detection (Unicode-block heuristic) is stored as metadata for filtering/analytics.
- Sentence splitting knows the Devanagari danda (।); chunk-size estimation assumes ~3 chars/token, conservative for Indic scripts.
- Honest limitations: PaddleOCR's accuracy on stylised Tamil fonts varies; LLM answer quality depends on the chosen provider's Tamil/Hindi fluency (documented, not hidden).

## 15. Multimodal processing

Images flow through two complementary extractors:
1. **OCR** recovers literal text (labels, totals, captions).
2. **Vision model** (optional, OpenAI-compatible VLM) produces a factual structured description — chart type, axes, series values, trends, or table transcription.

Both sections are stored with explicit markers (`[OCR text]`, `[Visual description]`) inside the chunk text, so the generator can attribute claims correctly ("the chart sales_chart.png shows…"). Scanned PDF pages reuse the same OCR runner after being rendered at 200 DPI.

## 16. OCR

PaddleOCR is loaded lazily once per process (models download on first use). Output shapes differ across PaddleOCR 2.x/3.x — both are normalised to plain text lines. When PaddleOCR isn't installed the system logs a warning and continues: text PDFs, DOCX, TXT work fully; images index filename-only until OCR/vision is available. This graceful degradation is deliberate — OCR must never take down ingestion.

## 17. LLM integration

`BaseLLMProvider` has two implementations:
- **OpenAICompatibleProvider** — works with OpenAI, Groq, OpenRouter, Together, Ollama/LM Studio (anything speaking `/chat/completions`). Switching providers = editing `.env`, zero code changes.
- **EchoProvider** — offline dev/test stub that returns the top retrieved excerpts clearly labelled `[dev echo mode]`. It exists so the whole pipeline is demonstrable and testable without any API key. It never fabricates content.

The system prompt enforces: context-only answers, no invention, exact refusal sentence, documents-as-data (injection defence), inline `[file | page N]` citations, concision, and marking inference as inference.

## 18. Installation

```powershell
# Windows PowerShell (adjust activate path on Linux/macOS)
git clone <your-repo-url> multimodal-knowledge-assistant
cd multimodal-knowledge-assistant

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

copy .env.example .env      # then edit .env (see §19)

# Optional: OCR (separate because paddlepaddle wheels lag some Python versions)
pip install -r requirements-ocr.txt
```

Model notes: BGE-M3 (~2.3GB) and the reranker (~1.1GB) download automatically from HuggingFace on first use and are cached locally.

## 19. Environment variables

All configuration lives in `.env` (see `.env.example` for the annotated list). Highlights:

| Variable | Purpose | Default |
|---|---|---|
| `QDRANT_MODE` | `memory` / `local_path` / `server` | `local_path` |
| `EMBEDDING_PROVIDER` | `bge_m3` or `hash` (offline test double) | `bge_m3` |
| `RERANKER_PROVIDER` / `RERANKER_ENABLED` | `bge` + on/off switch | `bge` / `true` |
| `RETRIEVAL_DENSE_TOP_K` / `_SPARSE_TOP_K` / `HYBRID_CANDIDATE_LIMIT` | retrieval sizes | 10 / 10 / 20 |
| `RERANK_TOP_K` | evidence chunks sent to LLM | 4 |
| `RELEVANCE_THRESHOLD` | refusal gate on rerank score | 0.30 |
| `CHUNK_SIZE_TOKENS` / `CHUNK_OVERLAP_TOKENS` | chunking | 600 / 80 |
| `LLM_PROVIDER` | `openai_compatible` or `echo` | `openai_compatible` |
| `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` | endpoint config | — / OpenAI / gpt-4o-mini |
| `VISION_*` | image understanding (falls back to `LLM_*`) | — |
| `OCR_ENABLED` / `OCR_LANGUAGES` | OCR toggle/language | true / en |

Secrets are read only in `app/config.py`; `.env` is git-ignored.

## 20. Running locally

```powershell
# Terminal 1 — backend
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000

# Terminal 2 — frontend
.venv\Scripts\python.exe -m streamlit run frontend/streamlit_app.py
```

Open http://localhost:8501. Generate demo files and ingest them:

```powershell
.venv\Scripts\python.exe scripts\generate_sample_data.py
.venv\Scripts\python.exe scripts\ingest_samples.py
```

CPU-friendly mode: set `RERANKER_ENABLED=false` (skips the cross-encoder) — everything still works; the relevance gate falls back to "no candidates found" checks and says so in a warning banner.

## 21. Running with Docker

```bash
docker compose up --build
# frontend http://localhost:8501 · API docs http://localhost:8000/docs
```

Compose starts three services: `qdrant` (persistent volume), `backend` (CPU-torch image), `frontend`. The compose environment pins `QDRANT_MODE=server` / `QDRANT_URL=http://qdrant:6333`.

Deployment options beyond a laptop: any VM with Docker (add a reverse proxy + TLS), or managed Qdrant Cloud + container hosting for the API/UI. The stateless API scales horizontally; models load per-instance.

## 22. API documentation

Interactive OpenAPI docs at `/docs` when the backend runs.

| Method & path | Body / params | Returns |
|---|---|---|
| `POST /api/documents/upload` | multipart `file` (pdf/docx/txt/md/jpg/jpeg/png) | `UploadResponse` (doc info, `duplicate`, `ocr_used`, warnings) |
| `GET /api/documents` | — | `list[DocumentInfo]` |
| `DELETE /api/documents/{id}` | — | `DeleteResponse` (chunks deleted) |
| `POST /api/chat` | `{question, history[], overrides{dense_top_k?, candidate_limit?, rerank_top_k?, min_relevance?}}` | `AnswerResponse` |
| `GET /api/documents/health` | — | component status + indexed points |

Example:

```bash
curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" \
  -d '{"question":"Which year had the highest revenue?"}'
```

```json
{
  "answer": "2024 had the highest revenue at 14 million dollars [company_handbook.pdf | page 2].",
  "sources": [{"document_name": "company_handbook.pdf", "page_number": 2,
               "chunk_id": "..._0001", "excerpt": "Revenue in fiscal year 2024 reached...", "score": 0.87}],
  "insufficient_evidence": false,
  "timings_ms": {"retrieval_ms": 41.2, "rerank_ms": 980.5, "generation_ms": 1210.8}
}
```

The chat API is stateless: clients send recent turns in `history` so follow-up pronouns resolve while servers scale horizontally.

## 23. Testing

```powershell
pytest                      # 50 fast offline tests (hash embedder, memory Qdrant, echo LLM)
set RUN_MODEL_TESTS=1 && pytest -m model   # real BGE-M3/reranker/OCR integration tests
```

Coverage highlights: text normalisation & language detection; chunking bounds/overlap/page-attribution/multilingual tagging; PDF/DOCX/TXT/image extraction (fixtures generated programmatically); scanned-page OCR hook; hash-embedder determinism; Qdrant round-trips incl. duplicate overwrite & deletion; reranker contract; citation dedup/truncation; system-prompt rule completeness; end-to-end API flows (upload→cited answer→delete→refusal).

Test doubles are clearly labelled engineering tools (`hash` provider, `NoopReranker`, `EchoProvider`) — they exercise *plumbing*, and every claim about real-model behaviour is gated behind `RUN_MODEL_TESTS`.

## 24. Evaluation

`data/evaluation/questions.json` defines 10 questions (EN/TA/HI) with expected source documents/pages. `scripts/evaluate_retrieval.py` ingests the sample corpus into an ephemeral collection and measures, live:

- **Recall@k** — expected document retrieved in top-k,
- **MRR** — rank of the first correct chunk,
- **Citation accuracy@1** — top-1 citation matches expected doc/page.

Unreadable inputs (e.g., chart PNG without OCR/vision) are reported as SKIPPED, never guessed. Reports save to `data/evaluation/latest_report.json` with provider/config stamped.

Reference run on this machine (`--provider hash --k 5 --no-rerank`, tiny corpus): recall@5 = 1.0, MRR = 0.95, citation@1 = 0.9. These numbers characterise the *offline harness*, not production quality — run `--provider bge_m3` (plus reranker) yourself and record those results; that is exactly why the script prints its config header.

## 25. Limitations

- **OCR coverage**: PaddleOCR excels on printed English; stylised/low-quality Tamil/Hindi scans vary. Handwriting is out of scope.
- **Tables**: detected rows become pipe-separated text; complex merged-cell layouts lose structure. No claim of perfect table understanding.
- **Vision descriptions depend on the configured VLM**; without one, chart understanding reduces to OCR'd labels.
- **Relevance gate needs the reranker**; CPU-fast mode disables calibrated refusals (warned in UI).
- **Chunk token counts are estimates** (chars-per-token ratio), tuned conservatively rather than tokenizer-exact.
- **DOCX/TXT have no page numbers** — citations show document + chunk instead of inventing pages.
- **First-run latency**: model downloads + loads dominate the first question; later requests are seconds.
- **EchoProvider is not an LLM** — it exists for offline testing/demos only.

## 26. Future improvements

1. ColBERT-style late-interaction vectors (BGE-M3 supports them) for finer matching.
2. Query rewriting step (LLM rewrites follow-ups into standalone questions before retrieval).
3. Streaming responses (SSE) and async batch ingestion workers.
4. Per-user collections with auth (multi-tenant payload filtering).
5. Layout-aware PDF parsing (reading order, nested tables) e.g. Docling/unstructured.io comparison.
6. Local VLM option (e.g. Qwen-VL via Ollama) for air-gapped deployments.
7. Larger evaluation set + statistical significance reporting; golden-answer generation grading with an judge model.
8. Hybrid BM25 fallback independent of the embedding model, for domain jargon the sparse head wasn't trained on.

---

See also: [ARCHITECTURE.md](ARCHITECTURE.md) · [INTERVIEW_GUIDE.md](INTERVIEW_GUIDE.md) · [RESUME_DESCRIPTION.md](RESUME_DESCRIPTION.md)
