# Architecture

This document maps every module to its responsibility and explains the data flow in detail. Pair it with README §5–6.

## Module map

```
app/
├── config.py                  # ONLY place env vars are read (Settings dataclass, cached)
├── logging_config.py          # structured console logging; never logs secrets/content
├── schemas.py                 # pydantic models shared by all layers (single source of truth)
├── main.py                    # FastAPI factory: CORS, routers, lifespan warm-up
│
├── api/
│   ├── routes_documents.py    # upload / list / delete / health endpoints
│   └── routes_chat.py         # POST /api/chat (stateless; history from client)
│
├── services/                  # composition roots (thin routes, reusable logic)
│   ├── ingestion_service.py   # bytes → processor → chunker → embedder → store
│   └── chat_service.py        # cached RAG pipeline assembly
│
├── ingestion/
│   ├── base.py                # processor protocol + result builder (content-hash ids)
│   ├── pdf_processor.py       # PyMuPDF pages, scanned-page detection, tables, OCR hook
│   ├── docx_processor.py      # headings ("## ") + pipe-row tables
│   ├── text_processor.py      # txt/md with encoding fallbacks
│   ├── image_processor.py     # OCR + vision sections, PIL validation
│   └── chunker.py             # paragraph/sentence packing, overlap, page integrity
│
├── embeddings/
│   ├── base.py                # BaseEmbedder interface (dense + sparse)
│   ├── bge_m3.py              # real model, lazy singleton, lexical-weight pruning
│   ├── hashing.py             # offline test double (feature hashing)
│   ├── cache.py               # sqlite cache keyed by provider+model+text hash
│   └── factory.py             # CachedEmbedder wrapper + build_embedder()
│
├── database/
│   └── qdrant_client.py       # collection mgmt, upsert, dense/sparse/hybrid(RRF), delete
│
├── retrieval/
│   ├── hybrid_search.py       # HybridRetriever (embed once → fused candidates)
│   ├── vector_search.py       # dense-only ablation helper
│   └── reranker.py            # NoopReranker / BGECrossEncoderReranker
│
├── rag/
│   ├── prompts.py             # system rules, fenced context blocks, refusal constant
│   ├── providers.py           # OpenAICompatibleProvider / EchoProvider
│   ├── pipeline.py            # retrieve → rerank → gate → generate (timings logged)
│   └── citation.py            # dedup by (doc,page), excerpt building
│
├── vision/
│   ├── ocr.py                 # lazy PaddleOCR wrapper, graceful absence
│   └── vision_model.py        # OpenAI-compatible VLM describer (+ downscaling)
│
└── utils/                     # text normalisation, language detection, stable ids, files
```

## Data contracts

- `DocumentResult` — pages + provenance before chunking (`doc_hash` = sha256 of normalised full text).
- `IngestedChunk` = `ChunkMetadata` (doc id/name/type/page/chunk id/language) + `text`.
- `EmbeddingResult` — `dense: [float]`, `sparse: {token_id: weight}`.
- `RetrievedChunk` — payload fields + `score` + per-stage scores (`stage_scores["hybrid"]`).
- `AnswerResponse` — answer, `sources[]`, `insufficient_evidence`, timings, warnings.

## Key design decisions & their WHYs

| Decision | Rationale |
|---|---|
| Deterministic point ids `uuid5(doc_hash, chunk_index)` | re-ingest overwrites; no vector duplicates |
| Content-hash duplicate detection pre-embedding | saves minutes of CPU on repeat uploads |
| Chunks never span PDF pages | page citations stay exactly truthful |
| Server-side RRF in Qdrant | one query; rank-based fusion immune to score-scale mismatch |
| Relevance gate BEFORE the LLM call | unsupported answers are never generated at all |
| Stateless chat API | horizontal scaling; UI owns conversation memory |
| Providers behind interfaces (LLM/embedder/reranker) | swap models via .env; test doubles for CI |
| OCR/vision as injected callables | processors unit-testable without heavy deps |

## Failure handling

- Ingestion errors → HTTP 400 with human-readable reason; unexpected errors → 500 without leaking internals.
- LLM call failure → refusal-style response with nearest sources + warning banner (never a raw stack trace to users).
- OCR/vision unavailable → warnings attached to the upload response, ingestion succeeds.

## Performance notes

- All heavy singletons (embedder, reranker, store, pipeline) are process-cached via `lru_cache` factories.
- Embedding cache (sqlite) makes re-indexing near-instant for unchanged content.
- Batch embedding (`EMBEDDING_BATCH_SIZE`) and bounded candidate counts keep CPU latency predictable.
