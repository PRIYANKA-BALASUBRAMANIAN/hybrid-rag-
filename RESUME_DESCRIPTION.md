# Resume Description

## Project title

**Multilingual Multimodal AI Knowledge Assistant — Hybrid RAG with Evidence-Grounded Retrieval**
*(Python · FastAPI · Streamlit · BGE-M3 · Qdrant · PaddleOCR)*

## Short project description (for the "projects" section)

Built a self-hostable question-answering system over private documents (PDF/DOCX/TXT/images, incl. scanned pages and charts) that answers in English, Tamil or Hindi using Retrieval-Augmented Generation: multilingual BGE-M3 embeddings power hybrid dense+sparse retrieval fused server-side in Qdrant (RRF), a cross-encoder reranks evidence, a configurable OpenAI-compatible LLM generates grounded answers, and every claim is cited with document name, page number and verifiable excerpt — with an explicit refusal path when evidence is missing. Ships as a FastAPI backend + Streamlit UI with Docker Compose, 50 automated tests, and a live retrieval-evaluation harness (Recall@k / MRR / citation accuracy).

## Resume bullet points

- Engineered an end-to-end multimodal RAG pipeline (PyMuPDF/DOCX/OCR ingestion → paragraph-aware chunking → BGE-M3 dense+sparse embeddings → Qdrant) that answers questions over uploaded documents with page-level citations and cross-language retrieval across English, Tamil and Hindi.
- Implemented hybrid retrieval fusing semantic and lexical search via Reciprocal Rank Fusion plus a bge-reranker-v2-m3 second stage, cutting LLM context to top-k evidence and enforcing a calibrated relevance-threshold gate that refuses unsupported answers to control hallucination.
- Delivered a production-style codebase — FastAPI backend, Streamlit chat UI, pluggable LLM/embedder/reranker providers configured via environment variables, Docker Compose deployment, sqlite embedding cache and content-hash duplicate detection — validated by a 50-test pytest suite and a reproducible evaluation script reporting measured Recall@k/MRR/citation accuracy.

## Technology list

Python 3.11+ · FastAPI · Uvicorn · Streamlit · Pydantic · PyMuPDF · python-docx · Pillow · PaddleOCR · FlagEmbedding (BAAI/bge-m3, bge-reranker-v2-m3) · PyTorch (CPU) · Qdrant (local/server modes; named dense+sparse vectors; RRF) · OpenAI-compatible LLM APIs (OpenAI/Groq/Ollama) · httpx · python-dotenv · SQLite caching · pytest · Docker & docker-compose.

## Notes on honesty (read before interviews)

- Do **not** quote accuracy percentages unless you ran `scripts/evaluate_retrieval.py` yourself and recorded the output — it prints exactly what was measured and under which configuration.
- The reference numbers currently in the README (recall@5 = 1.0 etc.) are from the offline `hash`-provider harness run on a tiny sample corpus; they demonstrate the evaluation *harness*, not production quality. Run `--provider bge_m3` with the reranker enabled and replace them with your own results if you want headline metrics.
