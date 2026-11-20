# =============================================================================
# Backend (+ shared code) container image.
#
# Notes on size: torch/FlagEmbedding/pymupdf make any RAG image inherently
# large (~4-6GB). We keep it as small as practical:
#   - slim base image, no build tools needed (wheels exist for cp311)
#   - CPU-only torch wheel (avoids ~2.5GB of NVIDIA/CUDA dependencies)
#   - paddleocr is NOT installed here (see requirements-ocr.txt) to keep the
#     default image leaner; OCR degrades gracefully at runtime.
# =============================================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Runtime libraries: libgomp1 (torch/scipy), libglib/libgl (Pillow/image ops)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libglib2.0-0 libgl1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/app

# Install CPU-only torch first from its own index, then everything else.
COPY requirements.txt .
RUN pip install --no-cache-dir torch>=2.6.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY app ./app
COPY frontend ./frontend
COPY scripts ./scripts
COPY pytest.ini ./

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=5 \
    CMD curl -fsS http://localhost:8000/ || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
