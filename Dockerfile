# ── Stage 1: dependency builder ─────────────────────────────────────────────
# Installs all Python packages (including heavy torch/transformers) into
# /root/.local so we can copy only that layer into the final image.
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile certain packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .

# Install CPU-only torch first (much smaller than CUDA build)
RUN pip install --user --no-cache-dir \
    torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu

# Install remaining app dependencies
RUN pip install --user --no-cache-dir \
    fastapi==0.111.0 \
    "uvicorn[standard]==0.29.0" \
    qdrant-client==1.9.1 \
    sentence-transformers==2.7.0 \
    openai==1.30.0 \
    prometheus-client==0.20.0 \
    mlflow==2.13.2 \
    pydantic-settings==2.2.1 \
    httpx==0.27.0 \
    python-json-logger==2.0.7 \
    tqdm==4.66.4 \
    pyyaml==6.0.1 \
    requests==2.32.2

# Pre-download the embedding model so startup is instant
# This bakes the model into the image (~90 MB) - a deliberate trade-off:
# larger image, zero download at runtime, deterministic behaviour.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user - principle of least privilege
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH
# Disable tokeniser fork-safety warning on macOS hosts
ENV TOKENIZERS_PARALLELISM=false
# Sentence-transformers caches models here; builder already downloaded them
ENV SENTENCE_TRANSFORMERS_HOME=/home/appuser/.local/share/sentence-transformers
ENV HF_HOME=/home/appuser/.local/share/huggingface

# Copy application source
COPY app/ ./app/

RUN chown -R appuser:appuser /app /home/appuser/.local
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
