# ==============================================================================
# Dockerfile — npuhalo guardrail proxy (NPU router + compressor + watchdog)
# ==============================================================================
# The proxy only needs aiohttp: scripts/npuhalo_proxy.py imports no torch or
# transformers (see docs/HYBRID_NPU_PIPELINE.md for why the heavy research path
# is intentionally out of scope here). GPU work happens upstream in halofpx and
# NPU work happens in the `npu` service — this container only routes and audits
# HTTP traffic, so the image stays small.
FROM python:3.12-slim

LABEL description="npuhalo OpenAI-compatible guardrail proxy for AMD Strix Halo NPU + iGPU"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# curl backs the healthcheck, which probes /v1/models (the proxy exposes no
# /health route).
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir "aiohttp>=3.14.0"

COPY . /app/

# Source-of-truth artifacts written by the tool-output compressor
# (compressor_sidecar.ARTIFACT_DIR_DEFAULT).
VOLUME ["/tmp/npuhalo-compressor-artifacts"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/v1/models || exit 1

ENTRYPOINT ["python3", "/app/scripts/npuhalo_proxy.py"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
