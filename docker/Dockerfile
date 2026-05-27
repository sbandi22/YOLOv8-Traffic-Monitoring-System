# syntax=docker/dockerfile:1.6
# =============================================================================
# YOLOv8 Traffic Monitoring System - Production Image
# Supports CPU and CUDA (build with: docker build --build-arg BASE=cuda ...)
# =============================================================================

ARG BASE=cpu

# ---------- CPU stage ----------
FROM python:3.10-slim AS base-cpu
ENV TORCH_INDEX=https://download.pytorch.org/whl/cpu

# ---------- CUDA stage ----------
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base-cuda
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3-pip python3.10-venv && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*
ENV TORCH_INDEX=https://download.pytorch.org/whl/cu121

# ---------- Final stage ----------
FROM base-${BASE} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps for OpenCV / video
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 curl tini && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --extra-index-url ${TORCH_INDEX} -r requirements.txt

# App source
COPY . /app

# Non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/app/docker/entrypoint.sh"]
CMD ["api"]
