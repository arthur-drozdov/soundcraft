# Basic Pitch worker (CPU). No GPU stack needed, so a slim image is enough.
# Build from the agent-studio folder:
#   docker build -f deploy/docker/basicpitch.Dockerfile -t forge-basicpitch .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
# On Linux basic-pitch defaults to the TFLite backend. If no aarch64 wheel of
# its runtime is available for your Python, `forge check` reports the model as
# unavailable with the import error; see deploy/README.md (troubleshooting).
RUN pip install basic-pitch

WORKDIR /opt/agent-studio
COPY pyproject.toml README.md* ./
COPY src ./src
RUN pip install .

ENTRYPOINT ["python", "-m", "forge.workers.basicpitch"]
