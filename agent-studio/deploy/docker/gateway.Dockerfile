# Gateway only (no models). Optional building block for operators who split
# the deployment; the gateway spawns workers as local subprocesses, so a split
# gateway needs families.<name>.python pointing at interpreters in the SAME
# container. For the supported all-in-one topology use forge.Dockerfile.
#
# Build from the agent-studio folder:
#   docker build -f deploy/docker/gateway.Dockerfile -t forge-gateway .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /opt/agent-studio
COPY pyproject.toml README.md* ./
COPY src ./src
RUN pip install ".[forge]" && useradd --system --create-home --home-dir /data forge \
    && install -d -o forge -g forge /data
USER forge
EXPOSE 7700
ENV FORGE_CONFIG=/etc/forge/forge.yaml
ENTRYPOINT ["forge", "serve"]
