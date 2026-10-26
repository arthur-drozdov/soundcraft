# The supported container: ONE image with the gateway plus one virtual
# environment per family. The gateway spawns workers as subprocesses, so all
# of them must live in the same container. Environments are created with
# --system-site-packages (deploy/scripts/setup-venvs.sh) so the base image's
# torch is shared and never replaced.
#
# Build context: the REPOSITORY ROOT (compose.yaml does this):
#   docker build -f agent-studio/deploy/docker/forge.Dockerfile -t forge .
# Requires `docker login nvcr.io` (user $oauthtoken, password = NGC API key).
#
# Build args:
#   BASE_IMAGE     NGC PyTorch image (aarch64, CUDA 13). 25.11-py3 is the tag the
#                  Arm learning path uses for the Spark; check the NGC catalog
#                  for a newer one.
#   WITH_RENDER    1 = compile soundcraft-cli in a Rust stage (slow), 0 = skip
#                  and mount a binary at /opt/forge/bin/soundcraft-cli instead.
#   WITH_STEMS     1 = build stems.cpp, 0 = skip.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3
ARG WITH_RENDER=1

FROM rust:1.99-bookworm AS render-1
RUN apt-get update && apt-get install -y --no-install-recommends pkg-config libasound2-dev libssl-dev cmake \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cargo build --release -p soundcraft-cli \
    && install -d /out && install -m 0755 target/release/soundcraft-cli /out/soundcraft-cli

FROM debian:bookworm-slim AS render-0
RUN mkdir -p /out

FROM render-${WITH_RENDER} AS render

FROM ${BASE_IMAGE}
ARG WITH_STEMS=1
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 FORGE_ROOT=/opt/forge SYSTEM_SITE=1 \
    PATH=/opt/forge/venvs/gateway/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends git cmake libsndfile1 ffmpeg libasound2t64 \
    && rm -rf /var/lib/apt/lists/*

COPY agent-studio /opt/agent-studio
COPY --from=render /out/ /opt/forge/bin/
RUN if [ "$WITH_STEMS" = "1" ]; then STEMS=stems; else STEMS=; fi \
    && /opt/agent-studio/deploy/scripts/setup-venvs.sh gateway foundation sa3 acestep basicpitch $STEMS \
    && rm -rf /opt/forge/src/*/.git /root/.cache

COPY agent-studio/deploy/forge.container.yaml /etc/forge/forge.yaml
RUN useradd --system --create-home --home-dir /data forge && install -d -o forge -g forge /data /models
VOLUME ["/data", "/models"]
USER forge
EXPOSE 7700
ENV FORGE_CONFIG=/etc/forge/forge.yaml
ENTRYPOINT ["forge"]
CMD ["serve"]
