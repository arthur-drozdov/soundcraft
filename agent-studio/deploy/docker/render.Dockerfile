# render worker: builds headless soundcraft-cli for aarch64 from this repo
# using its Rust toolchain (rust-version 1.90), then ships it with the worker
# runtime. Needs the REPOSITORY ROOT as build context:
#   docker build -f agent-studio/deploy/docker/render.Dockerfile -t forge-render .
#
# Alternative without compiling: download the released linux-aarch64 tarball,
# put soundcraft-cli at /opt/forge/bin/soundcraft-cli and skip the build stage
# (see deploy/README.md, "Render worker").
FROM rust:1.90-bookworm AS build
RUN apt-get update && apt-get install -y --no-install-recommends pkg-config libasound2-dev libssl-dev cmake \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cargo build --release -p soundcraft-cli

FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libasound2 libssl3 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/soundcraft-cli /opt/forge/bin/soundcraft-cli
WORKDIR /opt/agent-studio
COPY agent-studio/pyproject.toml ./
COPY agent-studio/src ./src
RUN pip install .
# families.render.options.binary=/opt/forge/bin/soundcraft-cli
ENTRYPOINT ["python", "-m", "forge.workers.render"]
