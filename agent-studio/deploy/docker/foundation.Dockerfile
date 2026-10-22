# Foundation worker environment (RC Stable Audio Tools). Building block for
# split deployments; see forge.Dockerfile for the supported all-in-one image.
# Build from the agent-studio folder:
#   docker build -f deploy/docker/foundation.Dockerfile -t forge-foundation .
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# Keep the base image's torch stack; refuse to replace it.
RUN python - >/etc/pip-constraints.txt <<'PY'
from importlib import metadata
for n in ("torch", "torchaudio", "torchvision", "triton", "numpy"):
    try:
        print(f"{n}=={metadata.version(n)}")
    except metadata.PackageNotFoundError:
        pass
PY
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt

ARG FOUNDATION_REF=
RUN git clone --recurse-submodules https://github.com/RoyalCities/RC-stable-audio-tools /opt/rc-stable-audio-tools \
    && if [ -n "$FOUNDATION_REF" ]; then git -C /opt/rc-stable-audio-tools checkout "$FOUNDATION_REF"; fi \
    && pip install /opt/rc-stable-audio-tools

# agent-studio without extras: the worker runtime only needs the core deps.
WORKDIR /opt/agent-studio
COPY pyproject.toml README.md* ./
COPY src ./src
RUN pip install .

ENTRYPOINT ["python", "-m", "forge.workers.foundation"]
