# sa3 worker environment. Upstream is installed with uv as its README says
# ("git clone" then "uv sync"), with --inexact so extra packages stay, and with
# the base image's torch excluded from the sync: the NGC torch build is the one
# that supports aarch64 + Blackwell (SM121).
# Build from the agent-studio folder:
#   docker build -f deploy/docker/sa3.Dockerfile -t forge-sa3 .
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN python - >/etc/pip-constraints.txt <<'PY'
from importlib import metadata
for n in ("torch", "torchaudio", "torchvision", "triton", "numpy"):
    try:
        print(f"{n}=={metadata.version(n)}")
    except metadata.PackageNotFoundError:
        pass
PY
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt
# A venv sharing the base image's torch; uv must not touch the system prefix.
RUN python -m venv --system-site-packages /opt/venv
ENV PATH=/opt/venv/bin:$PATH UV_PROJECT_ENVIRONMENT=/opt/venv
RUN pip install uv

ARG SA3_REF=
ARG SA3_UV_ARGS=
RUN git clone --recurse-submodules https://github.com/Stability-AI/stable-audio-3 /opt/sa3-upstream \
    && if [ -n "$SA3_REF" ]; then git -C /opt/sa3-upstream checkout "$SA3_REF"; fi
RUN cd /opt/sa3-upstream \
    && uv sync --inexact --no-install-package torch --no-install-package torchaudio \
       --no-install-package torchvision --no-install-package triton $SA3_UV_ARGS

WORKDIR /opt/agent-studio
COPY pyproject.toml README.md* ./
COPY src ./src
RUN pip install .

ENTRYPOINT ["python", "-m", "forge.workers.sa3"]
