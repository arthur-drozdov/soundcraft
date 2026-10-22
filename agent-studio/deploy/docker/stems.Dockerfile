# stems worker: builds the stems.cpp CLI from source (build.sh) and installs
# the agent-studio worker runtime. CUDA build by default; set
# --build-arg STEMS_BACKEND=cpu for a CPU-only image on a plain base.
# Build from the agent-studio folder:
#   docker build -f deploy/docker/stems.Dockerfile -t forge-stems .
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3
FROM ${BASE_IMAGE} AS build
ARG STEMS_BACKEND=cuda
ARG STEMS_REF=
ARG STEMS_MODELS=htdemucs
RUN git clone --recurse-submodules https://github.com/betweentwomidnights/stems.cpp /opt/stems.cpp \
    && if [ -n "$STEMS_REF" ]; then git -C /opt/stems.cpp checkout "$STEMS_REF" \
       && git -C /opt/stems.cpp submodule update --init --recursive; fi
WORKDIR /opt/stems.cpp
RUN ./build.sh "$STEMS_BACKEND"
RUN ./models.sh $STEMS_MODELS
RUN install -d /out/bin /out/models \
    && install -m 0755 "$(find /opt/stems.cpp -type f -name stems-split -perm -u+x | head -n 1)" /out/bin/stems-split \
    && cp models/*.gguf /out/models/

FROM ${BASE_IMAGE}
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY --from=build /out/bin/stems-split /opt/forge/bin/stems-split
COPY --from=build /out/models /opt/forge/models/stems
WORKDIR /opt/agent-studio
COPY pyproject.toml README.md* ./
COPY src ./src
RUN pip install .
# families.stems.options: binary=/opt/forge/bin/stems-split, model=/opt/forge/models/stems/<file>.gguf
ENTRYPOINT ["python", "-m", "forge.workers.stems"]
