#!/usr/bin/env bash
set -euo pipefail
# Creates the per-family Python environments for Forge and installs the
# upstream model packages into them. Safe to re-run: existing environments are
# reused, packages are upgraded only when a *_REF variable changes, and source
# checkouts are fetched, not re-cloned.
#
# Usage: setup-venvs.sh [family ...]
#   families: gateway foundation sa3 acestep basicpitch stems render
#   default:  all of them (render is skipped with a message when no
#             soundcraft-cli binary or Rust toolchain is available)
#
# Environment (all optional):
#   FORGE_ROOT        install root                        (default /opt/forge)
#   PYTHON            interpreter used to create venvs    (default python3)
#   SYSTEM_SITE       1 = venvs see the base image's site-packages so its
#                     torch is shared (NGC PyTorch image); 0 = isolated
#                     (default: 1 when `import torch` works in $PYTHON)
#   FOUNDATION_REF, SA3_REF, ACESTEP_REF, STEMS_REF   git refs (default: remote default branch)
#   SA3_UV_ARGS       extra args for `uv sync` in the SA3 checkout
#   ACESTEP_UV_ARGS   extra args for `uv sync` in the ACE-Step checkout
#   STEMS_BACKEND     cpu | cuda | vulkan                  (default cuda if nvcc exists, else cpu)
#   STEMS_MODELS      stems.cpp model names for models.sh (default htdemucs)
#   SOUNDCRAFT_CLI_BINARY  path of a prebuilt aarch64 soundcraft-cli to install
#   SOUNDCRAFT_REPO   repo root to build soundcraft-cli from with cargo

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_STUDIO_DIR="$(cd "$HERE/../.." && pwd)"
SOUNDCRAFT_REPO="${SOUNDCRAFT_REPO:-$(cd "$AGENT_STUDIO_DIR/.." && pwd)}"

FORGE_ROOT="${FORGE_ROOT:-/opt/forge}"
PYTHON="${PYTHON:-python3}"
VENVS="$FORGE_ROOT/venvs"
SRC="$FORGE_ROOT/src"
BIN="$FORGE_ROOT/bin"
MODELS="$FORGE_ROOT/models"

log() { printf '[setup-venvs] %s\n' "$*"; }

if [ -z "${SYSTEM_SITE:-}" ]; then
  if "$PYTHON" -c 'import torch' >/dev/null 2>&1; then SYSTEM_SITE=1; else SYSTEM_SITE=0; fi
fi
log "root=$FORGE_ROOT python=$PYTHON system_site=$SYSTEM_SITE"

mkdir -p "$VENVS" "$SRC" "$BIN" "$MODELS"

# Pins the base image's torch stack so that no dependency resolution can
# replace it with a wheel that lacks aarch64 / CUDA 13 support.
CONSTRAINTS="$FORGE_ROOT/constraints.txt"
if [ "$SYSTEM_SITE" = "1" ]; then
  "$PYTHON" - >"$CONSTRAINTS" <<'PY'
from importlib import metadata

for name in ("torch", "torchaudio", "torchvision", "triton", "numpy"):
    try:
        print(f"{name}=={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        pass
PY
  export PIP_CONSTRAINT="$CONSTRAINTS"
  # setuptools >= 81 no longer ships pkg_resources; packages that still
  # import it in setup.py (e.g. some pandas sdists) fail to build. Force
  # binary wheels for those so pip never runs their setup.py.
  export PIP_ONLY_BINARY="${PIP_ONLY_BINARY:-pandas}"
fi

make_venv() { # name
  local dir="$VENVS/$1"
  if [ ! -x "$dir/bin/python" ]; then
    log "creating venv $dir"
    if [ "$SYSTEM_SITE" = "1" ]; then
      "$PYTHON" -m venv --system-site-packages "$dir"
    else
      "$PYTHON" -m venv "$dir"
    fi
  fi
  "$dir/bin/python" -m pip install --quiet --upgrade pip
}

fetch_src() { # name url ref
  local dir="$SRC/$1"
  if [ ! -d "$dir/.git" ]; then
    log "cloning $2"
    git clone --recurse-submodules "$2" "$dir"
  else
    git -C "$dir" fetch --quiet --tags origin
  fi
  if [ -n "${3:-}" ]; then
    git -C "$dir" checkout --quiet "$3"
  else
    git -C "$dir" merge --ff-only --quiet '@{u}' 2>/dev/null || true
  fi
  git -C "$dir" submodule update --init --recursive --quiet
}

install_agent_studio() { # venv-name extras
  # uv-synced venvs have no pip; prefer the venv's uv when it has one.
  if [ -x "$VENVS/$1/bin/uv" ]; then
    "$VENVS/$1/bin/uv" pip install --python "$VENVS/$1/bin/python" "${AGENT_STUDIO_DIR}${2:-}"
  else
    "$VENVS/$1/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true
    "$VENVS/$1/bin/python" -m pip install --quiet "${AGENT_STUDIO_DIR}${2:-}"
  fi
}

do_gateway() {
  make_venv gateway
  install_agent_studio gateway "[forge]"
}

do_foundation() {
  make_venv foundation
  fetch_src rc-stable-audio-tools https://github.com/RoyalCities/RC-stable-audio-tools "${FOUNDATION_REF:-}"
  "$VENVS/foundation/bin/python" -m pip install --quiet "$SRC/rc-stable-audio-tools"
  install_agent_studio foundation
}

uv_sync() { # name url ref extra-args-string
  make_venv "$1"
  fetch_src "$1-upstream" "$2" "$3"
  # uv-managed interpreters must live somewhere the runtime user can read:
  # the default lands in /root/.local/share/uv (mode 700) when the image is
  # built as root, and the gateway runs as an unprivileged user.
  export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$FORGE_ROOT/uv-python}"
  mkdir -p "$UV_PYTHON_INSTALL_DIR"
  # --ignore-installed: the NGC base ships uv in system site-packages, so a
  # plain install into a --system-site-packages venv is a silent no-op.
  "$VENVS/$1/bin/python" -m pip install --quiet --ignore-installed uv
  local skip=""
  if [ "$SYSTEM_SITE" = "1" ]; then
    skip="--no-install-package torch --no-install-package torchaudio --no-install-package torchvision --no-install-package triton"
  fi
  # shellcheck disable=SC2086
  (cd "$SRC/$1-upstream" && UV_PROJECT_ENVIRONMENT="$VENVS/$1" "$VENVS/$1/bin/uv" sync --inexact $skip $4)
  install_agent_studio "$1"
}

do_sa3() { uv_sync sa3 https://github.com/Stability-AI/stable-audio-3 "${SA3_REF:-}" "${SA3_UV_ARGS:-}"; }
do_acestep() { uv_sync acestep https://github.com/ace-step/ACE-Step-1.5 "${ACESTEP_REF:-}" "${ACESTEP_UV_ARGS:-}"; }

do_basicpitch() {
  # CPU only; never shares the GPU stack, so it gets an isolated venv.
  local saved="$SYSTEM_SITE"
  SYSTEM_SITE=0
  make_venv basicpitch
  SYSTEM_SITE="$saved"
  "$VENVS/basicpitch/bin/python" -m pip install --quiet basic-pitch
  install_agent_studio basicpitch
}

do_stems() {
  fetch_src stems.cpp https://github.com/betweentwomidnights/stems.cpp "${STEMS_REF:-}"
  local backend="${STEMS_BACKEND:-}"
  if [ -z "$backend" ]; then
    if command -v nvcc >/dev/null 2>&1; then backend=cuda; else backend=cpu; fi
  fi
  # build.sh builds every cmake target; upstream's test binary (stems-libtest)
  # fails to link without -lcuda. Forge only runs stems-split, so tolerate a
  # non-zero make exit as long as the binary we need was produced (checked below).
  (cd "$SRC/stems.cpp" && ./build.sh "$backend") || log "stems.cpp build had failures; checking for stems-split anyway"
  local found
  found="$(find "$SRC/stems.cpp" -type f -name stems-split -perm -u+x | head -n 1)"
  if [ -z "$found" ]; then
    # make may have aborted on the broken test target before linking stems-split;
    # the tree is already configured, so build just that target.
    bdir="$(find "$SRC/stems.cpp" -maxdepth 3 -name CMakeCache.txt -exec dirname {} \; 2>/dev/null | head -n 1)"
    [ -n "$bdir" ] && { log "rebuilding just stems-split in $bdir"; cmake --build "$bdir" --target stems-split -j"$(nproc)" || true; }
    found="$(find "$SRC/stems.cpp" -type f -name stems-split -perm -u+x | head -n 1)"
  fi
  if [ -z "$found" ]; then
    log "stems-split not found after build; check the build output above"
    return 1
  fi
  install -m 0755 "$found" "$BIN/stems-split"
  mkdir -p "$MODELS/stems"
  (cd "$SRC/stems.cpp" && ./models.sh ${STEMS_MODELS:-htdemucs})
  # models.sh writes into <checkout>/models; link it where forge.yaml points.
  for f in "$SRC"/stems.cpp/models/*.gguf; do
    [ -e "$f" ] && ln -sf "$f" "$MODELS/stems/$(basename "$f")"
  done
}

do_render() {
  if [ -n "${SOUNDCRAFT_CLI_BINARY:-}" ]; then
    install -m 0755 "$SOUNDCRAFT_CLI_BINARY" "$BIN/soundcraft-cli"
  elif command -v cargo >/dev/null 2>&1 && [ -f "$SOUNDCRAFT_REPO/Cargo.toml" ]; then
    (cd "$SOUNDCRAFT_REPO" && cargo build --release -p soundcraft-cli)
    install -m 0755 "$SOUNDCRAFT_REPO/target/release/soundcraft-cli" "$BIN/soundcraft-cli"
  else
    log "render: no SOUNDCRAFT_CLI_BINARY and no cargo; skipping. See deploy/README.md."
    return 0
  fi
  log "soundcraft-cli installed at $BIN/soundcraft-cli"
}

if [ "$#" -eq 0 ]; then
  set -- gateway foundation sa3 acestep basicpitch stems render
fi
for fam in "$@"; do
  case "$fam" in
    gateway|foundation|sa3|acestep|basicpitch|stems|render)
      log "== $fam"
      "do_$fam"
      ;;
    *)
      log "unknown family: $fam"
      exit 2
      ;;
  esac
done
log "done. Next: FORGE_TOKEN=... $VENVS/gateway/bin/forge check --config /etc/forge/forge.yaml"
