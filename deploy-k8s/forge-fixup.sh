#!/bin/bash
# Bakes the runtime fixes that aren't yet in the forge:cpu-gpu-v2 image into
# the container's writable layer at startup (before the gateway serves).
# The next image build carries all of these (setup-venvs.sh + protocol.py are
# fixed on main), at which point this script becomes a no-op.
set -u
log() { echo "[forge-fixup] $*" >&2; }

# 1. sa3 venv (uv-managed Python 3.10) shares nothing with the NGC base: it
#    needs its own torch, and the worker code needs typing_extensions.NotRequired.
SA3=/opt/forge/venvs/sa3
if [ -x "$SA3/bin/python" ]; then
  if ! "$SA3/bin/python" -c "import torch" >/dev/null 2>&1; then
    log "installing torch into sa3 venv"
    /usr/local/bin/uv pip install --python "$SA3/bin/python" torch torchaudio >/dev/null 2>&1 \
      || log "sa3 torch install FAILED"
  fi
  f="$SA3/lib/python3.10/site-packages/forge/workers/protocol.py"
  if [ -f "$f" ] && grep -q "from typing import Any, NotRequired, TypedDict" "$f"; then
    log "patching sa3 protocol.py for python 3.10"
    sed -i "s/^from typing import Any, NotRequired, TypedDict$/from typing import Any, TypedDict\n\nfrom typing_extensions import NotRequired/" "$f"
    /usr/local/bin/uv pip install --python "$SA3/bin/python" typing_extensions >/dev/null 2>&1 || true
  fi
  "$SA3/bin/python" -c "import forge.workers.sa3" 2>/dev/null || log "WARNING: sa3 worker still not importable"
fi

# 2. Gateway writes its SQLite jobdb/asset index into the PVC; files created
#    by an earlier root run are unreadable to the service user.
if [ -d /data ]; then
  chown -R 999:999 /data 2>/dev/null || true
  chmod -R u+rwX /data 2>/dev/null || true
fi

exec "$@"
