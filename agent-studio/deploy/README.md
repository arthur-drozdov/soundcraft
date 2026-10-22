# Deploying Forge on the DGX Spark

Forge is the generation service of Agent Studio (spec section 6, design section 3). It runs on the
Spark; the laptop talks to it over a private network through `forge-mcp`.

Two supported installs:

| | Container (`compose.yaml`) | Bare metal (`systemd/forge.service`) |
|---|---|---|
| Image / venvs | one `forge` image, venvs under `/opt/forge/venvs` | venvs under `/opt/forge/venvs` made by `scripts/setup-venvs.sh` |
| Torch | from the NGC PyTorch base image | from your system Python (see prerequisites) |
| Config | `forge.container.yaml` (baked in, overridable) | `/etc/forge/forge.yaml` from `forge.example.yaml` |

Why one image: the gateway starts each model family as a **subprocess** (DESIGN 3.2) from the
interpreter named in `families.<name>.python`. A container per family cannot be spawned from the
gateway, so the supported image contains the gateway and a venv per family. The venvs are created
with `python -m venv --system-site-packages`, so the base image's torch (built for aarch64 and
Blackwell) is shared and never replaced: `setup-venvs.sh` pins it with a pip constraints file.
The per-family files in `docker/` (`foundation`, `sa3`, `acestep`, `basicpitch`, `stems`,
`render`, `gateway`) are building blocks for operators who split the deployment (for example by
running a worker behind their own wrapper); they are not used by `compose.yaml`.

## Files

| Path | Purpose |
|---|---|
| `forge.example.yaml` | Annotated config for the bare-metal install, every key from DESIGN 3.1 |
| `forge.container.yaml` | Config baked into the container image |
| `compose.yaml`, `forge.env.example` | Container deployment with GPU reservation |
| `docker/forge.Dockerfile` | The single all-in-one image (build context: repository root) |
| `docker/<family>.Dockerfile` | Optional per-family images (context: `agent-studio/`, `render` and `forge`: repo root) |
| `scripts/setup-venvs.sh` | Idempotent venv + upstream package installer |
| `systemd/forge.service` | Unit for the bare-metal install |

## Prerequisites on the Spark

- DGX OS with NVIDIA driver 580.95.05 or newer (CUDA 13) per NVIDIA's DGX Spark docs.
- Docker with the NVIDIA container runtime (preinstalled on DGX OS) for the container install.
- An NGC account: `docker login nvcr.io` with username `$oauthtoken` and an NGC API key as password.
- Tailscale (or WireGuard) up on the Spark and on the laptop.
- For bare metal: Python 3.11 or 3.12 with a working `import torch` on the GPU. The Spark has no
  generic aarch64 + CUDA 13 pip wheel set yet, so take torch from the NGC container or NVIDIA's
  Spark instructions rather than from PyPI. Also `git`, `cmake`, a C++ toolchain, `ffmpeg`,
  `libsndfile`. Python 3.11/3.12 is what ACE-Step 1.5 requires.
- The base image tag is pinned to `nvcr.io/nvidia/pytorch:25.11-py3`, the tag the Arm learning
  path for the Spark uses. This was taken from secondary sources; check the NGC catalog for a newer
  tag and set `BASE_IMAGE` in `compose.yaml` if you want one.

## Network binding and token

Forge must never be reachable from the open network.

1. Get the Tailscale address: `tailscale ip -4`.
2. Bare metal: set `listen.host` to it in `forge.yaml`. Container: put it in `forge.env` as
   `FORGE_BIND_IP`; Docker publishes port 7700 only on that address (inside the container Forge
   listens on `0.0.0.0`, which is why that config file must not be used outside the container).
3. Token: `python3 -c 'import secrets; print(secrets.token_urlsafe(32))'`. Store it as
   `FORGE_TOKEN` in `forge.env` (compose) or `/etc/forge/forge.env` (systemd), mode 0600. Forge
   refuses to start with an empty token. `GET /v1/health` is the only unauthenticated route.
4. On the laptop set `FORGE_URL=http://<tailscale-ip>:7700` and the same `FORGE_TOKEN` for
   `forge-mcp`.

## First run

Container:

```sh
cd agent-studio/deploy
cp forge.env.example forge.env        # fill FORGE_TOKEN and FORGE_BIND_IP, chmod 600
mkdir -p models                       # checkpoints, see below (or set FORGE_MODELS_DIR)
docker compose --env-file forge.env up -d --build
docker compose exec forge forge check
```

The first build is long: it compiles stems.cpp and soundcraft-cli and installs three model
stacks. Use `WITH_RENDER: "0"` / `WITH_STEMS: "0"` build args to skip those, then provide the
binaries (see "Render worker" and "stems" below).

Bare metal:

```sh
sudo useradd --system --create-home --home-dir /var/lib/forge forge
sudo install -d -o forge -g forge /opt/forge /var/lib/forge /etc/forge
sudo -u forge FORGE_ROOT=/opt/forge agent-studio/deploy/scripts/setup-venvs.sh   # all families, or name some
sudo cp agent-studio/deploy/forge.example.yaml /etc/forge/forge.yaml             # edit listen.host
# create /etc/forge/forge.env with FORGE_TOKEN=..., mode 0600, owner forge
sudo cp agent-studio/deploy/systemd/forge.service /etc/systemd/system/
sudo systemctl enable --now forge
journalctl -u forge -f
```

### `forge check`

`forge check` spawns every enabled family, prints each model with its availability (and the reason
when unavailable), and runs one tiny job per available model. Run it after every install or
upgrade:

```sh
FORGE_TOKEN=... /opt/forge/venvs/gateway/bin/forge check --config /etc/forge/forge.yaml
```

A family whose packages or checkpoints are missing reports its models as unavailable; it never
stops the gateway. `testtone` and `yin` always work and are the smoke test for the plumbing. Then
from the laptop: `curl http://<tailscale-ip>:7700/v1/health`.

## Models, downloads and licences

Weights are never baked into images. Hugging Face downloads go to `HF_HOME`
(`/var/lib/forge/hf-cache`, or `/data/hf-cache` in the container). For gated repositories accept
the terms on the model page and set `HF_TOKEN`.

| Family | Model ids | Source | Where it goes | Licence |
|---|---|---|---|---|
| foundation | `foundation-1`, `foundation-1.2-samples`, `foundation-1.2-keybeds` | RoyalCities, installed via RC Stable Audio Tools | `/opt/forge/models/<model>/` (container: `/models/<model>/`), model config JSON next to the checkpoint | Stability AI Community Licence (spec 6.2). The RC Stable Audio Tools code licence is unverified: check its LICENSE before redistributing |
| sa3 | `sa3-small-music`, `sa3-small-sfx`, `sa3-medium` | `stabilityai/stable-audio-3-small-music`, `-small-sfx`, `-medium` | Hugging Face cache, fetched on first load | Stability AI Community Licence for weights (spec 6.2); the code repository's LICENSE file is MIT. Read the model cards for exact weight terms |
| acestep | `ace-step-1.5-base`, `ace-step-1.5-xl-base` | `ACE-Step/acestep-v15-base`, `ACE-Step/acestep-v15-xl-base` | Hugging Face cache (upstream downloads on first run) | MIT |
| basicpitch | `basic-pitch` | `pip install basic-pitch` (bundled model) | the venv | Apache-2.0 |
| stems | `stems-cpp` | stems.cpp built from source; models from its `models.sh` | `/opt/forge/models/stems/*.gguf` | stems.cpp runtime: MIT per spec 6.2 (the repository licence was not independently confirmed). Models: `htdemucs*` are MIT; the anvuew dereverb model is GPL-3.0; viperx BS-RoFormer and UVR models have no upstream licence. Only use licensed models |
| render | `soundcraft` | this repository | `/opt/forge/bin/soundcraft-cli` | MIT OR Apache-2.0 |
| testtone, yin | `testtone`, `yin` | built in | - | project licence |

Each adapter carries its licence in a table and every asset sidecar records it. Non-commercial
weights (MusicGen, MelodyFlow, TangoFlux) are excluded. The Stability Community Licence has
conditions for commercial use (registration, an annual-revenue threshold); read the exact terms
before any release.

### Foundation checkpoints

The upstream research could not read the Hugging Face model cards, so the exact checkpoint file
names are not known to this repository. Download the three RoyalCities models from their Hugging
Face pages (accepting their terms), place each in its own directory with the config JSON beside
the checkpoint, and set the real file names in `families.foundation.options.checkpoints`
(`config` and `ckpt` per model id). Until you do, `forge check` shows the three models as
unavailable with the reason; nothing else is affected. Likewise, if you find the model cards name
other files, only that YAML block changes.

### Render worker

`render` needs a headless `soundcraft-cli` for aarch64. Either:

- compile it: the `forge` image does so by default (`WITH_RENDER=1`, Rust 1.90 as in the
  workspace `rust-version`); on bare metal `setup-venvs.sh render` runs
  `cargo build --release -p soundcraft-cli` when `cargo` is present; or
- use a released `linux-aarch64` build: unpack the tarball, then
  `SOUNDCRAFT_CLI_BINARY=/path/soundcraft-cli setup-venvs.sh render`, or mount the binary at
  `/opt/forge/bin/soundcraft-cli` in the container (build with `WITH_RENDER=0`).

If the Rust build lacks system libraries on your base, install them in the Dockerfile's build
stage (it installs `pkg-config`, `libasound2-dev`, `libssl-dev`, `cmake`); the workspace's own
packaging notes are authoritative.

## Memory budget

The Spark has 128 GB of unified memory shared by the OS, Forge's models and any SoundCraft render
process. `memory_budget_gb` caps the sum of loaded models' `vram_gb`; least recently used models
are evicted first, and `POST /v1/models/{id}/load` returns 409 if a model cannot fit.

- Default 96 GB leaves about 30 GB for the OS, page cache, the CPU workers (stems, basic-pitch,
  render) and CUDA overhead. Lower it to 64-80 GB if you also run a desktop session or other
  services on the Spark.
- Heavy models: SA3 medium (1.4B), ACE-Step XL-base (4B DiT plus a language model). Foundation
  needs about 7 GB per the author's figures. The two heavy families plus the LM do not need to be
  warm together; the budget forces eviction instead of out-of-memory kills.
- Only one GPU job runs at a time (`gpu` lane) and one CPU job at a time (`cpu` lane).
- Watch `GET /v1/health` (`memory.used_gb`, `gpu.memory_used_gb`) while testing a new model and
  adjust the model's `vram_gb` in its adapter if the real figure differs.

## Porting notes: ARM64 and Blackwell

- The Spark is aarch64 with an SM121 (Blackwell) GPU and CUDA 13. Prefer the NGC PyTorch image
  over bare pip wheels; the PyPI aarch64 + CUDA 13 set is still filling in.
- Never let pip or uv replace the base torch. `setup-venvs.sh` writes `/opt/forge/constraints.txt`
  from the installed torch, torchaudio, torchvision, triton and numpy and exports `PIP_CONSTRAINT`;
  for the two `uv sync` upstreams (SA3, ACE-Step) it also passes `--no-install-package` for the torch
  packages. If an upstream needs a different torch, run its sync with `SA3_UV_ARGS` /
  `ACESTEP_UV_ARGS` and verify on the device.
- Stable Audio 3's README installs with `uv sync` (extras `ui`, `lora`) and defaults to CUDA 12.6
  torch; its medium model needs flash-attn 2, which may need a source build on aarch64/SM121
  (`uv sync --inexact` keeps a hand-built flash-attn installed). The small models run on CPU as a
  fallback.
- torchaudio and attention kernels (flash-attn, xformers) may need source builds. Validate each
  model alone with `forge check` before enabling the next.
- The RC Stable Audio Tools README was tested on Python 3.10 and mentions optional INT4 via
  TorchAO; neither is configured here.
- Basic Pitch uses TFLite on Linux; if no aarch64 runtime wheel exists for your Python version,
  the model reports unavailable and `yin` remains the transcription fallback.
- The research could not verify: the Foundation checkpoint file names, the SA3 Python import path
  (the adapter imports it lazily and reports the error), the stems.cpp licence file, and the
  newest NGC tag. The deploy files leave each of these as an operator-supplied value as described
  above rather than guessing.

## Upgrading

1. `git pull` the repository.
2. Container: `docker compose --env-file forge.env up -d --build`. Data in the `forge-data`
   volume (assets, SQLite index) is kept. Pin upstream versions with build args
   (`FOUNDATION_REF`, `SA3_REF`, `ACESTEP_REF`, `STEMS_REF` are read by `setup-venvs.sh`).
3. Bare metal: re-run `setup-venvs.sh` (idempotent; fetches upstream checkouts and reinstalls
   `agent-studio`), then `sudo systemctl restart forge`.
4. Run `forge check`. Queued and running jobs are marked failed with "forge restarted" on restart;
   finish or resubmit them.
5. `forge purge` removes unrated, unused assets older than `retention_days` (also runs daily).

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| Gateway exits at start with a token error | `FORGE_TOKEN` is empty or not in the unit's/ container's environment (`token_env`) |
| Laptop cannot connect | Port is published only on `FORGE_BIND_IP`; check `tailscale status`, the address, and that the Spark's Tailscale IP did not change |
| `docker compose` complains about `FORGE_BIND_IP` | It is required; set it in `forge.env` and pass `--env-file forge.env` |
| `could not select device driver "nvidia"` | NVIDIA container runtime missing: `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker` |
| Pull from `nvcr.io` is denied | `docker login nvcr.io` with `$oauthtoken` and an NGC API key |
| Model shows `available: false` | Read `reason` in `/v1/models` or the `forge check` output: usually a missing package (venv not installed), missing checkpoint path in `options`, or a gated Hugging Face repo without `HF_TOKEN` |
| `torch` re-downloaded or `CUDA error: no kernel image` | A pip/uv step replaced the base torch. Delete the venv, rerun `setup-venvs.sh` with `SYSTEM_SITE=1` so the constraints file applies |
| Worker keeps restarting | The manager restarts with backoff; run the worker by hand: `/opt/forge/venvs/<family>/bin/python -m forge.workers.<family>` and type a `{"id":1,"op":"hello","options":{}}` line |
| Out of memory | Lower `memory_budget_gb`, unload models via `POST /v1/models/{id}/unload`, or disable a heavy family |
| `stems-split` not found | Build failed or `setup-venvs.sh stems` was skipped; check `families.stems.options.binary` |
| `render` unavailable | No `soundcraft-cli` at `families.render.options.binary`; see "Render worker" |
| Build fails: `README.md` not found | `pyproject.toml` references `README.md`; make sure `agent-studio/README.md` exists in the build context |
