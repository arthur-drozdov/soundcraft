# deploy-k8s — Forge on the k0s-ts DGX Spark cluster

One-command restore of the whole serving stack:

```sh
kubectl apply -f deploy-k8s/forge-gpu.yaml        # config, PVCs, deployment, LB service
```

Image builds run in CI: the `forge-image` workflow (`.github/workflows/forge-image.yml`)
builds `agent-studio/deploy/docker/forge.Dockerfile` with buildx on a GitHub-hosted
arm64 runner and pushes `ghcr.io/arthur-drozdov/forge-ci:<tag>` (workflow_dispatch takes a
`tag` input; pushes touching `agent-studio/**` rebuild the default `cpu-gpu-v3` tag).
The old in-cluster kaniko pod (`forge-build-pod.yaml`) is retired — the self-hosted
spark runners carry no container daemon (no privileged DinD), so building on the cluster
is not available; hosted arm runners build this image for free. After a build lands,
`kubectl apply -f forge-gpu.yaml` restores serving; retag by editing `--destination`
in CI usage and the deployment's `image:` together.

## Files
| file | purpose |
|---|---|
| `forge-gpu.yaml` | serving stack: ConfigMap `forge-config`, PVCs `forge-data`/`forge-models`, Deployment `forge-gateway`, Tailscale LoadBalancer `forge` (port 7700) |
| `forge-build-pod.yaml` | retired kaniko build pod (superseded by the `forge-image` GitHub Actions workflow) |
| `forge-cpu.yaml` | historical CPU-only stage (superseded; kept for reference) |
| `forge-fixup.sh` | historical startup hotpatch (pre-v3 images only; the fixes are now in the image) |

## Requirements / cluster conventions discovered
- **`runtimeClassName: nvidia` is mandatory** for GPU on this cluster — without
  it the container gets NVIDIA env vars but no `/dev/nvidia*` nodes.
- The GPU device plugin caps **shared pods at 1 `nvidia.com/gpu`**.
- `require-guaranteed-qos` ValidatingAdmissionPolicy: every container needs
  `requests == limits` (cpu, memory).
- `local-path` PVCs are node-pinned: `forge-data`/`forge-models` live on
  spark-4a06 (the deployment's nodeSelector).
- spark-15d6 is cordoned + tainted `qwen.sonnia.ai/dedicated` — do not schedule
  Forge there.
- Secrets needed before first apply (namespace `forge`): `forge-token`
  (`FORGE_TOKEN`), `forge-hf` (`HF_TOKEN`), `ghcr-pull` (docker-registry,
  ghcr.io, PAT with `read:packages`; refresh when the PAT rotates).

## Exemptions to request from the cluster automation
This namespace has twice been swept by autonomous cluster controllers
(deployments/PVCs deleted, the GHCR `forge` package purged mid-session, 15d6
cordoned). For stable operation, the janitors/agents (openclaw
`arc-controller`/`decision-controller`, the qwen reservation system) should:

1. **not garbage-collect the `forge` namespace** (or exempt label
   `app=forge-gateway`), and
2. **not delete/purge the `ghcr.io/arthur-drozdov/forge` package** — the node
   image cache is ephemeral; a purge plus a pod eviction takes Forge down
   until an image rebuild (~40 min),
3. treat `deploy-k8s/forge-gpu.yaml` as the source of truth: re-applying it
   fully restores serving (models re-download to the `forge-data` PVC cache
   only if that PVC is also lost).

## Verified working (2026-10-27)
`forge:cpu-gpu-v3` serves 10 models (ace-step ×2, sa3 ×3, stems.cpp, yin,
testtone, analysis, soundcraft render) on the GB10; the full demo-pack run
(6 ACE-Step loops, 5 samples, 4 SFX, MIDI transcription, 3-way stem split,
6 analyses, session render via `soundcraft-cli`) completed green end-to-end
over `forge-forge.tail3eb9a6.ts.net:7700`.
