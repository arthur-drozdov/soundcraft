"""The ``sa3`` family: Stable Audio 3 (small-music, small-sfx, medium) through the ``stable_audio_3`` package.

Run as ``python -m forge.workers.sa3`` in the environment made by ``uv sync`` in a clone of
https://github.com/Stability-AI/stable-audio-3 (the package is imported as ``stable_audio_3``).

``hello`` options (``deploy/forge.example.yaml``)::

    device: cuda
    checkpoints:
      sa3-small-music: small-music      # a model name known to StableAudioModel.from_pretrained ...
      sa3-medium: /opt/models/sa3-m     # ... or a directory holding model_config.json + model.safetensors

Optional extras: ``steps`` (8, the post-trained default), ``cfg_scale`` (1.0).

Network: the first ``load`` of a named model downloads ``stabilityai/stable-audio-3-<name>`` from
Hugging Face into ``HF_HOME`` (set ``HF_TOKEN`` if the repository is gated; ``HF_HUB_OFFLINE=1`` forbids
it). A local directory needs no network.

Upstream calls (source: stable-audio-3 @ 77e0398, ``stable_audio_3/model.py``; none run here)::

    stable_audio_3.StableAudioModel.from_pretrained(name, device=, model_half=) -> model
    stable_audio_3.loading_utils.load_diffusion_cond(config, ckpt, device=, model_half=)   # local directory
    model.generate(prompt=, negative_prompt=None, duration=, steps=, cfg_scale=, batch_size=1, sample_size=,
                   seed=, init_audio=(sr, tensor), init_noise_level=, inpaint_audio=(sr, tensor),
                   inpaint_mask_start_seconds=, inpaint_mask_end_seconds=, callback=) -> tensor [1, ch, n]

Job mapping:

* ``generate.sfx`` / ``generate.sample`` / ``generate.loop``  text-to-audio; loops get ``<N> BPM`` and the key in the
  prompt and are cut to ``bars * 4 * 60 / bpm`` seconds.
* ``transform.audio``   ``init_audio`` with ``init_noise_level = strength`` (SA3: 1.0 replaces the input
  completely, 0.1 is a close variation).
* ``inpaint.audio``     ``inpaint_audio`` with the mask ``[start_s, end_s]``.
* ``extend.audio``      continuation: ``inpaint_audio`` with the mask from the end of the input to
  ``input + seconds``, ``duration = input + seconds``; the result contains the original followed by the
  extension.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import _prompts as prompts
from . import _upstream as up
from .protocol import ModelInfo, Output, RunJob
from .runtime import CancelCheck, Family, Progress, check_cancel, main

log = logging.getLogger("forge.worker.sa3")

LICENCE = (
    "Stability AI Community Licence (weights; commercial use has registration and revenue conditions, "
    "see spec 6.7). Code repository: MIT"
)
ALL_JOBS = ["generate.sfx", "generate.loop", "generate.sample", "transform.audio", "extend.audio", "inpaint.audio"]
MODELS: dict[str, dict[str, Any]] = {
    "sa3-small-music": {
        "name": "small-music", "vram_gb": 3.0, "max_seconds": 120.0, "sfx": False, "cuda_only": False,
        "jobs": ["generate.loop", "generate.sample", "transform.audio", "extend.audio", "inpaint.audio"],
    },
    "sa3-small-sfx": {
        "name": "small-sfx", "vram_gb": 3.0, "max_seconds": 120.0, "sfx": True, "cuda_only": False,
        "jobs": ["generate.sfx", "generate.sample", "transform.audio", "extend.audio", "inpaint.audio"],
    },
    "sa3-medium": {
        "name": "medium", "vram_gb": 8.0, "max_seconds": 380.0, "sfx": False, "cuda_only": True, "jobs": ALL_JOBS,
    },
}
KNOWN_NAMES = ("small-music", "small-sfx", "medium", "small-music-base", "small-sfx-base", "medium-base")
SAMPLE_RATE = 44100
MIN_SAMPLE_SECONDS = 1.0
FADE_MS = 8.0


@dataclass
class _Loaded:
    model: Any
    device: str
    sample_rate: int
    sample_size: int
    max_seconds: float


class SA3Family(Family):
    name = "sa3"

    def __init__(self) -> None:
        self._device = "cuda"
        self._steps = 8
        self._cfg = 1.0
        self._checkpoints: dict[str, str] = {}
        self._config_error: str | None = None
        self._loaded: dict[str, _Loaded] = {}

    # ---- configuration and availability ----

    def configure(self, options: dict[str, Any]) -> None:
        self._config_error = None
        try:
            self._device = up.check_device(options.get("device", "cuda"))
            self._steps = up.integer(options, "steps", 8, 1, 200)
            self._cfg = up.number(options, "cfg_scale", 1.0, 0.0, 25.0)
            raw = options.get("checkpoints") or {}
            if not isinstance(raw, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in raw.items()):
                raise ValueError("options.checkpoints must map model ids to a model name or directory")
            self._checkpoints = dict(raw)
        except ValueError as exc:
            self._config_error = f"invalid options: {exc}"

    def _checkpoint(self, model: str) -> str:
        return self._checkpoints.get(model) or str(MODELS[model]["name"])

    def _local_dir(self, checkpoint: str) -> Path | None:
        """The directory a checkpoint setting points at, or ``None`` for a named (Hugging Face) model."""
        if checkpoint in KNOWN_NAMES:
            return None
        if "\x00" in checkpoint:
            raise up.MissingDependency("checkpoint path contains a NUL byte")
        path = Path(checkpoint).expanduser()
        if not path.is_dir():
            raise up.MissingDependency(
                f"checkpoint {checkpoint!r} is neither a model name ({', '.join(KNOWN_NAMES)}) nor a directory"
            )
        for fname in ("model_config.json", "model.safetensors"):
            if not (path / fname).is_file():
                raise up.MissingDependency(f"{path / fname} does not exist")
        return path

    def _availability(self, model: str) -> tuple[str | None, str, str | None]:
        checkpoint = self._checkpoint(model)
        if self._config_error:
            return self._config_error, checkpoint, None
        missing = up.first_missing("torch", "stable_audio_3")
        if missing:
            return f"{missing}; install https://github.com/Stability-AI/stable-audio-3 (uv sync)", checkpoint, None
        try:
            local = self._local_dir(checkpoint)
        except up.MissingDependency as exc:
            return str(exc), checkpoint, None
        return None, checkpoint, up.fingerprint(local / "model.safetensors") if local else None

    def models(self) -> list[ModelInfo]:
        out: list[ModelInfo] = []
        for model_id, spec in MODELS.items():
            reason, checkpoint, digest = self._availability(model_id)
            info: ModelInfo = {
                "id": model_id, "family": self.name, "job_types": list(spec["jobs"]), "available": reason is None,
                "licence": LICENCE, "checkpoint": checkpoint, "checkpoint_hash": digest,
                "vram_gb": float(spec["vram_gb"]), "max_seconds": float(spec["max_seconds"]),
            }
            if reason:
                info["reason"] = reason
            out.append(info)
        return out

    # ---- memory ----

    def load(self, model: str) -> None:
        if model not in MODELS:
            raise ValueError(f"unknown model {model!r}")
        if model in self._loaded:
            return
        reason, checkpoint, _ = self._availability(model)
        if reason:
            raise up.MissingDependency(reason)
        spec = MODELS[model]
        torch = up.import_module("torch")
        package = up.import_module("stable_audio_3")
        try:
            cls = package.StableAudioModel
        except AttributeError as exc:
            raise up.MissingDependency(f"stable_audio_3 has no StableAudioModel: {exc}") from exc
        device = up.resolve_device(torch, self._device)
        if spec["cuda_only"] and not device.startswith("cuda"):
            raise up.MissingDependency(f"{model} needs a CUDA GPU (it uses flash-attention); device is {device}")
        half = device.startswith("cuda")
        local = self._local_dir(checkpoint)
        log.info("loading %s (%s) on %s", model, checkpoint, device)
        if local is None:
            net = cls.from_pretrained(checkpoint, device=device, model_half=half)
        else:
            loading = up.import_module("stable_audio_3.loading_utils")
            config = up.read_json_file(local / "model_config.json")
            core = loading.load_diffusion_cond(config, str(local / "model.safetensors"), device=device, model_half=half)
            net = cls(core, config, device, half)
        rate = int(getattr(getattr(net, "model", None), "sample_rate", SAMPLE_RATE) or SAMPLE_RATE)
        max_seconds = float(spec["max_seconds"])
        self._loaded[model] = _Loaded(net, device, rate, max(5292032, int(max_seconds * rate)), max_seconds)
        up.free_accelerator_memory()

    def unload(self, model: str) -> None:
        if self._loaded.pop(model, None) is not None:
            up.free_accelerator_memory()

    # ---- generation ----

    def _generate(
        self, loaded: _Loaded, *, prompt: str, duration: float, seed: int, progress: up.Span, cancelled: CancelCheck,
        init: tuple[int, np.ndarray, float] | None = None, inpaint: tuple[int, np.ndarray, float, float] | None = None,
    ) -> np.ndarray:
        """One ``model.generate`` call; returns ``float32 (channels, frames)`` at the model rate."""
        check_cancel(cancelled)
        if duration > loaded.max_seconds + 1e-6:
            raise ValueError(f"{duration:.1f} s exceeds this model's maximum of {loaded.max_seconds:g} s")
        torch = up.import_module("torch")
        steps = self._steps

        def on_step(info: dict[str, Any]) -> None:
            check_cancel(cancelled)
            try:
                step = int(info.get("i", 0))
            except (TypeError, ValueError):
                step = 0
            progress((step + 1) / max(1, steps), f"step {step + 1}/{steps}")

        kwargs: dict[str, Any] = {
            "prompt": prompt, "duration": float(duration), "steps": steps, "cfg_scale": self._cfg, "batch_size": 1,
            "sample_size": loaded.sample_size, "seed": int(seed), "callback": on_step,
        }
        if init is not None:
            sr, arr, level = init
            kwargs.update(init_audio=(int(sr), torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))), init_noise_level=float(level))
        if inpaint is not None:
            sr, arr, start, end = inpaint
            kwargs.update(
                inpaint_audio=(int(sr), torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))),
                inpaint_mask_start_seconds=float(start), inpaint_mask_end_seconds=float(end),
            )
        log.info("generating %.2f s seed=%d prompt=%r", duration, seed, prompt)
        with torch.no_grad():
            out = loaded.model.generate(**kwargs)
        audio = up.tensor_to_audio(out)
        check_cancel(cancelled)
        up.free_accelerator_memory()
        return audio[:, : max(1, int(round(duration * loaded.sample_rate)))]

    def run(self, job: RunJob, progress: Progress, cancelled: CancelCheck) -> list[Output]:
        model_id = str(job.get("model"))
        if model_id not in MODELS:
            raise ValueError(f"unknown model {model_id!r}")
        loaded = self._loaded.get(model_id)
        if loaded is None:
            raise RuntimeError(f"model {model_id!r} is not loaded")
        params, out_dir = up.check_job(job, MODELS[model_id]["jobs"])
        seed, count = up.job_seed(job), up.candidate_count(params)
        tags: dict[str, Any] = params["tags"] if isinstance(params.get("tags"), dict) else {}
        prompt = str(params.get("prompt") or "")
        span = up.Span(progress)
        span(0.0, "started")
        jtype = job["type"]
        input_audio: tuple[np.ndarray, int] | None = None
        if jtype in ("transform.audio", "extend.audio", "inpaint.audio"):
            input_audio = up.read_audio_input(job, "audio", max_seconds=loaded.max_seconds)
        outputs: list[Output] = []
        for candidate in range(count):
            sub = span.part(candidate, count)
            audio, extra = self._one(loaded, model_id, job, params, tags, prompt, input_audio, up.candidate_seed(seed, candidate), sub, cancelled)
            path = up.write_audio(out_dir, f"candidate_{candidate}.wav", audio, loaded.sample_rate)
            outputs.append({"path": str(path), "kind": "audio", "candidate": candidate, "extra": extra})
        span(1.0, "done")
        return outputs

    def _one(
        self, loaded: _Loaded, model_id: str, job: RunJob, params: Any, tags: dict[str, Any], prompt: str,
        input_audio: tuple[np.ndarray, int] | None, seed: int, span: up.Span, cancelled: CancelCheck,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        jtype = job["type"]

        def gen(
            text: str, duration: float, init: tuple[int, np.ndarray, float] | None = None,
            inpaint: tuple[int, np.ndarray, float, float] | None = None,
        ) -> np.ndarray:
            return self._generate(
                loaded, prompt=text, duration=duration, seed=seed, progress=span, cancelled=cancelled,
                init=init, inpaint=inpaint,
            )

        if jtype == "generate.sfx":
            text = prompts.sa3_sfx_prompt(prompt, tags)
            return gen(text, up.number(params, "duration_s", 4.0, 0.1, 47.0)), {"conditioning_prompt": text}
        if jtype == "generate.loop":
            if "bpm" not in params:
                raise ValueError("generate.loop needs bpm")
            bars, bpm = up.integer(params, "bars", 4, 1, 16), up.number(params, "bpm", 120.0, 40.0, 240.0)
            text = prompts.sa3_loop_prompt(prompt, tags, bpm=bpm, key=params.get("key"), bars=bars)
            seconds = 60.0 / bpm * 4 * bars
            return gen(text, seconds), {"conditioning_prompt": text}
        if jtype == "generate.sample":
            note, length = up.integer(params, "note", 60, 0, 127), up.number(params, "length_s", 2.0, 0.05, 20.0)
            text = prompts.sa3_sample_prompt(prompt, tags, midi_note=note, sfx_model=bool(MODELS[model_id]["sfx"]))
            audio = gen(text, max(length, MIN_SAMPLE_SECONDS))
            audio = up.fade_out(audio[:, : max(1, int(round(length * loaded.sample_rate)))], loaded.sample_rate, FADE_MS)
            return audio, {"conditioning_prompt": text, "note": note}
        assert input_audio is not None
        audio_in, in_sr = input_audio
        length_in = audio_in.shape[1] / in_sr
        text = prompts.sa3_edit_prompt(prompt, tags)
        if jtype == "transform.audio":
            strength = up.number(params, "strength", 0.5, 0.0, 1.0)
            return gen(text, length_in, init=(in_sr, audio_in, strength)), {
                "conditioning_prompt": text, "init_noise_level": strength,
            }
        if jtype == "inpaint.audio":
            start, end = up.number(params, "start_s", 0.0, 0.0, 3600.0), up.number(params, "end_s", 0.0, 0.0, 3600.0)
            if start >= length_in or end <= start:
                raise ValueError(f"the region {start:g}..{end:g} s lies outside the {length_in:.2f} s input")
            end = min(end, length_in)
            return gen(text, length_in, inpaint=(in_sr, audio_in, start, end)), {
                "conditioning_prompt": text, "region_s": [start, end],
            }
        if jtype == "extend.audio":
            extra_s = up.number(params, "seconds", 8.0, 1.0, 60.0)
            total = length_in + extra_s
            if total > loaded.max_seconds:
                raise ValueError(f"{length_in:.1f} s of input plus {extra_s:g} s exceeds the model maximum of {loaded.max_seconds:g} s")
            return gen(text, total, inpaint=(in_sr, audio_in, length_in, total)), {
                "conditioning_prompt": text, "extended_from_s": round(length_in, 4),
            }
        raise ValueError(f"sa3 does not run {jtype!r}")  # pragma: no cover - check_job already filtered


if __name__ == "__main__":
    main(SA3Family())
