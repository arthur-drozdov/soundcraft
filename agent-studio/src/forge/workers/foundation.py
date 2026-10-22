"""The ``foundation`` family: RoyalCities Foundation-1, 1.2 Samples and 1.2 Keybeds through RC Stable Audio Tools.

Run as ``python -m forge.workers.foundation`` in the environment where ``RC-stable-audio-tools`` is
installed (``pip install .`` from a clone of https://github.com/RoyalCities/RC-stable-audio-tools).

``hello`` options (``deploy/forge.example.yaml``)::

    device: cuda                       # cpu | mps | cuda | cuda:N (falls back to cpu with a warning)
    checkpoints:
      foundation-1:
        config: /opt/forge/models/foundation-1/model_config.json
        ckpt:   /opt/forge/models/foundation-1/model.ckpt        # .ckpt or .safetensors

Optional extras: ``steps`` (75), ``cfg_scale`` (7.0), ``sampler_type`` (``dpmpp-3m-sde``), ``sigma_min``
(0.03), ``sigma_max`` (500), ``cfg_rescale`` (0.0), ``keybed_steps`` (80), ``keybed_cfg_scale`` (6.0),
``dtype`` (``auto``, ``float32``, ``float16`` or ``bfloat16``). The defaults are the ones RC's own
Gradio UI uses.

Network: none. The checkpoints are local files; nothing is downloaded.

Upstream calls this adapter makes (all read from RC-stable-audio-tools @ 43dcbb4, none run here)::

    stable_audio_tools.models.factory.create_model_from_config(config) -> model
    stable_audio_tools.models.utils.load_ckpt_state_dict(path) -> state_dict
    stable_audio_tools.inference.generation.generate_diffusion_cond(model, steps=, cfg_scale=,
        conditioning=[{"prompt", "seconds_start", "seconds_total"}], batch_size=1, sample_size=,
        sample_rate=, seed=, device=, sampler_type=, sigma_min=, sigma_max=, init_audio=(sr, tensor),
        init_noise_level=, mask_args=None, callback=, scale_phi=) -> tensor [batch, channels, n]

Job mapping:

* ``generate.loop``       ``<tags>, Wet|Dry, <fx>, <key>, <N> Bars, <N> BPM``; the clip length is
  ``bars * 4 * 60 / bpm`` seconds, the model window is that rounded up to whole seconds (and to the model's
  ``min_input_length``), the output is cut to the exact length.
* ``generate.sample``     One-shot prompt with register and note; a 2 s window (what the model was trained
  on) or longer when ``length_s`` asks for it; the tail silence is trimmed and the result cut to ``length_s``.
* ``generate.instrument`` Keybed model only: one generation per root note of ``plan_keybed(low, high, step)``
  and per velocity layer, a shared seed per layer, each written as ``<name>/samples/*.wav``; the zone
  list goes in ``Output.extra`` for post-processing to build the SFZ/DecentSampler bundle.
* ``transform.audio``     Audio-to-audio: ``init_audio`` with ``init_noise_level`` mapped from ``strength``
  (see :func:`noise_level`).

Cancellation is checked between generations and inside the sampler callback (v-objective models). The
rectified-flow sampler in RC does not call back, so cancelling there takes effect after the current clip.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from audiokit.instruments import plan_keybed

from . import _prompts as prompts
from . import _upstream as up
from .protocol import ModelInfo, Output, RunJob
from .runtime import CancelCheck, Family, Progress, check_cancel, main

log = logging.getLogger("forge.worker.foundation")

LICENCE = (
    "Stability AI Community Licence (RoyalCities Foundation-1; registration and revenue conditions apply to "
    "commercial use, see spec 6.2/6.7). RC Stable Audio Tools code: MIT"
)
MODELS: dict[str, dict[str, Any]] = {
    "foundation-1": {"jobs": ["generate.loop", "transform.audio"], "vram_gb": 7.0},
    "foundation-1.2-samples": {"jobs": ["generate.sample", "generate.loop", "transform.audio"], "vram_gb": 7.0},
    "foundation-1.2-keybeds": {"jobs": ["generate.instrument"], "vram_gb": 7.0},
}
DEFAULT_MAX_SECONDS = 47.0
TRAINED_BPMS = (100, 110, 120, 128, 130, 140, 150)
TRAINED_BARS = (4, 8)
ONESHOT_SECONDS = 2.0
BEATS_PER_BAR = 4
ZONE_GAIN_STEP_DB = 6.0
MAX_VELOCITY_LAYERS = 3
TAIL_THRESHOLD_DB = -60.0
TAIL_FRAME_MS = 10.0
TAIL_PAD_MS = 60.0
TAIL_MIN_KEEP_MS = 120.0
TAIL_FADE_MS = 8.0


# ---- pure helpers (unit tested) -------------------------------------------------------------------

def plan_window(clip_seconds: float, sample_rate: int, min_input_length: int | None) -> tuple[int, int]:
    """``(seconds_total, sample_size)`` for a clip: whole seconds for the conditioner, and a sample count that
    is a multiple of the model's ``min_input_length`` (RC's ``target_samples_for_generation``)."""
    if not math.isfinite(clip_seconds) or clip_seconds <= 0:
        raise ValueError("clip length must be positive")
    seconds_total = int(math.ceil(clip_seconds - 1e-9))
    samples = seconds_total * sample_rate
    if isinstance(min_input_length, int) and min_input_length > 0 and samples % min_input_length:
        samples += min_input_length - samples % min_input_length
    return seconds_total, samples


def loop_seconds(bars: int, bpm: float, beats_per_bar: int = BEATS_PER_BAR) -> float:
    return 60.0 / bpm * beats_per_bar * bars


def noise_level(strength: float, objective: str) -> float:
    """``init_noise_level`` for a job ``strength`` in 0..1 (higher changes the input more).

    ``v`` models take a sigma: RC's style-transfer slider runs 0.1..5 (default 0.9), so strength maps
    geometrically onto that range. ``rectified_flow`` models clamp it to at most 1, so strength maps linearly."""
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    if objective == "rectified_flow":
        return max(0.01, strength)
    return 0.1 * (5.0 / 0.1) ** strength


def trim_tail_silence(audio: np.ndarray, sr: int) -> np.ndarray:
    """Cut trailing silence (below -60 dBFS, 10 ms frames) keeping 60 ms of padding and at least 120 ms,
    then a short fade-out. Mirrors RC's ``trim_oneshot_trailing_silence``; the head is never touched."""
    n = audio.shape[1]
    frame = max(1, int(round(sr * TAIL_FRAME_MS / 1000.0)))
    threshold = 10.0 ** (TAIL_THRESHOLD_DB / 20.0)
    peaks = np.abs(audio).max(axis=0)
    last = 0
    for end in range(n, 0, -frame):
        if float(peaks[max(0, end - frame):end].max()) > threshold:
            last = end
            break
    keep = max(int(round(sr * TAIL_MIN_KEEP_MS / 1000.0)), last + int(round(sr * TAIL_PAD_MS / 1000.0)))
    return up.fade_out(audio[:, : min(n, keep)], sr, TAIL_FADE_MS)


def velocity_edges(layers: int) -> list[int]:
    return [round(i * 128 / layers) for i in range(layers + 1)]


def layer_seed(base: int, layer: int) -> int:
    """Seed of velocity layer ``layer`` for a candidate whose seed is ``base``: one seed per keybed layer."""
    return (base + layer * 1_000_003) % up.MAX_SEED


@dataclass
class _Loaded:
    model: Any
    config: dict[str, Any]
    sample_rate: int
    device: str
    min_input_length: int | None
    objective: str
    max_seconds: float


def _sat() -> SimpleNamespace:
    """Import torch and the three RC Stable Audio Tools entry points this worker uses (lazily)."""
    torch = up.import_module("torch")
    factory = up.import_module("stable_audio_tools.models.factory")
    utils = up.import_module("stable_audio_tools.models.utils")
    generation = up.import_module("stable_audio_tools.inference.generation")
    try:
        return SimpleNamespace(
            torch=torch,
            create_model_from_config=factory.create_model_from_config,
            load_ckpt_state_dict=utils.load_ckpt_state_dict,
            generate_diffusion_cond=generation.generate_diffusion_cond,
        )
    except AttributeError as exc:
        raise up.MissingDependency(f"stable_audio_tools API differs from what this adapter expects: {exc}") from exc


class FoundationFamily(Family):
    name = "foundation"

    def __init__(self) -> None:
        self._device = "cuda"
        self._dtype = "auto"
        self._steps = 75
        self._cfg = 7.0
        self._keybed_steps = 80
        self._keybed_cfg = 6.0
        self._sampler = "dpmpp-3m-sde"
        self._sigma_min = 0.03
        self._sigma_max = 500.0
        self._cfg_rescale = 0.0
        self._checkpoints: dict[str, Any] = {}
        self._config_error: str | None = None
        self._loaded: dict[str, _Loaded] = {}
        self._sat_cache: SimpleNamespace | None = None

    # ---- configuration and availability ----

    def configure(self, options: dict[str, Any]) -> None:
        self._config_error = None
        try:
            self._device = up.check_device(options.get("device", "cuda"))
            self._dtype = str(options.get("dtype", "auto"))
            if self._dtype not in ("auto", "float32", "float16", "bfloat16"):
                raise ValueError(f"unsupported dtype {self._dtype!r}")
            self._steps = up.integer(options, "steps", 75, 1, 500)
            self._keybed_steps = up.integer(options, "keybed_steps", 80, 1, 500)
            self._cfg = up.number(options, "cfg_scale", 7.0, 0.0, 25.0)
            self._keybed_cfg = up.number(options, "keybed_cfg_scale", 6.0, 0.0, 25.0)
            self._sigma_min = up.number(options, "sigma_min", 0.03, 0.0, 2.0)
            self._sigma_max = up.number(options, "sigma_max", 500.0, 0.0, 1000.0)
            self._cfg_rescale = up.number(options, "cfg_rescale", 0.0, 0.0, 1.0)
            self._sampler = str(options.get("sampler_type", "dpmpp-3m-sde"))[:64]
            checkpoints = options.get("checkpoints") or {}
            if not isinstance(checkpoints, dict):
                raise ValueError("options.checkpoints must be a mapping of model id to {config, ckpt}")
            self._checkpoints = dict(checkpoints)
        except ValueError as exc:
            self._config_error = f"invalid options: {exc}"

    def _paths(self, model: str) -> tuple[Path, Path]:
        entry = self._checkpoints.get(model)
        if not isinstance(entry, dict):
            raise up.MissingDependency(f"options.checkpoints.{model} is not configured ({{config, ckpt}} paths)")
        paths = []
        for key in ("config", "ckpt"):
            raw = entry.get(key)
            if not isinstance(raw, str) or not raw.strip() or "\x00" in raw:
                raise up.MissingDependency(f"options.checkpoints.{model}.{key} is not set")
            path = Path(raw).expanduser()
            if not path.is_file():
                raise up.MissingDependency(f"{key} file {str(path)!r} does not exist")
            paths.append(path)
        return paths[0], paths[1]

    def _availability(self, model: str) -> tuple[str | None, float, str, str | None]:
        """``(reason, max_seconds, checkpoint, hash)``; ``reason`` is ``None`` when the model can load."""
        if self._config_error:
            return self._config_error, DEFAULT_MAX_SECONDS, "", None
        missing = up.first_missing("torch", "stable_audio_tools")
        if missing:
            return f"{missing}; install RC-stable-audio-tools", DEFAULT_MAX_SECONDS, "", None
        try:
            config_path, ckpt_path = self._paths(model)
            config = up.read_json_file(config_path)
        except up.MissingDependency as exc:
            return str(exc), DEFAULT_MAX_SECONDS, "", None
        seconds = DEFAULT_MAX_SECONDS
        try:
            seconds = round(int(config["sample_size"]) / int(config["sample_rate"]), 2)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return f"{config_path.name} lacks sample_size/sample_rate", DEFAULT_MAX_SECONDS, str(ckpt_path), None
        return None, seconds, str(ckpt_path), up.fingerprint(ckpt_path)

    def models(self) -> list[ModelInfo]:
        out: list[ModelInfo] = []
        for model_id, spec in MODELS.items():
            reason, seconds, checkpoint, digest = self._availability(model_id)
            info: ModelInfo = {
                "id": model_id, "family": self.name, "job_types": list(spec["jobs"]), "available": reason is None,
                "licence": LICENCE, "checkpoint": checkpoint, "checkpoint_hash": digest,
                "vram_gb": float(spec["vram_gb"]), "max_seconds": seconds,
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
        reason, seconds, _, _ = self._availability(model)
        if reason:
            raise up.MissingDependency(reason)
        config_path, ckpt_path = self._paths(model)
        sat = self._sat_cache = _sat()
        torch = sat.torch
        config = up.read_json_file(config_path)
        sample_rate = int(config["sample_rate"])
        if not 8000 <= sample_rate <= 192000:
            raise ValueError(f"model sample_rate {sample_rate} is out of range")
        log.info("loading %s from %s", model, ckpt_path)
        net = sat.create_model_from_config(config)
        state = sat.load_ckpt_state_dict(str(ckpt_path))
        tensors = [v for v in state.values() if torch.is_tensor(v)]
        if tensors and all(t.dtype == torch.float16 for t in tensors):
            net.to(torch.float16)  # a half-precision checkpoint loads into a half-precision model
        net.load_state_dict(state)
        del state, tensors
        device = up.resolve_device(torch, self._device)
        net.to(device).eval().requires_grad_(False)
        if device != "cpu":
            net.to(self._compute_dtype(torch, device))
        min_input = getattr(net, "min_input_length", None)
        self._loaded[model] = _Loaded(
            net, config, sample_rate, device, min_input if isinstance(min_input, int) else None,
            str(getattr(net, "diffusion_objective", "v")), seconds,
        )
        up.free_accelerator_memory()

    def _compute_dtype(self, torch: Any, device: str) -> Any:
        if self._dtype != "auto":
            return getattr(torch, self._dtype)
        if device.startswith("cuda"):
            try:
                return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            except Exception as exc:  # old drivers raise instead of answering
                log.debug("bf16 probe failed: %s", exc)
                return torch.float16
        return torch.float16 if device == "mps" else torch.float32

    def unload(self, model: str) -> None:
        if self._loaded.pop(model, None) is not None:
            up.free_accelerator_memory()

    # ---- generation ----

    def _generate(
        self, loaded: _Loaded, *, prompt: str, clip_seconds: float, seed: int, steps: int, cfg: float,
        progress: up.Span, cancelled: CancelCheck, init: tuple[int, np.ndarray] | None = None,
        init_noise_level: float = 1.0,
    ) -> np.ndarray:
        """One diffusion run. Returns ``float32 (channels, frames)`` at the model rate, cut to ``clip_seconds``."""
        check_cancel(cancelled)
        sat = self._sat_cache or _sat()
        torch = sat.torch
        seconds_total, sample_size = plan_window(clip_seconds, loaded.sample_rate, loaded.min_input_length)
        if seconds_total > loaded.max_seconds + 1e-6:
            raise ValueError(f"{clip_seconds:.1f} s exceeds this model's maximum of {loaded.max_seconds:g} s")
        conditioning = [{"prompt": prompt, "seconds_start": 0.0, "seconds_total": float(seconds_total)}]

        def on_step(info: dict[str, Any]) -> None:
            check_cancel(cancelled)
            try:
                step = int(info.get("i", 0))
            except (TypeError, ValueError):
                step = 0
            progress((step + 1) / max(1, steps), f"step {step + 1}/{steps}")

        init_audio = None
        if init is not None:
            init_sr, init_arr = init
            init_audio = (int(init_sr), torch.from_numpy(np.ascontiguousarray(init_arr, dtype=np.float32)))
        log.info("generating %.2f s (window %d s) seed=%d prompt=%r", clip_seconds, seconds_total, seed, prompt)
        with torch.no_grad():
            out = sat.generate_diffusion_cond(
                loaded.model, steps=steps, cfg_scale=cfg, conditioning=conditioning, negative_conditioning=None,
                batch_size=1, sample_size=int(sample_size), sample_rate=loaded.sample_rate, seed=int(seed),
                device=loaded.device, sampler_type=self._sampler, sigma_min=self._sigma_min,
                sigma_max=self._sigma_max, init_audio=init_audio, init_noise_level=float(init_noise_level),
                mask_args=None, callback=on_step, scale_phi=self._cfg_rescale,
            )
        audio = up.tensor_to_audio(out)
        check_cancel(cancelled)
        up.free_accelerator_memory()
        return audio[:, : max(1, int(round(clip_seconds * loaded.sample_rate)))]

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
        spec = prompts.spec_from_tags(tags, str(params.get("prompt") or ""))
        keybed = job["type"] == "generate.instrument"
        for token in prompts.unknown_tokens(spec, keybed=keybed):
            log.warning("tag %r is not in the Foundation vocabulary; passing it through unchanged", token)
        span = up.Span(progress)
        span(0.0, "started")
        handler = {
            "generate.loop": self._loop, "generate.sample": self._sample, "generate.instrument": self._instrument,
            "transform.audio": self._transform,
        }[job["type"]]
        outputs = handler(loaded, job, params, spec, out_dir, seed, count, span, cancelled)
        span(1.0, "done")
        return outputs

    def _audio_output(self, out_dir: Path, candidate: int, audio: np.ndarray, sr: int, extra: dict[str, Any]) -> Output:
        path = up.write_audio(out_dir, f"candidate_{candidate}.wav", audio, sr)
        return {"path": str(path), "kind": "audio", "candidate": candidate, "extra": extra}

    def _loop(
        self, loaded: _Loaded, job: RunJob, params: Any, spec: prompts.FoundationSpec, out_dir: Path, seed: int,
        count: int, span: up.Span, cancelled: CancelCheck,
    ) -> list[Output]:
        if "bpm" not in params:
            raise ValueError("generate.loop needs bpm")
        bars = up.integer(params, "bars", 4, 1, 16)
        bpm = up.number(params, "bpm", 120.0, 40.0, 240.0)
        if bars not in TRAINED_BARS or round(bpm) not in TRAINED_BPMS:
            log.warning("bars=%d bpm=%g is outside what Foundation was trained on (bars %s, bpm %s)", bars, bpm, TRAINED_BARS, TRAINED_BPMS)
        prompt = prompts.build_loop_prompt(spec, bars=bars, bpm=bpm, key=params.get("key"))
        seconds = loop_seconds(bars, bpm)
        outputs: list[Output] = []
        for candidate in range(count):
            audio = self._generate(
                loaded, prompt=prompt, clip_seconds=seconds, seed=up.candidate_seed(seed, candidate),
                steps=self._steps, cfg=self._cfg, progress=span.part(candidate, count), cancelled=cancelled,
            )
            outputs.append(self._audio_output(out_dir, candidate, audio, loaded.sample_rate, {"conditioning_prompt": prompt}))
        return outputs

    def _sample(
        self, loaded: _Loaded, job: RunJob, params: Any, spec: prompts.FoundationSpec, out_dir: Path, seed: int,
        count: int, span: up.Span, cancelled: CancelCheck,
    ) -> list[Output]:
        note = up.integer(params, "note", 60, 0, 127)
        length = up.number(params, "length_s", 2.0, 0.05, 20.0)
        prompt = prompts.build_oneshot_prompt(spec, midi_note=note)
        window = max(ONESHOT_SECONDS, length)
        if window > ONESHOT_SECONDS:
            log.warning("length_s=%g is longer than the %g s one-shots Foundation was trained on", length, ONESHOT_SECONDS)
        outputs: list[Output] = []
        for candidate in range(count):
            audio = self._generate(
                loaded, prompt=prompt, clip_seconds=window, seed=up.candidate_seed(seed, candidate),
                steps=self._steps, cfg=self._cfg, progress=span.part(candidate, count), cancelled=cancelled,
            )
            audio = trim_tail_silence(audio, loaded.sample_rate)
            audio = up.fade_out(audio[:, : max(1, int(round(length * loaded.sample_rate)))], loaded.sample_rate, TAIL_FADE_MS)
            outputs.append(self._audio_output(out_dir, candidate, audio, loaded.sample_rate, {"conditioning_prompt": prompt, "note": note}))
        return outputs

    def _transform(
        self, loaded: _Loaded, job: RunJob, params: Any, spec: prompts.FoundationSpec, out_dir: Path, seed: int,
        count: int, span: up.Span, cancelled: CancelCheck,
    ) -> list[Output]:
        strength = up.number(params, "strength", 0.5, 0.0, 1.0)
        audio_in, in_sr = up.read_audio_input(job, "audio", max_seconds=loaded.max_seconds)
        clip_seconds = audio_in.shape[1] / in_sr
        prompt = prompts.build_descriptor_prompt(spec)
        level = noise_level(strength, loaded.objective)
        outputs: list[Output] = []
        for candidate in range(count):
            audio = self._generate(
                loaded, prompt=prompt, clip_seconds=clip_seconds, seed=up.candidate_seed(seed, candidate),
                steps=self._steps, cfg=self._cfg, progress=span.part(candidate, count), cancelled=cancelled,
                init=(in_sr, audio_in), init_noise_level=level,
            )
            outputs.append(self._audio_output(out_dir, candidate, audio, loaded.sample_rate, {"conditioning_prompt": prompt, "init_noise_level": round(level, 4)}))
        return outputs

    def _instrument(
        self, loaded: _Loaded, job: RunJob, params: Any, spec: prompts.FoundationSpec, out_dir: Path, seed: int,
        count: int, span: up.Span, cancelled: CancelCheck,
    ) -> list[Output]:
        low = up.integer(params, "note_low", 36, 0, 127)
        high = up.integer(params, "note_high", 84, 0, 127)
        step = up.integer(params, "step", 3, 1, 12)
        layers = up.integer(params, "velocity_layers", 1, 1, MAX_VELOCITY_LAYERS)
        length = up.number(params, "length_s", 3.0, 0.05, 20.0)
        name = up.safe_name(str(params.get("name") or "instrument"), "instrument")
        plan = plan_keybed(low, high, step)
        if len(plan) * layers > 256:
            raise ValueError(f"{len(plan) * layers} zones requested (at most 256)")
        edges = velocity_edges(layers)
        total = len(plan) * layers * count
        done = 0
        outputs: list[Output] = []
        for candidate in range(count):
            folder = name if candidate == 0 else f"{name}_{candidate}"
            base = up.candidate_seed(seed, candidate)
            zones: list[dict[str, Any]] = []
            for layer in range(layers):
                lseed = layer_seed(base, layer)  # the same seed for every root of this layer (shared timbre)
                for root, key_low, key_high in plan:
                    prompt = prompts.build_keybed_prompt(spec, midi_note=root)
                    audio = self._generate(
                        loaded, prompt=prompt, clip_seconds=length, seed=lseed, steps=self._keybed_steps,
                        cfg=self._keybed_cfg, progress=up.Span(lambda f, m: None), cancelled=cancelled,
                    )
                    audio = trim_tail_silence(audio, loaded.sample_rate)
                    relative = f"{folder}/samples/{name}_{root:03d}_v{layer + 1}.wav"
                    up.write_audio(out_dir, relative, audio, loaded.sample_rate)
                    zone: dict[str, Any] = {
                        "sample": f"samples/{name}_{root:03d}_v{layer + 1}.wav",
                        "root_note": root, "key_low": key_low, "key_high": key_high,
                        "vel_low": edges[layer], "vel_high": edges[layer + 1] - 1,
                    }
                    gain = -ZONE_GAIN_STEP_DB * (layers - 1 - layer)
                    if gain:
                        zone["gain_db"] = gain
                    zones.append(zone)
                    done += 1
                    span(done / total, f"zone {done}/{total} (note {root})")
            bundle = up.confine(out_dir, folder)
            outputs.append({
                "path": str(bundle), "kind": "instrument", "candidate": candidate,
                "extra": {"name": name, "zones": zones, "keybed_seed": base, "velocity_layers": layers},
            })
        return outputs


if __name__ == "__main__":
    main(FoundationFamily())
