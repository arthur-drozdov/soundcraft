"""The ``acestep`` family: ACE-Step 1.5 base and XL-base (the variants that support lego and extract).

Run as ``python -m forge.workers.acestep`` in the environment made by ``uv sync`` in a clone of
https://github.com/ace-step/ACE-Step-1.5 (Python 3.11 or 3.12; the project installs the ``acestep`` package).

``hello`` options (``deploy/forge.example.yaml``)::

    device: cuda
    checkpoints:
      ace-step-1.5-base: ACE-Step/acestep-v15-base       # the directory name under checkpoints/ is the last part
      ace-step-1.5-xl-base: ACE-Step/acestep-v15-xl-base
    lm: acestep-5Hz-lm-1.7B       # optional 5 Hz language model used for generate.loop planning

Optional extras: ``project_root`` (default ``$ACESTEP_PROJECT_ROOT`` or the worker's working directory;
weights live in ``<project_root>/checkpoints`` unless ``ACESTEP_CHECKPOINTS_DIR`` is set),
``lm_backend`` (``pt`` by default; ``vllm`` needs nano-vllm), ``steps`` (50, the base-model default),
``guidance_scale`` (7.0).

Network: ``AceStepHandler.initialize_service`` downloads missing weights from Hugging Face (or ModelScope)
on first load, as upstream does. Nothing else touches the network.

Upstream calls (source: ACE-Step-1.5 @ ca1e85f, ``docs/en/INFERENCE.md`` and ``acestep/inference.py``; none run
here)::

    acestep.handler.AceStepHandler().initialize_service(project_root, config_path, device) -> (status, ok)
    acestep.llm_inference.LLMHandler().initialize(checkpoint_dir, lm_model_path, backend, device) -> (status, ok)
    acestep.inference.generate_music(dit, llm, GenerationParams(...), GenerationConfig(...), save_dir=None,
                                     progress=fn(fraction, desc=...)) -> GenerationResult
        result.audios[i] = {"path", "tensor" [channels, n] float32 cpu, "sample_rate", "key", "params"}

``save_dir=None`` makes the library write nothing; the audio is taken from the returned tensors and written
by this worker inside ``out_dir``.

Job mapping:

* ``generate.loop``  ``task_type="text2music"``, instrumental, ``bpm``, ``keyscale``, ``duration`` = loop length
  (a whole number of repetitions when that is under ACE-Step's 10 s minimum, cut back to the loop length).
* ``layer.add``      ``task_type="lego"`` with ``instruction="Generate the <track> track based on the audio
  context:"``. The track comes from ``tags.track`` or a keyword of ``description`` (``pad`` -> ``synth``).
* ``stems.split``    ``task_type="extract"``, one run per requested stem
  (``Extract the <track> track from the audio:``); no stems requested means ``vocals, drums, bass``.

Seeds are passed modulo 2**32 (ACE-Step's range). Cancellation raises from the progress callback and is
re-checked after ``generate_music`` returns (it reports exceptions as a failed result).
"""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import _prompts as prompts
from . import _upstream as up
from .protocol import ModelInfo, Output, RunJob
from .runtime import CancelCheck, Cancelled, Family, Progress, check_cancel, main

log = logging.getLogger("forge.worker.acestep")

LICENCE = "MIT (ACE-Step 1.5; spec 6.2)"
MODELS: dict[str, dict[str, Any]] = {
    "ace-step-1.5-base": {"default": "ACE-Step/acestep-v15-base", "vram_gb": 12.0},
    "ace-step-1.5-xl-base": {"default": "ACE-Step/acestep-v15-xl-base", "vram_gb": 20.0},
}
JOB_TYPES = ["layer.add", "stems.split", "generate.loop"]
MAX_SECONDS = 600.0
MIN_GENERATE_SECONDS = 10.0
DEFAULT_STEMS = ("vocals", "drums", "bass")
EXTRACT_INSTRUCTION = "Extract the {TRACK_NAME} track from the audio:"
LEGO_INSTRUCTION = "Generate the {TRACK_NAME} track based on the audio context:"
LEGO_DEFAULT_INSTRUCTION = "Generate the track based on the audio context:"
SEED_MODULUS = 2**32


def make_dataclass(cls: Any, **values: Any) -> Any:
    """Instantiate an upstream dataclass with only the fields it has (tolerates upstream renames, with a warning)."""
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(values) - names)
    if unknown:
        log.warning("%s has no field(s) %s in this ACE-Step version; skipping them", cls.__name__, unknown)
    return cls(**{k: v for k, v in values.items() if k in names})


def variant_of(checkpoint: str) -> str:
    """``ACE-Step/acestep-v15-base`` -> ``acestep-v15-base`` (the directory ``initialize_service`` expects)."""
    name = checkpoint.strip().rstrip("/").rsplit("/", 1)[-1]
    if not name or name.startswith(".") or "\x00" in name or "\\" in name:
        raise ValueError(f"invalid ACE-Step checkpoint {checkpoint!r}")
    return name


def loop_generation_plan(bars: int, bpm: float) -> tuple[float, float]:
    """``(loop_seconds, generate_seconds)``: ACE-Step produces at least 10 s, so short loops are generated as a
    whole number of repetitions and cut back afterwards."""
    loop = 60.0 / bpm * 4 * bars
    reps = 1
    while loop * reps < MIN_GENERATE_SECONDS:
        reps += 1
    return loop, min(MAX_SECONDS, loop * reps)


@dataclass
class _Loaded:
    dit: Any
    llm: Any
    inference: Any
    device: str


class AceStepFamily(Family):
    name = "acestep"

    def __init__(self) -> None:
        self._device = "cuda"
        self._lm: str | None = None
        self._lm_backend = "pt"
        self._steps = 50
        self._guidance = 7.0
        self._project_root: str | None = None
        self._checkpoints: dict[str, str] = {}
        self._config_error: str | None = None
        self._loaded: dict[str, _Loaded] = {}

    # ---- configuration and availability ----

    def configure(self, options: dict[str, Any]) -> None:
        self._config_error = None
        try:
            self._device = up.check_device(options.get("device", "cuda"))
            self._steps = up.integer(options, "steps", 50, 1, 200)
            self._guidance = up.number(options, "guidance_scale", 7.0, 1.0, 15.0)
            lm = options.get("lm")
            self._lm = None if not lm else str(lm)
            if self._lm is not None and (not self._lm.replace("-", "").replace(".", "").replace("_", "").isalnum() or len(self._lm) > 128):
                raise ValueError(f"invalid lm name {self._lm!r}")
            self._lm_backend = str(options.get("lm_backend", "pt"))
            if self._lm_backend not in ("pt", "vllm"):
                raise ValueError("lm_backend must be 'pt' or 'vllm'")
            root = options.get("project_root")
            self._project_root = str(root) if root else None
            raw = options.get("checkpoints") or {}
            if not isinstance(raw, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in raw.items()):
                raise ValueError("options.checkpoints must map model ids to Hugging Face ids")
            self._checkpoints = dict(raw)
        except ValueError as exc:
            self._config_error = f"invalid options: {exc}"

    def _checkpoint(self, model: str) -> str:
        return self._checkpoints.get(model) or str(MODELS[model]["default"])

    def models(self) -> list[ModelInfo]:
        missing = up.first_missing("torch", "acestep")
        out: list[ModelInfo] = []
        for model_id, spec in MODELS.items():
            reason = self._config_error or (f"{missing}; install https://github.com/ace-step/ACE-Step-1.5 (uv sync)" if missing else None)
            info: ModelInfo = {
                "id": model_id, "family": self.name, "job_types": list(JOB_TYPES), "available": reason is None,
                "licence": LICENCE, "checkpoint": self._checkpoint(model_id), "checkpoint_hash": None,
                "vram_gb": float(spec["vram_gb"]) + (4.0 if self._lm else 0.0), "max_seconds": MAX_SECONDS,
            }
            if reason:
                info["reason"] = reason
            out.append(info)
        return out

    # ---- memory ----

    def _roots(self) -> tuple[str, str]:
        root = self._project_root or os.environ.get("ACESTEP_PROJECT_ROOT") or os.getcwd()
        checkpoints = os.environ.get("ACESTEP_CHECKPOINTS_DIR") or os.path.join(root, "checkpoints")
        return root, checkpoints

    def load(self, model: str) -> None:
        if model not in MODELS:
            raise ValueError(f"unknown model {model!r}")
        if model in self._loaded:
            return
        if self._config_error:
            raise up.MissingDependency(self._config_error)
        reason = up.first_missing("torch", "acestep")
        if reason:
            raise up.MissingDependency(reason)
        torch = up.import_module("torch")
        handler_mod = up.import_module("acestep.handler")
        inference = up.import_module("acestep.inference")
        device = up.resolve_device(torch, self._device)
        root, checkpoint_dir = self._roots()
        variant = variant_of(self._checkpoint(model))
        log.info("initialising %s (%s) on %s; weights are downloaded on first use if missing", model, variant, device)
        try:
            dit = handler_mod.AceStepHandler()
            status, ok = dit.initialize_service(project_root=root, config_path=variant, device=device)
        except AttributeError as exc:
            raise up.MissingDependency(f"acestep API differs from what this adapter expects: {exc}") from exc
        if not ok:
            raise RuntimeError(f"ACE-Step could not initialise {variant}: {up.tail(str(status))}")
        llm = self._load_lm(device, checkpoint_dir)
        self._loaded[model] = _Loaded(dit, llm, inference, device)
        up.free_accelerator_memory()

    def _load_lm(self, device: str, checkpoint_dir: str) -> Any:
        """The 5 Hz language model, or ``None`` (with a warning) when it is not configured or cannot load.

        The model is optional (it only plans ``generate.loop``), so any failure here degrades to DiT-only."""
        if not self._lm:
            return None
        try:
            llm_mod = up.import_module("acestep.llm_inference")
            try:  # the downloader is optional: initialize() reports a missing model itself
                downloads = up.import_module("acestep.api.model_download")
                downloads.ensure_model_downloaded(self._lm, checkpoint_dir)
            except Exception as exc:  # network or layout problems must not fail the DiT load
                log.warning("could not fetch language model %s: %s", self._lm, exc)
            llm = llm_mod.LLMHandler()
            status, ok = llm.initialize(
                checkpoint_dir=checkpoint_dir, lm_model_path=self._lm, backend=self._lm_backend, device=device,
            )
        except Exception as exc:  # optional component: run without it
            log.warning("language model %s unavailable: %s", self._lm, exc)
            return None
        if not ok:
            log.warning("language model %s failed to initialise: %s", self._lm, up.tail(str(status)))
            return None
        return llm

    def unload(self, model: str) -> None:
        loaded = self._loaded.pop(model, None)
        if loaded is not None:
            del loaded
            up.free_accelerator_memory()

    # ---- generation ----

    def _generate(
        self, loaded: _Loaded, *, params: dict[str, Any], seed: int, progress: up.Span, cancelled: CancelCheck,
        use_lm: bool = False,
    ) -> tuple[np.ndarray, int]:
        """One ``generate_music`` call. Returns ``(float32 (channels, frames), sample_rate)``."""
        check_cancel(cancelled)
        inference = loaded.inference

        def report(fraction: float = 0.0, *_: Any, desc: str = "", **__: Any) -> None:
            if cancelled():
                raise Cancelled("cancelled")
            progress(float(fraction), str(desc))

        gen_params = make_dataclass(
            inference.GenerationParams, seed=int(seed % SEED_MODULUS), inference_steps=self._steps,
            guidance_scale=self._guidance, thinking=bool(use_lm and loaded.llm is not None),
            use_cot_caption=False, use_cot_language=False, use_cot_metas=False, enable_normalization=False, **params,
        )
        gen_config = make_dataclass(
            inference.GenerationConfig, batch_size=1, use_random_seed=False, seeds=[int(seed % SEED_MODULUS)],
            audio_format="wav32",
        )
        result = inference.generate_music(loaded.dit, loaded.llm, gen_params, gen_config, save_dir=None, progress=report)
        check_cancel(cancelled)
        if not getattr(result, "success", False):
            raise RuntimeError(f"ACE-Step failed: {up.tail(str(getattr(result, 'error', None) or getattr(result, 'status_message', '')))}")
        audios = getattr(result, "audios", None) or []
        if not audios:
            raise RuntimeError("ACE-Step returned no audio")
        first = audios[0]
        tensor = first.get("tensor") if isinstance(first, dict) else None
        if tensor is None:
            raise RuntimeError("ACE-Step returned no audio tensor")
        rate = int(first.get("sample_rate") or 48000)
        up.free_accelerator_memory()
        return up.tensor_to_audio(tensor), rate

    def run(self, job: RunJob, progress: Progress, cancelled: CancelCheck) -> list[Output]:
        model_id = str(job.get("model"))
        if model_id not in MODELS:
            raise ValueError(f"unknown model {model_id!r}")
        loaded = self._loaded.get(model_id)
        if loaded is None:
            raise RuntimeError(f"model {model_id!r} is not loaded")
        params, out_dir = up.check_job(job, JOB_TYPES)
        seed, count = up.job_seed(job), up.candidate_count(params)
        tags: dict[str, Any] = params["tags"] if isinstance(params.get("tags"), dict) else {}
        prompt = str(params.get("prompt") or "")
        span = up.Span(progress)
        span(0.0, "started")
        jtype = job["type"]
        outputs: list[Output] = []
        if jtype == "generate.loop":
            outputs = self._loop(loaded, job, params, tags, prompt, out_dir, seed, count, span, cancelled)
        elif jtype == "layer.add":
            outputs = self._layer(loaded, job, params, tags, prompt, out_dir, seed, count, span, cancelled)
        else:
            outputs = self._extract(loaded, job, params, prompt, out_dir, seed, count, span, cancelled)
        span(1.0, "done")
        return outputs

    @staticmethod
    def _write(out_dir: Path, name: str, audio: np.ndarray, rate: int, candidate: int, extra: dict[str, Any]) -> Output:
        path = up.write_audio(out_dir, name, audio, rate)
        return {"path": str(path), "kind": "audio", "candidate": candidate, "extra": extra}

    def _loop(
        self, loaded: _Loaded, job: RunJob, params: Any, tags: dict[str, Any], prompt: str, out_dir: Path, seed: int,
        count: int, span: up.Span, cancelled: CancelCheck,
    ) -> list[Output]:
        if "bpm" not in params:
            raise ValueError("generate.loop needs bpm")
        bars, bpm = up.integer(params, "bars", 4, 1, 16), up.number(params, "bpm", 120.0, 40.0, 240.0)
        loop_s, gen_s = loop_generation_plan(bars, bpm)
        caption = prompts.ace_caption(prompt, tags, extra=["instrumental", "seamless loop"])
        outputs: list[Output] = []
        for candidate in range(count):
            audio, rate = self._generate(
                loaded, seed=up.candidate_seed(seed, candidate), progress=span.part(candidate, count), cancelled=cancelled,
                use_lm=True,
                params={
                    "task_type": "text2music", "caption": caption, "lyrics": "[Instrumental]", "instrumental": True,
                    "bpm": int(round(bpm)), "keyscale": prompts.ace_key(params.get("key")), "timesignature": "4",
                    "duration": float(gen_s),
                },
            )
            audio = audio[:, : max(1, int(round(loop_s * rate)))]
            outputs.append(self._write(out_dir, f"candidate_{candidate}.wav", audio, rate, candidate, {"caption": caption, "generated_s": round(gen_s, 3)}))
        return outputs

    def _layer(
        self, loaded: _Loaded, job: RunJob, params: Any, tags: dict[str, Any], prompt: str, out_dir: Path, seed: int,
        count: int, span: up.Span, cancelled: CancelCheck,
    ) -> list[Output]:
        description = str(params.get("description") or "").strip()
        if not description:
            raise ValueError("layer.add needs a description")
        up.read_audio_input(job, "audio", max_seconds=MAX_SECONDS)  # validates the WAV and its length
        src = str(up.input_path(job, "audio"))
        track = prompts.ace_track_from_description(description, tags)
        instruction = LEGO_INSTRUCTION.format(TRACK_NAME=track) if track else LEGO_DEFAULT_INSTRUCTION
        caption = prompts.ace_caption(description, tags, extra=[prompts.clean_prompt(prompt)] if prompt else [])
        outputs: list[Output] = []
        for candidate in range(count):
            audio, rate = self._generate(
                loaded, seed=up.candidate_seed(seed, candidate), progress=span.part(candidate, count), cancelled=cancelled,
                params={
                    "task_type": "lego", "instruction": instruction, "src_audio": src, "caption": caption,
                    "repainting_start": 0.0, "repainting_end": -1, "duration": -1.0,
                },
            )
            outputs.append(self._write(out_dir, f"candidate_{candidate}.wav", audio, rate, candidate, {"track": track or "", "instruction": instruction}))
        return outputs

    def _extract(
        self, loaded: _Loaded, job: RunJob, params: Any, prompt: str, out_dir: Path, seed: int, count: int,
        span: up.Span, cancelled: CancelCheck,
    ) -> list[Output]:
        requested = params.get("stems") or []
        if not isinstance(requested, list) or len(requested) > 16:
            raise ValueError("stems must be a list of at most 16 names")
        stems = list(dict.fromkeys(prompts.ace_track(str(s)) for s in requested)) or list(DEFAULT_STEMS)
        up.read_audio_input(job, "audio", max_seconds=MAX_SECONDS)
        src = str(up.input_path(job, "audio"))
        outputs: list[Output] = []
        slices = count * len(stems)
        index = 0
        for candidate in range(count):
            for stem in stems:
                audio, rate = self._generate(
                    loaded, seed=up.candidate_seed(seed, candidate), progress=span.part(index, slices), cancelled=cancelled,
                    params={
                        "task_type": "extract", "instruction": EXTRACT_INSTRUCTION.format(TRACK_NAME=stem), "src_audio": src,
                        "caption": prompts.clean_prompt(prompt), "repainting_start": 0.0, "repainting_end": -1, "duration": -1.0,
                    },
                )
                outputs.append(self._write(out_dir, f"candidate_{candidate}_{up.safe_name(stem)}.wav", audio, rate, candidate, {"stem": stem}))
                index += 1
        return outputs


if __name__ == "__main__":
    main(AceStepFamily())
