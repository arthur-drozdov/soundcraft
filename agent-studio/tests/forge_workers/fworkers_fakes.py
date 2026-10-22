"""Helpers for the model-family worker tests.

The real packages (torch, stable-audio-tools, stable_audio_3, acestep, basic_pitch) are not installed in
the test environment. These fakes mimic only the call signatures the adapters use, so the tests exercise the
adapters' own code: option parsing, prompt building, parameter mapping, file layout, cancellation and the
JSON-lines protocol. They say nothing about whether the real packages behave as the adapters assume; that is
what the "UNVERIFIED" list in each worker module is for. Audio is synthesised, never committed.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import os
import queue
import stat
import sys
import textwrap
import threading
import time
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from audiokit import wav
from forge.workers import runtime

SR = 44100


# ---------------------------------------------------------------- audio and jobs

def tone(seconds: float = 1.0, sr: int = SR, freq: float = 440.0, channels: int = 2) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    mono = (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.tile(mono, (channels, 1))


def make_wav(path: Path, seconds: float = 1.0, sr: int = SR, freq: float = 440.0, channels: int = 2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    wav.write(path, tone(seconds, sr, freq, channels), sr, "16")
    return path


def make_job(
    tmp_path: Path, job_type: str, model: str, params: dict[str, Any] | None = None, inputs: dict[str, str] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    return {
        "job_id": "j_test", "type": job_type, "model": model, "params": params or {}, "inputs": inputs or {},
        "out_dir": str(out), "seed": seed,
    }


def run_job(family: Any, job: dict[str, Any], cancelled: Callable[[], bool] = lambda: False) -> tuple[list[dict[str, Any]], list[tuple[float, str]]]:
    """Run a job directly on a loaded family, returning ``(outputs, progress events)``."""
    events: list[tuple[float, str]] = []
    outputs = family.run(job, lambda f, m="": events.append((f, m)), cancelled)
    return outputs, events


def inside(root: str | Path, path: str | Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def listing(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


# ---------------------------------------------------------------- fake torch


class FakeTensor:
    """A numpy-backed stand-in with just the methods ``tensor_to_audio`` and the adapters call."""

    def __init__(self, data: Any, dtype: str = "float32", device: str = "cpu") -> None:
        self.data = np.asarray(data)
        self.dtype = dtype
        self.device = device

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    def dim(self) -> int:
        return int(self.data.ndim)

    def detach(self) -> FakeTensor:
        return self

    def float(self) -> FakeTensor:
        return FakeTensor(self.data.astype(np.float32), "float32", self.device)

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.data


class FakeCuda:
    def __init__(self, available: bool, bf16: bool) -> None:
        self.available, self.bf16, self.empty_cache_calls = available, bf16, 0

    def is_available(self) -> bool:
        return self.available

    def is_bf16_supported(self) -> bool:
        return self.bf16

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


def install_torch(monkeypatch: pytest.MonkeyPatch, cuda: bool = False, bf16: bool = True) -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.float16, torch.float32, torch.bfloat16 = "float16", "float32", "bfloat16"  # type: ignore[attr-defined]
    torch.cuda = FakeCuda(cuda, bf16)  # type: ignore[attr-defined]
    torch.Tensor = FakeTensor  # type: ignore[attr-defined]  # scipy's array-API probe looks for torch.Tensor
    torch.is_tensor = lambda x: isinstance(x, FakeTensor)  # type: ignore[attr-defined]
    torch.from_numpy = lambda a: FakeTensor(a)  # type: ignore[attr-defined]

    @contextlib.contextmanager
    def no_grad() -> Iterator[None]:
        yield

    torch.no_grad = no_grad  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def module(name: str, **attrs: Any) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def put(monkeypatch: pytest.MonkeyPatch, *mods: types.ModuleType) -> None:
    for mod in mods:
        monkeypatch.setitem(sys.modules, mod.__name__, mod)


def block(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Make ``import <name>`` fail even if the package were installed."""
    for name in names:
        monkeypatch.setitem(sys.modules, name, None)  # type: ignore[arg-type]


def decaying_tone(n: int, sr: int, seed: int) -> np.ndarray:
    """Two channels: a tone whose pitch depends on the seed, silent after 0.5 s."""
    t = np.arange(n) / sr
    env = np.exp(-t / 0.15) * (t < 0.5)
    wave = (0.5 * np.sin(2 * np.pi * (200.0 + seed % 97) * t) * env).astype(np.float32)
    return np.stack([wave, wave])


# ---------------------------------------------------------------- fake RC Stable Audio Tools


class FakeSat:
    """Records every call; the models it builds are :class:`FakeSatModel`."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.models: list[FakeSatModel] = []
        self.states: list[dict[str, Any]] = []
        self.configs: list[dict[str, Any]] = []
        self.state_dtype = "float32"
        self.sample_rate = SR
        self.step_delay = 0.0


class FakeSatModel:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.sample_rate = config["sample_rate"]
        self.min_input_length = config.get("min_input_length")
        self.diffusion_objective = config.get("diffusion_objective", "v")
        self.pretransform = None
        self.events: list[tuple[str, Any]] = []
        self.state: dict[str, Any] | None = None

    def to(self, target: Any) -> FakeSatModel:
        self.events.append(("to", target))
        return self

    def eval(self) -> FakeSatModel:
        self.events.append(("eval", None))
        return self

    def requires_grad_(self, flag: bool) -> FakeSatModel:
        self.events.append(("requires_grad_", flag))
        return self

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.state = state
        self.events.append(("load_state_dict", len(state)))


def install_sat(monkeypatch: pytest.MonkeyPatch, cuda: bool = False) -> FakeSat:
    install_torch(monkeypatch, cuda=cuda)
    sat = FakeSat()

    def create_model_from_config(config: dict[str, Any]) -> FakeSatModel:
        sat.configs.append(config)
        model = FakeSatModel(config)
        sat.models.append(model)
        return model

    def load_ckpt_state_dict(path: str) -> dict[str, Any]:
        sat.states.append({"path": path})
        return {"layer.weight": FakeTensor(np.zeros(3), sat.state_dtype)}

    def generate_diffusion_cond(model: FakeSatModel, **kw: Any) -> FakeTensor:
        sat.calls.append({"model": model, **kw})
        callback = kw.get("callback")
        for i in range(int(kw["steps"])):
            if callback is not None:
                callback({"i": i, "denoised": None})
            if sat.step_delay:
                time.sleep(sat.step_delay)
        n, sr = int(kw["sample_size"]), int(kw["sample_rate"])
        return FakeTensor(decaying_tone(n, sr, int(kw["seed"]))[None])

    put(
        monkeypatch,
        module("stable_audio_tools"),
        module("stable_audio_tools.models"),
        module("stable_audio_tools.models.factory", create_model_from_config=create_model_from_config),
        module("stable_audio_tools.models.utils", load_ckpt_state_dict=load_ckpt_state_dict),
        module("stable_audio_tools.inference"),
        module("stable_audio_tools.inference.generation", generate_diffusion_cond=generate_diffusion_cond),
    )
    return sat


def foundation_options(tmp_path: Path, models: tuple[str, ...] = ("foundation-1", "foundation-1.2-samples", "foundation-1.2-keybeds"), **extra: Any) -> dict[str, Any]:
    """Checkpoint files for the Foundation models (config JSON + a placeholder checkpoint) and the matching options."""
    checkpoints: dict[str, Any] = {}
    for model in models:
        folder = tmp_path / "models" / model
        folder.mkdir(parents=True, exist_ok=True)
        config = {"sample_rate": SR, "sample_size": SR * 47, "min_input_length": 4096, "model_type": "diffusion_cond"}
        (folder / "model_config.json").write_text(json.dumps(config))
        (folder / "model.ckpt").write_bytes(b"checkpoint-" + model.encode())
        checkpoints[model] = {"config": str(folder / "model_config.json"), "ckpt": str(folder / "model.ckpt")}
    return {"device": "cpu", "checkpoints": checkpoints, "steps": 3, "keybed_steps": 2, **extra}


# ---------------------------------------------------------------- fake Stable Audio 3


class FakeSA3:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.loads: list[tuple[str, Any]] = []
        self.local_loads: list[dict[str, Any]] = []


def install_sa3(monkeypatch: pytest.MonkeyPatch, cuda: bool = False) -> FakeSA3:
    install_torch(monkeypatch, cuda=cuda)
    state = FakeSA3()

    class StableAudioModel:
        def __init__(self, model: Any, model_config: Any, device: str, model_half: bool) -> None:
            self.model, self.model_config, self.device, self.model_half = model, model_config, device, model_half
            self.model.sample_rate = getattr(self.model, "sample_rate", SR)

        @staticmethod
        def from_pretrained(model_name: str, device: str | None = None, model_half: bool = True) -> StableAudioModel:
            state.loads.append((model_name, {"device": device, "model_half": model_half}))
            return StableAudioModel(types.SimpleNamespace(sample_rate=SR), {"name": model_name}, str(device), model_half)

        def generate(self, **kw: Any) -> FakeTensor:
            state.calls.append(kw)
            callback = kw.get("callback")
            for i in range(int(kw["steps"])):
                if callback is not None:
                    callback({"i": i})
            n = int(float(kw["duration"]) * SR) + 1000  # real SA3 returns a little more than asked for sometimes
            return FakeTensor(decaying_tone(n, SR, int(kw["seed"]))[None])

    def load_diffusion_cond(config: Any, ckpt_path: str, device: str = "cuda", model_half: bool = False) -> Any:
        state.local_loads.append({"config": config, "ckpt": ckpt_path, "device": device, "model_half": model_half})
        return types.SimpleNamespace(sample_rate=SR)

    put(
        monkeypatch,
        module("stable_audio_3", StableAudioModel=StableAudioModel),
        module("stable_audio_3.loading_utils", load_diffusion_cond=load_diffusion_cond),
    )
    return state


# ---------------------------------------------------------------- fake ACE-Step


@dataclasses.dataclass
class AceParams:
    task_type: str = "text2music"
    instruction: str = "Fill the audio semantic mask based on the given conditions:"
    reference_audio: str | None = None
    src_audio: str | None = None
    caption: str = ""
    lyrics: str = ""
    instrumental: bool = False
    vocal_language: str = "unknown"
    bpm: int | None = None
    keyscale: str = ""
    timesignature: str = ""
    duration: float = -1.0
    enable_normalization: bool = True
    inference_steps: int = 8
    seed: int = -1
    guidance_scale: float = 7.0
    repainting_start: float = 0.0
    repainting_end: float = -1
    thinking: bool = True
    use_cot_caption: bool = True
    use_cot_language: bool = True
    use_cot_metas: bool = True


@dataclasses.dataclass
class AceConfig:
    batch_size: int = 2
    use_random_seed: bool = True
    seeds: list[int] | None = None
    audio_format: str = "flac"


@dataclasses.dataclass
class AceResult:
    audios: list[dict[str, Any]]
    status_message: str = ""
    success: bool = True
    error: str | None = None


class FakeAce:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.init_calls: list[dict[str, Any]] = []
        self.lm_calls: list[dict[str, Any]] = []
        self.fail_with: str | None = None
        self.init_ok = True
        self.lm_ok = True
        self.progress_steps = 3
        self.swallow_exceptions = False  # the real generate_music turns exceptions into a failed result


def install_ace(monkeypatch: pytest.MonkeyPatch, cuda: bool = False) -> FakeAce:
    install_torch(monkeypatch, cuda=cuda)
    state = FakeAce()

    class AceStepHandler:
        def initialize_service(self, project_root: str, config_path: str, device: str = "auto", **_: Any) -> tuple[str, bool]:
            state.init_calls.append({"project_root": project_root, "config_path": config_path, "device": device})
            return ("ready" if state.init_ok else "boom: no weights"), state.init_ok

    class LLMHandler:
        def __init__(self) -> None:
            self.llm_initialized = False

        def initialize(self, checkpoint_dir: str, lm_model_path: str, backend: str = "vllm", device: str = "auto", **_: Any) -> tuple[str, bool]:
            state.lm_calls.append({"checkpoint_dir": checkpoint_dir, "lm_model_path": lm_model_path, "backend": backend, "device": device})
            self.llm_initialized = state.lm_ok
            return "lm", state.lm_ok

    def generate_music(dit: Any, llm: Any, params: AceParams, config: AceConfig, save_dir: str | None = None, progress: Any = None) -> AceResult:
        state.calls.append({"params": params, "config": config, "save_dir": save_dir, "llm": llm})
        if progress is not None:
            try:
                for k in range(state.progress_steps):
                    progress((k + 1) / (state.progress_steps + 1), desc=f"step {k}")
            except Exception as exc:
                if not state.swallow_exceptions:
                    raise
                return AceResult([], success=False, error=f"{type(exc).__name__}: {exc}")
        if state.fail_with:
            return AceResult([], success=False, error=state.fail_with)
        rate = 48000
        seconds = params.duration if params.duration and params.duration > 0 else 2.0
        n = int(seconds * rate)
        tensor = FakeTensor(decaying_tone(n, rate, int(params.seed)))
        return AceResult([{"path": "", "tensor": tensor, "sample_rate": rate, "key": "k", "params": {}}])

    put(
        monkeypatch,
        module("acestep"),
        module("acestep.handler", AceStepHandler=AceStepHandler),
        module("acestep.inference", GenerationParams=AceParams, GenerationConfig=AceConfig, generate_music=generate_music),
        module("acestep.llm_inference", LLMHandler=LLMHandler),
    )
    return state


# ---------------------------------------------------------------- fake basic-pitch


class FakeBasicPitch:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.models: list[Any] = []
        self.events: list[Any] = [
            (0.0, 0.5, 60, 0.8, None), (0.5, 1.0, 64, 0.4, [1, 2]), (0.25, 0.75, 67, 1.0, None),
        ]


def install_basic_pitch(monkeypatch: pytest.MonkeyPatch) -> FakeBasicPitch:
    state = FakeBasicPitch()

    class Model:
        def __init__(self, path: Any) -> None:
            self.path = path
            state.models.append(self)

    def predict(audio_path: Any, model_or_model_path: Any = "default", **kw: Any) -> tuple[dict[str, Any], Any, list[Any]]:
        state.calls.append({"audio_path": str(audio_path), "model": model_or_model_path, **kw})
        return {}, None, list(state.events)

    put(
        monkeypatch,
        module("basic_pitch", ICASSP_2022_MODEL_PATH="/fake/nmp"),
        module("basic_pitch.inference", Model=Model, predict=predict),
    )
    return state


# ---------------------------------------------------------------- fake executables


def write_script(path: Path, body: str) -> Path:
    """An executable Python script (shebang = the test interpreter) standing in for a native tool."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!{sys.executable}\n" + textwrap.dedent(body))
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a zombie still answers signal 0; check the state if /proc exists
    status = Path(f"/proc/{pid}/status")
    if status.exists():
        return "State:\tZ" not in status.read_text()
    return True


# ---------------------------------------------------------------- protocol harness


class Harness:
    """Drives ``runtime.serve`` over pipes like the gateway does: send a request, read the replies."""

    def __init__(self, family: runtime.Family) -> None:
        in_r, in_w = os.pipe()
        out_r, out_w = os.pipe()
        self._to_worker = os.fdopen(in_w, "w", encoding="utf-8")
        self._from_worker = os.fdopen(out_r, "r", encoding="utf-8")
        self.code: int | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._next = 0
        stdin, stdout = os.fdopen(in_r, "r", encoding="utf-8"), os.fdopen(out_w, "w", encoding="utf-8")

        def serve() -> None:
            self.code = runtime.serve(family, stdin, stdout)
            stdout.close()  # the reader sees EOF

        def read() -> None:
            for line in self._from_worker:
                self._lines.put(line)
            self._lines.put(None)

        self._server = threading.Thread(target=serve, daemon=True)
        self._reader = threading.Thread(target=read, daemon=True)
        self._server.start()
        self._reader.start()

    def send(self, op: str, **fields: Any) -> int:
        self._next += 1
        self._to_worker.write(json.dumps({"id": self._next, "op": op, **fields}) + "\n")
        self._to_worker.flush()
        return self._next

    def raw(self, text: str) -> None:
        self._to_worker.write(text)
        self._to_worker.flush()

    def recv(self, timeout: float = 20.0) -> dict[str, Any]:
        line = self._lines.get(timeout=timeout)
        if line is None:
            raise EOFError("worker closed its output")
        return json.loads(line)  # type: ignore[no-any-return]

    def call(self, op: str, timeout: float = 20.0, **fields: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Send a request; returns ``(progress events, final reply)``."""
        req = self.send(op, **fields)
        events: list[dict[str, Any]] = []
        while True:
            message = self.recv(timeout)
            if message.get("id") != req:
                raise AssertionError(f"reply for another request: {message}")
            if message.get("event") == "progress":
                events.append(message)
                continue
            return events, message

    def close(self) -> None:
        with contextlib.suppress(OSError, ValueError):
            self._to_worker.close()
        self._server.join(timeout=10)

    def __enter__(self) -> Harness:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def hello_models(family: runtime.Family, options: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """The ``models`` of a ``hello`` reply, by id, through the real protocol loop."""
    with Harness(family) as h:
        _, reply = h.call("hello", options=options or {})
        assert reply["ok"], reply
        return {m["id"]: m for m in reply["result"]["models"]}


def in_memory(requests: list[dict[str, Any]], family: runtime.Family) -> list[dict[str, Any]]:
    """Run only synchronous requests (no ``run``) through ``serve`` with StringIO streams."""
    stdin = io.StringIO("".join(json.dumps(r) + "\n" for r in requests))
    stdout = io.StringIO()
    runtime.serve(family, stdin, stdout)
    return [json.loads(line) for line in stdout.getvalue().splitlines()]
