"""Job types, parameter models (DESIGN 3.4) and job records (DESIGN 3.5)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .errors import ValidationFailed
from .music import KEY_PATTERN

JOB_TYPES = (
    "generate.loop",
    "generate.sample",
    "generate.instrument",
    "generate.sfx",
    "transform.audio",
    "extend.audio",
    "inpaint.audio",
    "layer.add",
    "stems.split",
    "transcribe.melody",
    "analyze.audio",
    "render.session",
)
OUTPUT_KINDS = {
    "generate.loop": "audio",
    "generate.sample": "audio",
    "generate.instrument": "instrument",
    "generate.sfx": "audio",
    "transform.audio": "audio",
    "extend.audio": "audio",
    "inpaint.audio": "audio",
    "layer.add": "audio",
    "stems.split": "audio",
    "transcribe.melody": "midi",
    "analyze.audio": "none",
    "render.session": "audio",
}
# Required input names per job type; the value is the kind of asset each must be.
INPUT_KINDS: dict[str, dict[str, str]] = {
    "transform.audio": {"audio": "audio"},
    "extend.audio": {"audio": "audio"},
    "inpaint.audio": {"audio": "audio"},
    "layer.add": {"audio": "audio"},
    "stems.split": {"audio": "audio"},
    "transcribe.melody": {"audio": "audio"},
    "analyze.audio": {"audio": "audio"},
    "render.session": {"session": "session"},
}
PRIORITIES = ("high", "normal", "low")
PRIORITY_RANK = {name: rank for rank, name in enumerate(PRIORITIES)}
JobState = Literal["queued", "running", "succeeded", "failed", "cancelled"]
FINAL_STATES = ("succeeded", "failed", "cancelled")
ASSET_ID_PATTERN = r"^a_[0-9a-f]{24}$"
MAX_ZONES = 256
MAX_INSTRUMENT_SECONDS = 1200.0
_ASSET_ID = re.compile(ASSET_ID_PATTERN)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class PostParams(_Strict):
    trim_silence: bool = False
    fade_in_ms: float = Field(default=0.0, ge=0, le=1000)
    fade_out_ms: float = Field(default=0.0, ge=0, le=5000)
    normalize: Literal["none", "peak", "lufs"] = "peak"
    target: float = -1.0  # dBFS for "peak", LUFS for "lufs"
    loop_crossfade_ms: float = Field(default=0.0, ge=0, le=500)
    bit_depth: Literal["16", "24", "32f"] = "24"

    @model_validator(mode="after")
    def _target_range(self) -> PostParams:
        lo, hi = (-70.0, 0.0) if self.normalize == "lufs" else (-120.0, 0.0)
        if self.normalize != "none" and not lo <= self.target <= hi:
            raise ValueError(f"target must be between {lo:g} and {hi:g} for normalize={self.normalize!r}")
        return self


class CommonParams(_Strict):
    prompt: str = Field(default="", max_length=2000)
    tags: dict[str, str | list[str]] = Field(default_factory=dict)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    candidates: int = Field(default=1, ge=1, le=8)
    sample_rate: int = Field(default=48000, ge=22050, le=192000)
    channels: Literal[1, 2] = 2
    post: PostParams = Field(default_factory=lambda: PostParams())

    @field_validator("tags")
    @classmethod
    def _tags(cls, value: dict[str, str | list[str]]) -> dict[str, str | list[str]]:
        if len(value) > 64:
            raise ValueError("at most 64 tags")
        for key, item in value.items():
            values = [item] if isinstance(item, str) else item
            if not 1 <= len(key) <= 64 or len(values) > 32 or any(len(v) > 200 for v in values):
                raise ValueError(f"tag {key!r} is too long or has too many values")
        return value


class LoopParams(CommonParams):
    bars: int = Field(default=4, ge=1, le=16)
    bpm: float = Field(ge=40, le=240)
    key: str | None = Field(default=None, max_length=32, pattern=KEY_PATTERN)


class SampleParams(CommonParams):
    note: int = Field(default=60, ge=0, le=127)
    length_s: float = Field(default=2.0, ge=0.05, le=20)


class InstrumentParams(CommonParams):
    note_low: int = Field(default=36, ge=0, le=127)
    note_high: int = Field(default=84, ge=0, le=127)
    step: int = Field(default=3, ge=1, le=12)
    velocity_layers: int = Field(default=1, ge=1, le=3)
    length_s: float = Field(default=3.0, ge=0.05, le=20)
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9 _.\-]*$")

    @model_validator(mode="after")
    def _range(self) -> InstrumentParams:
        if self.note_low > self.note_high:
            raise ValueError("note_low must not be above note_high")
        keys = self.note_high - self.note_low + 1
        zones = -(-keys // self.step) * self.velocity_layers
        if zones > MAX_ZONES:
            raise ValueError(f"{zones} zones requested (at most {MAX_ZONES})")
        if zones * self.length_s * self.candidates > MAX_INSTRUMENT_SECONDS:
            raise ValueError(f"too much audio requested (zones x length_s x candidates > {MAX_INSTRUMENT_SECONDS:g} s)")
        return self


class SfxParams(CommonParams):
    duration_s: float = Field(default=4.0, ge=0.1, le=47)


class TransformParams(CommonParams):
    strength: float = Field(default=0.5, ge=0, le=1)


class ExtendParams(CommonParams):
    seconds: float = Field(default=8.0, ge=1, le=60)


class InpaintParams(CommonParams):
    start_s: float = Field(ge=0, le=3600)
    end_s: float = Field(ge=0, le=3600)

    @model_validator(mode="after")
    def _region(self) -> InpaintParams:
        if self.end_s <= self.start_s:
            raise ValueError("end_s must be greater than start_s")
        return self


class LayerParams(CommonParams):
    description: str = Field(min_length=1, max_length=2000)


class StemsParams(CommonParams):
    stems: list[str] = Field(default_factory=list, max_length=16)

    @field_validator("stems")
    @classmethod
    def _names(cls, value: list[str]) -> list[str]:
        if any(not 1 <= len(s) <= 64 for s in value):
            raise ValueError("stem names must be 1..64 characters")
        return value


class TranscribeParams(CommonParams):
    min_note_ms: float = Field(default=60.0, ge=1, le=5000)
    bpm: float = Field(default=120.0, ge=1, le=1000)


class AnalyzeParams(CommonParams):
    pass


class RenderCommand(_Strict):
    command: str = Field(min_length=1, max_length=128)
    params: dict[str, Any] = Field(default_factory=dict)


class RenderParams(CommonParams):
    commands: list[RenderCommand] = Field(default_factory=list, max_length=1000)
    start: float = Field(default=0.0, ge=0, le=86400)
    end: float | None = Field(default=None, ge=0, le=86400)

    @model_validator(mode="after")
    def _span(self) -> RenderParams:
        if self.end is not None and self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self


PARAM_MODELS: dict[str, type[CommonParams]] = {
    "generate.loop": LoopParams,
    "generate.sample": SampleParams,
    "generate.instrument": InstrumentParams,
    "generate.sfx": SfxParams,
    "transform.audio": TransformParams,
    "extend.audio": ExtendParams,
    "inpaint.audio": InpaintParams,
    "layer.add": LayerParams,
    "stems.split": StemsParams,
    "transcribe.melody": TranscribeParams,
    "analyze.audio": AnalyzeParams,
    "render.session": RenderParams,
}


@dataclass(frozen=True)
class ValidatedJob:
    type: str
    params: CommonParams
    inputs: dict[str, str]

    @property
    def kind(self) -> str:
        return OUTPUT_KINDS[self.type]


_UNION_LABELS = ("str", "list[str]")


def _path(prefix: str, loc: tuple[Any, ...]) -> str:
    """``params.post.fade_in_ms``; pydantic's union-branch labels (``str``, ``list[str]``) are left out."""
    return ".".join([prefix, *(str(p) for p in loc if str(p) not in _UNION_LABELS)])


def _errors(prefix: str, exc: ValidationError) -> list[dict[str, str]]:
    merged: dict[str, list[str]] = {}
    for e in exc.errors():
        merged.setdefault(_path(prefix, e["loc"]), []).append(e["msg"])
    return [{"path": path, "message": "; ".join(dict.fromkeys(msgs))} for path, msgs in merged.items()]


def validate_job(job_type: Any, params: Any = None, inputs: Any = None) -> ValidatedJob:
    """Validate a submission; raises :class:`ValidationFailed` listing every error with its field path."""
    errors: list[dict[str, str]] = []
    if not isinstance(job_type, str) or job_type not in PARAM_MODELS:
        raise ValidationFailed([{"path": "type", "message": f"unknown job type; expected one of {', '.join(JOB_TYPES)}"}])
    parsed: CommonParams | None = None
    raw_params = {} if params is None else params
    if not isinstance(raw_params, dict):
        errors.append({"path": "params", "message": "must be an object"})
    else:
        try:
            parsed = PARAM_MODELS[job_type].model_validate(raw_params)
        except ValidationError as exc:
            errors += _errors("params", exc)
    clean_inputs: dict[str, str] = {}
    raw_inputs = {} if inputs is None else inputs
    if not isinstance(raw_inputs, dict):
        errors.append({"path": "inputs", "message": "must be an object"})
    else:
        wanted = INPUT_KINDS.get(job_type, {})
        for name, value in raw_inputs.items():
            if name not in wanted:
                errors.append({"path": f"inputs.{name}", "message": "unexpected input"})
            elif not isinstance(value, str) or not _ASSET_ID.match(value):
                errors.append({"path": f"inputs.{name}", "message": "must be an asset id (a_ + 24 hex digits)"})
            else:
                clean_inputs[name] = value
        for name in wanted:
            if name not in raw_inputs:
                errors.append({"path": f"inputs.{name}", "message": "required input is missing"})
    if errors or parsed is None:
        raise ValidationFailed(errors)
    return ValidatedJob(job_type, parsed, clean_inputs)


class JobResult(BaseModel):
    asset_ids: list[str] = Field(default_factory=list)
    analysis: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)


class JobRecord(BaseModel):
    """The public job record returned by ``GET /v1/jobs/{id}``."""

    job_id: str
    type: str
    model: str | None = None
    family: str | None = None
    lane: str | None = None
    state: JobState = "queued"
    priority: Literal["high", "normal", "low"] = "normal"
    params: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, str] = Field(default_factory=dict)
    progress: float = 0.0
    message: str = ""
    queue_position: int | None = None
    eta_s: float | None = None
    error: str | None = None
    result: JobResult | None = None
    idempotency_key: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
