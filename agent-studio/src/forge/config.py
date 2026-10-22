"""``forge.yaml``: the Forge configuration (DESIGN 3.1). Every field is optional."""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .errors import ConfigError

MAX_CONFIG_BYTES = 1024 * 1024
DEFAULT_ENABLED = ("testtone", "yin", "analysis")
CPU_FAMILIES = ("yin", "basicpitch", "stems", "render", "analysis")
_FAMILY_NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")

Lane = Literal["gpu", "cpu"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ListenConfig(_Strict):
    host: str = Field(default="127.0.0.1", min_length=1, max_length=255)
    port: int = Field(default=7700, ge=1, le=65535)


class UploadsConfig(_Strict):
    max_bytes: int = Field(default=200 * 1024 * 1024, ge=1, le=8 * 1024**3)
    max_seconds: float = Field(default=600.0, gt=0, le=3600)


class FamilyConfig(_Strict):
    python: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    enabled: bool | None = None
    lane: Lane | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("env")
    @classmethod
    def _env_names(cls, value: dict[str, str]) -> dict[str, str]:
        for name, item in value.items():
            if not _ENV_NAME.match(name) or "\x00" in item:
                raise ValueError(f"invalid environment entry {name!r}")
        return value


class GaryShimConfig(_Strict):
    enabled: bool = False
    ports: dict[str, int] = Field(default_factory=lambda: {"foundation": 8015, "sa3": 8006})

    @field_validator("ports")
    @classmethod
    def _ports(cls, value: dict[str, int]) -> dict[str, int]:
        if any(not 1 <= p <= 65535 for p in value.values()):
            raise ValueError("ports must be in 1..65535")
        return value


class ForgeConfig(_Strict):
    listen: ListenConfig = Field(default_factory=lambda: ListenConfig())
    token_env: str = "FORGE_TOKEN"
    insecure_no_auth: bool = False
    data_dir: Path = Field(default_factory=lambda: Path("~/.local/share/forge").expanduser())
    retention_days: int = Field(default=30, ge=0, le=36500)
    uploads: UploadsConfig = Field(default_factory=lambda: UploadsConfig())
    memory_budget_gb: float = Field(default=96.0, gt=0, le=65536)
    families: dict[str, FamilyConfig] = Field(default_factory=dict)
    gary_shim: GaryShimConfig = Field(default_factory=GaryShimConfig)

    @field_validator("token_env")
    @classmethod
    def _token_env(cls, value: str) -> str:
        if not _ENV_NAME.match(value):
            raise ValueError("token_env must be an environment variable name")
        return value

    @field_validator("data_dir")
    @classmethod
    def _data_dir(cls, value: Path) -> Path:
        return value.expanduser()

    @field_validator("families")
    @classmethod
    def _family_names(cls, value: dict[str, FamilyConfig]) -> dict[str, FamilyConfig]:
        for name in value:
            if not _FAMILY_NAME.match(name):
                raise ValueError(f"invalid family name {name!r}")
        return value

    def family(self, name: str) -> FamilyConfig:
        """The family's settings with every default filled in."""
        given = self.families.get(name, FamilyConfig())
        return FamilyConfig(
            python=given.python or sys.executable,
            env=dict(given.env),
            enabled=given.enabled if given.enabled is not None else name in DEFAULT_ENABLED,
            lane=given.lane or ("cpu" if name in CPU_FAMILIES else "gpu"),
            options=dict(given.options),
        )

    def enabled_families(self) -> list[str]:
        """Names of enabled families: the built-in defaults first, then the configured ones."""
        names = list(DEFAULT_ENABLED) + [n for n in self.families if n not in DEFAULT_ENABLED]
        return [n for n in names if self.family(n).enabled]

    def token(self, env: Mapping[str, str] | None = None) -> str:
        return (env if env is not None else os.environ).get(self.token_env, "")

    def require_token(self, env: Mapping[str, str] | None = None) -> str | None:
        """The bearer token, or ``None`` when ``insecure_no_auth`` is set. Fails if it is empty."""
        if self.insecure_no_auth:
            return None
        token = self.token(env)
        if not token:
            raise ConfigError(f"environment variable {self.token_env} is empty: set a bearer token")
        return token


def _format(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in exc.errors())


def parse_config(data: Any) -> ForgeConfig:
    """Validate an already-parsed configuration mapping."""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError("the configuration must be a mapping")
    try:
        return ForgeConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration: {_format(exc)}") from exc


def load_config(path: str | os.PathLike[str] | None = None, env: Mapping[str, str] | None = None) -> ForgeConfig:
    """Load ``path``, else ``$FORGE_CONFIG``, else the defaults."""
    environ = env if env is not None else os.environ
    chosen = path if path is not None else (environ.get("FORGE_CONFIG") or None)
    if chosen is None:
        return parse_config({})
    file = Path(chosen).expanduser()
    try:
        if file.stat().st_size > MAX_CONFIG_BYTES:
            raise ConfigError(f"configuration file larger than {MAX_CONFIG_BYTES} bytes")
        text = file.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read configuration {file}: {exc.strerror or exc}") from exc
    except UnicodeDecodeError as exc:
        raise ConfigError(f"configuration {file} is not UTF-8") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"configuration {file} is not valid YAML: {exc}") from exc
    return parse_config(data)
