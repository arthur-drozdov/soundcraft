"""Strict, predictable YAML for profiles and patches.

PyYAML's default (YAML 1.1) turns ``On``/``Off``/``yes`` into booleans, ``4:1`` into the integer
241 and ``2026-10-22`` into a date, and expands aliases without limit. None of that is wanted in
files that hold parameter texts such as "4:1" or choice lists such as ``[Off, On]``. This module
reads a YAML 1.2-core-like subset: only ``true``/``false`` are booleans, no sexagesimal numbers,
no timestamps, and no aliases (so no alias bombs). Input size is capped.
"""

from __future__ import annotations

import re
from typing import Any

import yaml

MAX_YAML_BYTES = 2 * 1024 * 1024


class YamlError(ValueError):
    """The text is not acceptable YAML (syntax, size, aliases, nesting)."""


class _Loader(yaml.SafeLoader):
    yaml_implicit_resolvers: dict[str, list[tuple[str, re.Pattern[str]]]] = {}

    def compose_node(self, parent: Any, index: Any) -> Any:
        if self.check_event(yaml.AliasEvent):
            raise YamlError("YAML aliases are not allowed")
        return super().compose_node(parent, index)


_CORE_RESOLVERS: tuple[tuple[str, str, str], ...] = (
    ("tag:yaml.org,2002:bool", r"^(?:true|True|TRUE|false|False|FALSE)$", "tTfF"),
    ("tag:yaml.org,2002:null", r"^(?:~|null|Null|NULL|)$", "~nN"),
    ("tag:yaml.org,2002:int", r"^(?:[-+]?[0-9]+|0x[0-9a-fA-F]+)$", "-+0123456789"),
    (
        "tag:yaml.org,2002:float",
        r"^(?:[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$",
        "-+0123456789.",
    ),
)
for _tag, _pattern, _first in _CORE_RESOLVERS:
    _Loader.add_implicit_resolver(_tag, re.compile(_pattern), list(_first))
_Loader.add_implicit_resolver("tag:yaml.org,2002:null", re.compile(r"^$"), [""])


class _Dumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        return super().increase_indent(flow, False)  # indent list items under their key


_CORE_PATTERNS = [re.compile(pattern) for _, pattern, _ in _CORE_RESOLVERS]


def _represent_str(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    # PyYAML already quotes strings that look like YAML 1.1 scalars; also quote the ones the
    # loader above would read as booleans, null or numbers (e.g. "1e3").
    style = "|" if "\n" in data else ("'" if any(p.match(data) for p in _CORE_PATTERNS) else None)
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_Dumper.add_representer(str, _represent_str)


def loads(text: str) -> Any:
    """Parse YAML text with the restrictions described in the module docstring."""
    if len(text.encode("utf-8", errors="replace")) > MAX_YAML_BYTES:
        raise YamlError(f"YAML larger than {MAX_YAML_BYTES} bytes")
    try:
        return yaml.load(text, Loader=_Loader)  # noqa: S506 - _Loader derives from SafeLoader
    except YamlError:
        raise
    except yaml.YAMLError as exc:
        raise YamlError(str(exc)) from exc
    except RecursionError as exc:
        raise YamlError("YAML nested too deeply") from exc


def dumps(data: Any) -> str:
    """Serialise keeping the mapping order given (callers order keys), UTF-8 friendly."""
    return yaml.dump(
        data,
        Dumper=_Dumper,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=100,
    )


def read_file(path: Any) -> Any:
    """Read and parse a YAML file, enforcing the size cap before reading it all."""
    from pathlib import Path

    p = Path(path)
    try:
        size = p.stat().st_size
        if size > MAX_YAML_BYTES:
            raise YamlError(f"{p}: file larger than {MAX_YAML_BYTES} bytes")
        text = p.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise YamlError(f"{p}: {exc}") from exc
    return loads(text)
