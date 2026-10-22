from __future__ import annotations

import sys
from pathlib import Path

import pytest

from forge.config import MAX_CONFIG_BYTES, load_config, parse_config
from forge.errors import ConfigError


def test_defaults() -> None:
    cfg = parse_config({})
    assert (cfg.listen.host, cfg.listen.port) == ("127.0.0.1", 7700)
    assert cfg.token_env == "FORGE_TOKEN" and not cfg.insecure_no_auth
    assert cfg.retention_days == 30 and cfg.memory_budget_gb == 96
    assert cfg.uploads.max_bytes == 200 * 1024 * 1024 and cfg.uploads.max_seconds == 600
    assert cfg.data_dir == Path("~/.local/share/forge").expanduser()
    assert cfg.gary_shim.enabled is False and cfg.gary_shim.ports == {"foundation": 8015, "sa3": 8006}


def test_family_defaults() -> None:
    cfg = parse_config({"families": {"foundation": {}, "basicpitch": {"enabled": True}, "yin": {"lane": "gpu"}}})
    assert [cfg.family(n).enabled for n in ("testtone", "yin", "analysis", "foundation")] == [True, True, True, False]
    assert cfg.family("foundation").lane == "gpu" and cfg.family("testtone").lane == "gpu"
    assert all(cfg.family(n).lane == "cpu" for n in ("basicpitch", "stems", "render", "analysis"))
    assert cfg.family("yin").lane == "gpu"  # explicit setting wins
    assert cfg.family("foundation").python == sys.executable
    assert cfg.family("foundation").env == {} and cfg.family("foundation").options == {}
    assert cfg.enabled_families() == ["testtone", "yin", "analysis", "basicpitch"]


def test_load_from_file_and_env(tmp_path: Path) -> None:
    file = tmp_path / "forge.yaml"
    file.write_text("listen: {port: 7801}\nretention_days: 7\nfamilies:\n  sa3:\n    enabled: true\n    python: /x/py\n", encoding="utf-8")
    assert load_config(file).listen.port == 7801
    cfg = load_config(None, {"FORGE_CONFIG": str(file)})
    assert cfg.retention_days == 7 and cfg.family("sa3").python == "/x/py"
    assert load_config(None, {}).listen.port == 7700
    assert "sa3" in cfg.enabled_families()


@pytest.mark.parametrize(
    "text",
    ["listen: {port: 0}", "listen: {port: 70000}", "bogus: 1", "retention_days: -1", "memory_budget_gb: .nan",
     "families: {Bad-Name: {}}", "token_env: 'not a name'", "families: {x: {lane: tpu}}", "- a\n- b", "uploads: {max_seconds: 0}",
     "families: {x: {env: {'A=B': c}}}"],
)
def test_invalid_configs_are_rejected(tmp_path: Path, text: str) -> None:
    file = tmp_path / "bad.yaml"
    file.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(file)


def test_unreadable_oversized_and_malformed_files(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.yaml")
    big = tmp_path / "big.yaml"
    big.write_text("# " + "x" * MAX_CONFIG_BYTES, encoding="utf-8")
    with pytest.raises(ConfigError, match="larger"):
        load_config(big)
    broken = tmp_path / "broken.yaml"
    broken.write_text("a: [1, 2", encoding="utf-8")
    with pytest.raises(ConfigError, match="YAML"):
        load_config(broken)
    binary = tmp_path / "bin.yaml"
    binary.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(ConfigError):
        load_config(binary)


def test_error_names_the_field() -> None:
    with pytest.raises(ConfigError, match=r"listen\.port"):
        parse_config({"listen": {"port": 0}})


def test_token_handling() -> None:
    cfg = parse_config({"token_env": "MY_TOKEN"})
    assert cfg.require_token({"MY_TOKEN": "s3cret"}) == "s3cret"
    with pytest.raises(ConfigError, match="MY_TOKEN"):
        cfg.require_token({})
    with pytest.raises(ConfigError):
        cfg.require_token({"MY_TOKEN": ""})
    assert parse_config({"insecure_no_auth": True}).require_token({}) is None
