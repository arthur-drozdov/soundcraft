"""Static checks of the deploy/ files. No Docker, GPU or network needed."""
from __future__ import annotations

import re
import stat
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / "deploy"
FAMILIES = ["foundation", "sa3", "acestep", "basicpitch", "stems", "render", "testtone", "yin"]
TOP_KEYS = ["listen", "token_env", "data_dir", "retention_days", "uploads", "memory_budget_gb", "families", "gary_shim"]


def _load(name: str) -> dict[str, Any]:
    data = yaml.safe_load((DEPLOY / name).read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_design_families_are_in_the_checked_list() -> None:
    design = (ROOT / "DESIGN.md").read_text(encoding="utf-8")
    section = design.split("### 3.3", 1)[1].split("### 3.4", 1)[0]
    named = set(re.findall(r"^\| `(\w+)` \|", section, flags=re.M))
    assert named, "no families found in DESIGN.md 3.3"
    assert named <= set(FAMILIES)


@pytest.mark.parametrize("name", ["forge.example.yaml", "forge.container.yaml"])
def test_config_has_all_families(name: str) -> None:
    cfg = _load(name)
    for fam in FAMILIES:
        assert fam in cfg["families"], fam
    for key in TOP_KEYS:
        assert key in cfg, key


def test_example_has_every_documented_key() -> None:
    cfg = _load("forge.example.yaml")
    assert set(cfg["listen"]) == {"host", "port"}
    assert set(cfg["uploads"]) == {"max_bytes", "max_seconds"}
    assert set(cfg["gary_shim"]) == {"enabled", "ports"}
    assert cfg["listen"]["host"] != "0.0.0.0"
    assert cfg["families"]["yin"]["lane"] == "cpu"
    assert cfg["families"]["foundation"]["lane"] == "gpu"
    for fam in ("foundation", "sa3", "acestep", "basicpitch"):
        assert cfg["families"][fam]["python"].endswith(f"/venvs/{fam}/bin/python")
    assert cfg["families"]["stems"]["options"]["binary"]
    assert cfg["families"]["render"]["options"]["binary"]


@pytest.mark.parametrize("name", ["forge.example.yaml", "forge.container.yaml"])
def test_config_loads_with_forge_loader(name: str) -> None:
    if not (ROOT / "src" / "forge" / "config.py").exists():
        pytest.skip("forge.config not present")
    config = pytest.importorskip("forge.config")
    loader = next(
        (getattr(config, n) for n in ("load_config", "load", "load_forge_config") if hasattr(config, n)),
        None,
    )
    if loader is None:
        pytest.skip("forge.config has no loader function")
    assert loader(DEPLOY / name) is not None


def test_compose_is_valid_and_reserves_gpu() -> None:
    compose = _load("compose.yaml")
    forge = compose["services"]["forge"]
    devices = forge["deploy"]["resources"]["reservations"]["devices"]
    assert any(d.get("driver") == "nvidia" and "gpu" in d.get("capabilities", []) for d in devices)
    assert "FORGE_BIND_IP" in "".join(forge["ports"])
    assert not any(str(p).startswith("0.0.0.0") for p in forge["ports"])
    assert any("/data" in str(v) for v in forge["volumes"])
    assert forge["env_file"]
    dockerfile = ROOT.parent / forge["build"]["dockerfile"]
    assert dockerfile.is_file()


@pytest.mark.parametrize("family", ["foundation", "sa3", "acestep", "basicpitch", "stems", "render"])
def test_family_dockerfile_entrypoint(family: str) -> None:
    text = (DEPLOY / "docker" / f"{family}.Dockerfile").read_text(encoding="utf-8")
    assert f"forge.workers.{family}" in text
    assert re.search(r"^FROM ", text, flags=re.M)


def test_other_dockerfiles_exist() -> None:
    for name in ("gateway", "forge"):
        text = (DEPLOY / "docker" / f"{name}.Dockerfile").read_text(encoding="utf-8")
        assert "forge" in text
    assert "--system-site-packages" in (DEPLOY / "scripts" / "setup-venvs.sh").read_text(encoding="utf-8")


def test_shell_scripts_set_strict_mode() -> None:
    scripts = sorted((DEPLOY / "scripts").glob("*.sh"))
    assert scripts
    for script in scripts:
        lines = script.read_text(encoding="utf-8").splitlines()
        assert lines[0].startswith("#!")
        assert "set -euo pipefail" in lines[1:3], script.name
        assert script.stat().st_mode & stat.S_IXUSR, script.name


def test_systemd_unit() -> None:
    text = (DEPLOY / "systemd" / "forge.service").read_text(encoding="utf-8")
    assert "ExecStart=/opt/forge/venvs/gateway/bin/forge serve" in text
    assert "EnvironmentFile=" in text


def test_readme_covers_required_topics() -> None:
    text = (DEPLOY / "README.md").read_text(encoding="utf-8").lower()
    for topic in ("tailscale", "forge check", "licence", "blackwell", "memory budget", "upgrading", "troubleshooting"):
        assert topic in text, topic
    assert "todo" not in text
