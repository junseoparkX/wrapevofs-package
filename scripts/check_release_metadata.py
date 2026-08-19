"""Fail-fast checks for release metadata; performs no network or data access."""

from __future__ import annotations

from pathlib import Path

import yaml

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from wrapevofs import ARTIFACT_SCHEMA_VERSION, CONFIG_SCHEMA_VERSION, __version__


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert project["project"]["version"] == __version__
    assert citation["version"] == __version__
    assert ARTIFACT_SCHEMA_VERSION == "2.1"
    assert CONFIG_SCHEMA_VERSION == "1.0"
    for config_path in sorted((ROOT / "configs").glob("*.yaml")):
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert isinstance(loaded, dict), config_path
    assert project["project"]["license"] == "BSD-3-Clause"
    assert project["project"]["license-files"] == ["LICENSE"]
    assert citation["license"] == "BSD-3-Clause"
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert license_text.startswith("BSD 3-Clause License")
    assert "Copyright (c) 2026, Junseo Park" in license_text


if __name__ == "__main__":
    main()
