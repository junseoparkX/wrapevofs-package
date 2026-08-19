import json
import runpy
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
from pathlib import Path

import pytest
import yaml

from wrapevofs import (
    ARTIFACT_SCHEMA_VERSION,
    CONFIG_SCHEMA_VERSION,
    PipelineConfig,
    __version__,
    validate_locking_artifact_directory,
)
from wrapevofs.cli import build_parser


ROOT = Path(__file__).resolve().parents[1]


def test_release_version_is_consistent_across_package_metadata_and_citation():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == __version__
    assert citation["version"] == __version__


def test_bsd_license_is_consistent_and_present():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert pyproject["project"]["license"] == "BSD-3-Clause"
    assert pyproject["project"]["license-files"] == ["LICENSE"]
    assert citation["license"] == "BSD-3-Clause"
    assert license_text.startswith("BSD 3-Clause License")
    assert "Copyright (c) 2026, Junseo Park" in license_text


def test_cli_help_and_version_are_available(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit) as help_exit:
        parser.parse_args(["--help"])
    assert help_exit.value.code == 0
    assert "usage: wrapevofs" in capsys.readouterr().out

    with pytest.raises(SystemExit) as version_exit:
        parser.parse_args(["--version"])
    assert version_exit.value.code == 0
    assert f"wrapevofs {__version__}" in capsys.readouterr().out


def test_configuration_schema_parsing_and_mismatch_rejection():
    config = PipelineConfig.from_dict(
        {
            "config_schema_version": CONFIG_SCHEMA_VERSION,
            "ga": {"fitness_mode": "untruncated_shifted_linear"},
        }
    )
    assert config.to_dict()["config_schema_version"] == CONFIG_SCHEMA_VERSION
    with pytest.raises(ValueError, match="Unsupported configuration schema"):
        PipelineConfig.from_dict({"config_schema_version": "999"})


def test_documented_python_quickstart_executes(capsys):
    namespace = runpy.run_path(str(ROOT / "examples" / "quickstart.py"))
    namespace["main"]()
    assert "RFECV target k:" in capsys.readouterr().out


def test_toy_regret_locking_round_trip_and_schema_validation(tmp_path):
    namespace = runpy.run_path(str(ROOT / "examples" / "toy_regret_locking.py"))
    output = namespace["run_demo"](tmp_path)
    summary = validate_locking_artifact_directory(output)
    assert summary["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert summary["selected_within_declared_tolerance"] is True

    reloaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert reloaded == summary


def test_artifact_missing_field_and_version_mismatch_errors(tmp_path):
    namespace = runpy.run_path(str(ROOT / "examples" / "toy_regret_locking.py"))
    namespace["run_demo"](tmp_path)
    summary_path = tmp_path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.pop("configuration_hash")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required fields"):
        validate_locking_artifact_directory(tmp_path)

    namespace["run_demo"](tmp_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["artifact_schema_version"] = "0"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="Artifact schema version mismatch"):
        validate_locking_artifact_directory(tmp_path)
