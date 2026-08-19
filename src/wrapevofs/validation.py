"""Validation helpers for versioned machine-readable release artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from wrapevofs._version import ARTIFACT_SCHEMA_VERSION, __version__


_LOCKING_AUDIT_REQUIRED = {
    "run_id",
    "canonical_features",
    "canonical_mask",
    "stable_mask_hash",
    "locking_score",
    "absolute_regret",
    "eligible",
    "mean_jaccard",
    "selected",
    "software_version",
    "configuration_hash",
}
_PAIRWISE_REQUIRED = {
    "run_i",
    "run_j",
    "stable_mask_hash_i",
    "stable_mask_hash_j",
    "jaccard",
}
_SUMMARY_REQUIRED = {
    "artifact_schema_version",
    "software_version",
    "strategy",
    "selected_feature_set",
    "selected_stable_mask_hash",
    "selected_within_declared_tolerance",
    "configuration_hash",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Artifact JSON is missing or invalid: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Artifact JSON must contain an object: {path}")
    return value


def _require_fields(actual: set[str], required: set[str], label: str) -> None:
    missing = sorted(required - actual)
    if missing:
        raise ValueError(f"{label} is missing required fields: {', '.join(missing)}")


def validate_locking_artifact_directory(path: str | Path) -> dict[str, Any]:
    """Validate a locking artifact directory and return its summary."""

    root = Path(path)
    summary = _read_json(root / "summary.json")
    _require_fields(set(summary), _SUMMARY_REQUIRED, "summary.json")
    if str(summary["artifact_schema_version"]) != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            "Artifact schema version mismatch: expected "
            f"{ARTIFACT_SCHEMA_VERSION}, found {summary['artifact_schema_version']}."
        )
    if str(summary["software_version"]) != __version__:
        raise ValueError(
            f"Software version mismatch: expected {__version__}, "
            f"found {summary['software_version']}."
        )
    try:
        audit = pd.read_csv(root / "locking_candidate_audit.csv")
        pairwise = pd.read_csv(root / "pairwise_jaccard.csv")
        selected = pd.read_csv(root / "selected_features.csv")
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Locking CSV artifact is missing or invalid in: {root}") from exc
    _require_fields(set(audit.columns), _LOCKING_AUDIT_REQUIRED, "locking audit")
    _require_fields(set(pairwise.columns), _PAIRWISE_REQUIRED, "pairwise Jaccard table")
    _require_fields(set(selected.columns), {"feature"}, "selected feature table")
    if int(audit["selected"].astype(bool).sum()) < 1:
        raise ValueError("Locking audit does not identify a selected candidate record.")
    if selected["feature"].astype(str).tolist() != [
        str(item) for item in summary["selected_feature_set"]
    ]:
        raise ValueError("Selected feature CSV and summary.json disagree.")
    if summary["strategy"] == "regret_constrained_medoid":
        selected_rows = audit.loc[audit["selected"].astype(bool)]
        if not selected_rows["eligible"].astype(bool).all():
            raise ValueError("A selected regret-constrained candidate is ineligible.")
        if summary["selected_within_declared_tolerance"] is not True:
            raise ValueError("Summary does not confirm strict tolerance feasibility.")
    return summary
