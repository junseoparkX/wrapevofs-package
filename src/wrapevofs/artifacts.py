"""Artifact writing helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wrapevofs.pipeline import PipelineResult, PreparedData
from wrapevofs._version import ARTIFACT_SCHEMA_VERSION, __version__


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def save_prepared_data(prepared: PreparedData, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prepared.X_train.to_csv(output / "X_train.csv", index=False)
    prepared.X_test.to_csv(output / "X_test.csv", index=False)
    prepared.y_train.to_frame("target").to_csv(output / "y_train.csv", index=False)
    prepared.y_test.to_frame("target").to_csv(output / "y_test.csv", index=False)
    with (output / "preprocessing_report.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(prepared.preprocessor.report), handle, indent=2)


def save_pipeline_result(result: PipelineResult, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "software_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "software_name": "wrapevofs",
                "software_version": __version__,
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            },
            handle,
            indent=2,
        )
    save_prepared_data(result.prepared, output / "preprocessed")
    for name, selection in result.first_stage.items():
        method_dir = output / "first_stage" / name
        method_dir.mkdir(parents=True, exist_ok=True)
        selection.feature_table.to_csv(method_dir / "feature_table.csv", index=False)
        pd.DataFrame({"feature": selection.selected_features}).to_csv(
            method_dir / "selected_features.csv",
            index=False,
        )
        np.save(
            method_dir / "selected_features.npy",
            np.asarray(selection.selected_features, dtype=str),
        )
        with (method_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(selection.metadata, handle, indent=2, default=_json_default)

    for name, target in result.rfecv_targets.items():
        method_dir = output / "rfecv" / name
        method_dir.mkdir(parents=True, exist_ok=True)
        target.score_table.to_csv(method_dir / "score_table.csv", index=False)
        summary = {
            "target_k": target.target_k,
            "target_score": target.target_score,
            "global_best_k": target.global_best_k,
            "global_best_score": target.global_best_score,
            "selected_features_at_target": target.selected_features_at_target,
            "metadata": target.metadata,
        }
        np.save(
            method_dir / "target_features.npy",
            np.asarray(target.selected_features_at_target, dtype=str),
        )
        np.save(
            method_dir / "global_best_features.npy",
            np.asarray(target.selected_features_at_global_best, dtype=str),
        )
        with (method_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, default=_json_default)

    final_feature_sets: dict[str, list[Any]] = {}
    best_feature_sets: dict[str, list[str]] = {}
    for name, ga_result in result.ga_results.items():
        method_dir = output / "ga" / name
        method_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for solution in ga_result.top_solutions:
            rows.append(
                {
                    "rank": solution.rank,
                    "run_id": solution.run_id,
                    "score": solution.score,
                    "base_score": solution.base_score,
                    "n_features": solution.n_features,
                    "selected_features": "|".join(solution.selected_features),
                    "raw_objective": solution.raw_objective,
                    "legacy_truncated_fitness": solution.legacy_truncated_fitness,
                    "target_deviation": solution.target_deviation,
                    "penalty_amount": solution.penalty_amount,
                    "stable_mask_hash": solution.stable_mask_hash,
                }
            )
        pd.DataFrame(rows).to_csv(method_dir / "top_solutions.csv", index=False)
        if not ga_result.history.empty:
            ga_result.history.to_csv(method_dir / "history.csv", index=False)
        top_feature_sets = np.asarray(
            [np.asarray(item.selected_features, dtype=str) for item in ga_result.top_solutions],
            dtype=object,
        )
        top_masks = np.asarray([item.mask for item in ga_result.top_solutions], dtype=bool)
        np.save(method_dir / "top_feature_sets.npy", top_feature_sets, allow_pickle=True)
        np.save(method_dir / "top_masks.npy", top_masks)
        np.save(
            method_dir / "scores.npy",
            np.asarray([item.score for item in ga_result.top_solutions], dtype=float),
        )
        if ga_result.top_solutions:
            np.save(
                method_dir / "best_feature_set.npy",
                np.asarray(ga_result.best_solution.selected_features, dtype=str),
            )
            np.save(method_dir / "best_mask.npy", ga_result.best_solution.mask)
            best_feature_sets[name] = ga_result.best_solution.selected_features
            final_feature_sets[name] = [
                solution.selected_features for solution in ga_result.top_solutions
            ]
        summary = {
            "name": ga_result.name,
            "target_k": ga_result.target_k,
            "best_score": ga_result.best_solution.score if ga_result.top_solutions else None,
            "best_n_features": (
                ga_result.best_solution.n_features if ga_result.top_solutions else None
            ),
            "metadata": ga_result.metadata,
            "warnings": ga_result.warnings,
        }
        with (method_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, default=_json_default)

    locked_feature_sets: dict[str, list[str]] = {}
    for name, locking_result in result.locking_results.items():
        method_dir = output / "locking" / name
        method_dir.mkdir(parents=True, exist_ok=True)
        locking_result.candidate_audit.to_csv(
            method_dir / "locking_candidate_audit.csv",
            index=False,
        )
        locking_result.pairwise_jaccard.to_csv(
            method_dir / "pairwise_jaccard.csv",
            index=False,
        )
        pd.DataFrame({"feature": locking_result.selected_features}).to_csv(
            method_dir / "selected_features.csv",
            index=False,
        )
        np.save(
            method_dir / "selected_features.npy",
            np.asarray(locking_result.selected_features, dtype=str),
        )
        with (method_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(locking_result.metadata, handle, indent=2, default=_json_default)
        locked_feature_sets[name] = list(locking_result.selected_features)

    if locked_feature_sets:
        final_feature_sets = locked_feature_sets
        np.save(output / "locked_feature_sets.npy", locked_feature_sets, allow_pickle=True)
    elif not final_feature_sets:
        if result.rfecv_targets:
            final_feature_sets = {
                name: target.selected_features_at_target
                for name, target in result.rfecv_targets.items()
            }
        else:
            final_feature_sets = {
                name: selection.selected_features
                for name, selection in result.first_stage.items()
            }
    if best_feature_sets:
        np.save(output / "best_feature_sets.npy", best_feature_sets, allow_pickle=True)
    np.save(output / "final_feature_sets.npy", final_feature_sets, allow_pickle=True)

    exported_warnings = list(result.warnings)
    for name, ga_result in result.ga_results.items():
        exported_warnings.extend(f"{name}: {message}" for message in ga_result.warnings)
    with (output / "warnings.json").open("w", encoding="utf-8") as handle:
        json.dump(exported_warnings, handle, indent=2)
