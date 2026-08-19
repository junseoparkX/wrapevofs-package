import numpy as np
import pandas as pd
import json
from sklearn.datasets import make_classification

from wrapevofs import WrapEvoPipeline, PipelineConfig
from wrapevofs.artifacts import save_pipeline_result
from wrapevofs import ARTIFACT_SCHEMA_VERSION, __version__


def test_save_pipeline_result_exports_top_ga_feature_sets(tmp_path):
    X, y = make_classification(
        n_samples=64,
        n_features=10,
        n_informative=5,
        n_redundant=1,
        class_sep=1.5,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y

    config = PipelineConfig.from_dict(
        {
            "split": {"ratio": "7:3", "random_state": 42},
            "preprocessing": {"impute_strategy": "zero"},
            "first_stage": {
                "enabled_methods": ["svm_l1"],
                "svm_l1": {
                    "c_grid": [0.1, 1.0],
                    "cv_folds": 2,
                    "n_jobs": 1,
                    "max_iter": 100000,
                },
            },
            "rfecv": {
                "method_max_features_to_consider": {"svm_l1": 5},
                "cv_folds": 2,
                "step": 1,
                "n_jobs": 1,
                "rf_params": {"n_estimators": 5, "max_depth": 3},
            },
            "ga": {
                "backend": "cpu",
                "population_size": 6,
                "n_generations": 1,
                "n_runs": 5,
                "top_k": 5,
                "cv_folds": 2,
                "n_jobs": 1,
                "rf_params": {"n_estimators": 3, "max_depth": 2},
            },
        }
    )

    result = WrapEvoPipeline(config).run_full(df=df, target_column="target")
    save_pipeline_result(result, tmp_path)

    final_feature_sets = np.load(
        tmp_path / "final_feature_sets.npy",
        allow_pickle=True,
    ).item()
    best_feature_sets = np.load(
        tmp_path / "best_feature_sets.npy",
        allow_pickle=True,
    ).item()
    branch_top_sets = np.load(
        tmp_path / "ga" / "svm_l1" / "top_feature_sets.npy",
        allow_pickle=True,
    )
    top_table = pd.read_csv(tmp_path / "ga" / "svm_l1" / "top_solutions.csv")
    history = pd.read_csv(tmp_path / "ga" / "svm_l1" / "history.csv")
    software_metadata = json.loads(
        (tmp_path / "software_metadata.json").read_text(encoding="utf-8")
    )

    assert "svm_l1" in final_feature_sets
    assert len(final_feature_sets["svm_l1"]) == 5
    assert branch_top_sets.shape[0] == 5
    assert best_feature_sets["svm_l1"] == final_feature_sets["svm_l1"][0]
    assert {
        "score",
        "base_score",
        "raw_objective",
        "legacy_truncated_fitness",
        "target_deviation",
        "penalty_amount",
        "stable_mask_hash",
    }.issubset(top_table.columns)
    assert {
        "fraction_legacy_zero",
        "number_unique_raw_objectives",
        "uniform_sampling_fallback",
        "population_unique_masks",
    }.issubset(history.columns)
    assert software_metadata == {
        "software_name": "wrapevofs",
        "software_version": __version__,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
    }
