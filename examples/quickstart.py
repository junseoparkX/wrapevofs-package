"""Minimal package quickstart on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from wrapevofs import WrapEvoPipeline, PipelineConfig


def main() -> None:
    X, y = make_classification(
        n_samples=80,
        n_features=12,
        n_informative=4,
        n_redundant=2,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"feature_{idx}" for idx in range(X.shape[1])])
    df["target"] = y
    df.loc[0:5, "feature_0"] = np.nan
    df["feature_corr_copy"] = df["feature_1"]

    config = PipelineConfig.from_dict(
        {
            "split": {"ratio": "7:3"},
            "preprocessing": {"impute_strategy": "zero"},
            "first_stage": {
                "enabled_methods": ["svm_l1"],
                "svm_l1": {
                    "c_grid": [0.01, 0.1],
                    "cv_folds": 2,
                    "n_jobs": 1,
                    "max_iter": 5000,
                },
            },
            "rfecv": {
                "method_max_features_to_consider": {"svm_l1": 5},
                "cv_folds": 2,
                "n_jobs": 1,
                "rf_params": {"n_estimators": 5, "max_depth": 3},
            },
        }
    )

    result = WrapEvoPipeline(config).run_until_rfecv(df, target_column="target")
    print("Preprocessed shape:", result.prepared.X_train.shape)
    print("SVM-L1 selected:", result.first_stage["svm_l1"].n_selected)
    print("RFECV target k:", result.rfecv_targets["svm_l1"].target_k)


if __name__ == "__main__":
    main()
