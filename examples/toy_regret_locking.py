"""Deterministic software demonstration of fixed-CV regret locking.

This is unrestricted synthetic data for software validation. It is not a
biomedical analysis and provides no empirical validation evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification

from wrapevofs import LockingConfig, lock_representative_run
from wrapevofs.locking import score_candidate_feature_sets


def run_demo(output_dir: str | Path) -> Path:
    X_values, y_values = make_classification(
        n_samples=60,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=2026,
    )
    features = [f"toy_feature_{index}" for index in range(6)]
    X = pd.DataFrame(X_values, columns=features)
    y = pd.Series(y_values, name="toy_target")
    candidate_sets = {
        0: features[:2],
        1: features[:3],
        2: [features[0], features[2], features[3]],
        3: features[1:5],
    }
    candidates = score_candidate_feature_sets(
        X,
        y,
        candidate_sets,
        locking_metric="balanced_accuracy",
        cv_folds=3,
        random_state=2026,
        rf_params={"n_estimators": 10, "max_depth": 3},
        n_jobs=1,
    )
    result = lock_representative_run(
        candidates,
        LockingConfig(
            enabled=True,
            strategy="regret_constrained_medoid",
            tolerance_mode="absolute",
            regret_tolerance=0.05,
            minimum_pool_size=1,
            fallback_rule="strict_eligible_only",
            locking_metric="balanced_accuracy",
            cv_folds=3,
            random_state=2026,
        ),
        full_configuration={"example": "unrestricted_toy_regret_locking"},
        seeds={"locking_cv_seed": 2026},
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.candidate_audit.to_csv(output / "locking_candidate_audit.csv", index=False)
    result.pairwise_jaccard.to_csv(output / "pairwise_jaccard.csv", index=False)
    pd.DataFrame({"feature": result.selected_features}).to_csv(
        output / "selected_features.csv", index=False
    )
    (output / "summary.json").write_text(
        json.dumps(result.metadata, indent=2), encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/toy_regret_locking")
    args = parser.parse_args()
    print(run_demo(args.out))


if __name__ == "__main__":
    main()
