import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning

from wrapevofs.config import SVML1WrapperConfig
from wrapevofs.selectors.svm_l1_wrapper import select_svm_l1


def test_svm_l1_standardizes_internally_without_convergence_warning():
    X, y = make_classification(
        n_samples=80,
        n_features=8,
        n_informative=4,
        random_state=42,
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    X["large_scale"] = X["f0"] * 100000

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = select_svm_l1(
            X,
            pd.Series(y),
            SVML1WrapperConfig(
                standardize=True,
                c_grid=[0.01, 0.1, 1.0],
                cv_folds=2,
                n_jobs=1,
                max_iter=100000,
            ),
        )

    convergence_warnings = [
        item for item in caught if issubclass(item.category, ConvergenceWarning)
    ]
    assert not convergence_warnings
    assert result.n_selected >= 1
    assert result.metadata["standardize"] is True
    assert result.metadata["problem_type"] == "binary"
    assert result.metadata["resolved_scoring"] == "roc_auc"


def test_svm_l1_handles_multiclass_coefficients():
    X, y = make_classification(
        n_samples=90,
        n_features=9,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        random_state=7,
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

    result = select_svm_l1(
        X,
        pd.Series(y),
        SVML1WrapperConfig(
            standardize=True,
            c_grid=[0.01, 0.1, 1.0],
            cv_folds=3,
            n_jobs=1,
            max_iter=100000,
        ),
    )

    assert result.n_selected >= 1
    assert result.feature_table["abs_coef"].notna().all()
    assert result.metadata["problem_type"] == "multiclass"
    assert result.metadata["n_classes"] == 3
    assert result.metadata["resolved_scoring"] == "balanced_accuracy"
    assert result.metadata["coef_aggregation"] == "max_abs_across_classes"
