"""SVM-L1 first-stage wrapper selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from wrapevofs.config import SVML1WrapperConfig
from wrapevofs._scoring import infer_problem_type, resolve_auto_scoring, unique_classes
from wrapevofs.selectors._result import SelectionResult


def select_svm_l1(
    X: pd.DataFrame,
    y: pd.Series,
    config: SVML1WrapperConfig | None = None,
) -> SelectionResult:
    config = config or SVML1WrapperConfig()
    classes = unique_classes(y)
    problem_type = infer_problem_type(y)
    scoring = resolve_auto_scoring(
        config.scoring,
        y,
        binary="roc_auc",
        multiclass="balanced_accuracy",
    )
    cv = StratifiedKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )
    svm = LinearSVC(
        penalty=config.penalty,
        dual=config.dual,
        class_weight=config.class_weight,
        max_iter=config.max_iter,
        tol=config.tol,
        random_state=config.random_state,
    )
    base_model = (
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", svm),
            ]
        )
        if config.standardize
        else svm
    )
    c_parameter = "model__C" if config.standardize else "C"
    search = GridSearchCV(
        estimator=base_model,
        param_grid={c_parameter: config.c_grid},
        scoring=scoring,
        cv=cv,
        n_jobs=config.n_jobs,
        refit=True,
    )
    search.fit(X, y)
    model = search.best_estimator_
    coef_model = model.named_steps["model"] if config.standardize else model
    raw_coef = np.asarray(coef_model.coef_)
    if raw_coef.ndim == 1 or raw_coef.shape[0] == 1:
        coef = raw_coef.ravel()
        abs_coef = np.abs(coef)
        coef_aggregation = "binary_signed"
    else:
        abs_coef = np.max(np.abs(raw_coef), axis=0)
        coef = abs_coef.copy()
        coef_aggregation = "max_abs_across_classes"

    table = pd.DataFrame(
        {
            "feature": X.columns,
            "coef": coef,
            "abs_coef": abs_coef,
        }
    ).sort_values("abs_coef", ascending=False, kind="mergesort")
    selected = table.loc[
        table["abs_coef"] > config.coefficient_threshold,
        "feature",
    ].tolist()
    if not selected:
        raise RuntimeError("SVM-L1 selected zero features; increase C or lower threshold.")

    table["selected"] = table["feature"].isin(selected)
    return SelectionResult(
        name="svm_l1",
        selected_features=selected,
        feature_table=table.reset_index(drop=True),
        estimator=model,
        metadata={
            "best_params": search.best_params_,
            "best_cv_score": float(search.best_score_),
            "scoring": config.scoring,
            "resolved_scoring": scoring,
            "problem_type": problem_type,
            "n_classes": int(len(classes)),
            "classes": classes.tolist(),
            "coefficient_threshold": config.coefficient_threshold,
            "coef_aggregation": coef_aggregation,
            "standardize": config.standardize,
            "penalty": config.penalty,
            "dual": config.dual,
            "class_weight": config.class_weight,
            "max_iter": config.max_iter,
        },
    )
