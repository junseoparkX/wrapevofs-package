"""RFECV-based target feature-count selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import StratifiedKFold

from wrapevofs.config import RFECVConfig
from wrapevofs._scoring import infer_problem_type, resolve_auto_scoring, unique_classes


@dataclass
class RFECVTargetResult:
    target_k: int
    target_score: float | None
    global_best_k: int
    global_best_score: float | None
    score_table: pd.DataFrame
    selected_features_at_target: list[str]
    selected_features_at_global_best: list[str]
    selector: Any
    target_selector: Any
    metadata: dict[str, Any]


def _make_estimator(config: RFECVConfig) -> Any:
    if config.estimator != "random_forest":
        raise ValueError("Only estimator='random_forest' is implemented for RFECV.")
    params = dict(config.rf_params)
    params.update(
        {
            "random_state": config.random_state,
            "n_jobs": config.n_jobs,
        }
    )
    return RandomForestClassifier(**params)


def _score_table(selector: RFECV, n_input_features: int) -> pd.DataFrame:
    cv_results = selector.cv_results_
    mean_scores = np.asarray(cv_results["mean_test_score"], dtype=float)
    std_scores = np.asarray(
        cv_results.get("std_test_score", np.full_like(mean_scores, np.nan)),
        dtype=float,
    )
    if "n_features" in cv_results:
        n_features = np.asarray(cv_results["n_features"], dtype=int)
    else:
        n_features = np.linspace(
            selector.min_features_to_select,
            n_input_features,
            num=len(mean_scores),
            dtype=int,
        )
    return pd.DataFrame(
        {
            "n_features": n_features,
            "mean_test_score": mean_scores,
            "std_test_score": std_scores,
        }
    ).sort_values("n_features", kind="mergesort").reset_index(drop=True)


def _pick_target_from_table(table: pd.DataFrame, config: RFECVConfig) -> tuple[int, float | None, str]:
    candidates = table
    rule = "global_best"
    if config.max_features_to_consider is not None:
        candidates = table[table["n_features"] <= config.max_features_to_consider]
        rule = f"best_under_or_equal_{config.max_features_to_consider}"
        if candidates.empty:
            raise ValueError("No RFECV candidate satisfies max_features_to_consider.")

    ranked = candidates.sort_values(
        ["mean_test_score", "n_features"],
        ascending=[False, True],
        kind="mergesort",
    )
    row = ranked.iloc[0]
    return int(row["n_features"]), float(row["mean_test_score"]), rule


def find_rfecv_target(
    X: pd.DataFrame,
    y: pd.Series,
    config: RFECVConfig | None = None,
) -> RFECVTargetResult:
    """Fit RFECV and choose a target k.

    The returned target_k can be used by a later GA step as the desired subset
    size. RFECV is used here as a target-size diagnostic, not as the final
    feature selector.
    """

    config = config or RFECVConfig()
    classes = unique_classes(y)
    problem_type = infer_problem_type(y)
    scoring = resolve_auto_scoring(
        config.scoring,
        y,
        binary="roc_auc",
        multiclass="roc_auc_ovr",
    )
    cv = StratifiedKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )
    selector = RFECV(
        estimator=_make_estimator(config),
        step=config.step,
        min_features_to_select=config.min_features_to_select,
        cv=cv,
        scoring=scoring,
        n_jobs=config.n_jobs,
    )
    selector.fit(X, y)
    table = _score_table(selector, X.shape[1])
    target_k, target_score, rule = _pick_target_from_table(table, config)

    target_selector = RFE(
        estimator=_make_estimator(config),
        n_features_to_select=target_k,
        step=config.step,
    )
    target_selector.fit(X, y)
    target_features = X.columns[np.asarray(target_selector.support_, dtype=bool)].tolist()

    global_match = table.loc[table["n_features"] == int(selector.n_features_)]
    global_score = (
        float(global_match["mean_test_score"].iloc[0])
        if not global_match.empty
        else None
    )
    global_features = X.columns[np.asarray(selector.support_, dtype=bool)].tolist()

    return RFECVTargetResult(
        target_k=target_k,
        target_score=target_score,
        global_best_k=int(selector.n_features_),
        global_best_score=global_score,
        score_table=table,
        selected_features_at_target=target_features,
        selected_features_at_global_best=global_features,
        selector=selector,
        target_selector=target_selector,
        metadata={
            "rule": rule,
            "max_features_to_consider": config.max_features_to_consider,
            "scoring": config.scoring,
            "resolved_scoring": scoring,
            "problem_type": problem_type,
            "n_classes": int(len(classes)),
            "classes": classes.tolist(),
            "estimator": config.estimator,
        },
    )
