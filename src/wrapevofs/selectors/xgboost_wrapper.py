"""XGBoost first-stage wrapper selection."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from wrapevofs.config import XGBoostWrapperConfig
from wrapevofs._scoring import infer_problem_type, resolve_auto_scoring
from wrapevofs.selectors._result import SelectionResult


def select_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    config: XGBoostWrapperConfig | None = None,
) -> SelectionResult:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is not installed. Install with: pip install wrapevofs[xgboost]"
        ) from exc

    config = config or XGBoostWrapperConfig()
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    n_classes = len(encoder.classes_)
    problem_type = infer_problem_type(y)
    if n_classes < 2:
        raise ValueError("XGBoost wrapper requires at least two target classes.")
    scoring = resolve_auto_scoring(
        config.scoring,
        y,
        binary="roc_auc",
        multiclass="roc_auc_ovr",
    )

    params = dict(config.base_params)
    if n_classes > 2:
        params.update(
            {
                "objective": "multi:softprob",
                "num_class": n_classes,
                "eval_metric": "mlogloss",
            }
        )
    params.update(
        {
            "random_state": config.random_state,
            "n_jobs": config.n_jobs,
        }
    )
    base_model = XGBClassifier(**params)
    cv = StratifiedKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )
    search = GridSearchCV(
        estimator=base_model,
        param_grid=config.param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=config.n_jobs,
        refit=True,
    )
    search.fit(X, y_encoded)
    model = search.best_estimator_
    model.fit(X, y_encoded)

    table = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False, kind="mergesort")
    selected = table.loc[
        table["importance"] > config.importance_threshold,
        "feature",
    ].tolist()
    if config.top_k is not None:
        selected = selected[: min(config.top_k, len(selected))]
    if not selected:
        raise RuntimeError("XGBoost selected zero features; lower the threshold.")

    table["selected"] = table["feature"].isin(selected)
    return SelectionResult(
        name="xgboost",
        selected_features=selected,
        feature_table=table.reset_index(drop=True),
        estimator=model,
        metadata={
            "best_params": search.best_params_,
            "best_cv_score": float(search.best_score_),
            "scoring": config.scoring,
            "resolved_scoring": scoring,
            "importance_threshold": config.importance_threshold,
            "top_k": config.top_k,
            "problem_type": problem_type,
            "n_classes": n_classes,
            "classes": encoder.classes_.tolist(),
        },
    )
