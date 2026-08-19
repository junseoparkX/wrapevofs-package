"""Boruta-RF first-stage wrapper selection."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from wrapevofs.config import BorutaRFWrapperConfig
from wrapevofs.selectors._result import SelectionResult


def select_boruta_rf(
    X: pd.DataFrame,
    y: pd.Series,
    config: BorutaRFWrapperConfig | None = None,
) -> SelectionResult:
    try:
        from boruta import BorutaPy
    except ImportError as exc:
        raise ImportError(
            "boruta is not installed. Install with: pip install wrapevofs[boruta]"
        ) from exc

    config = config or BorutaRFWrapperConfig()
    rf = RandomForestClassifier(
        n_estimators=config.rf_n_estimators,
        max_depth=config.rf_max_depth,
        class_weight=config.rf_class_weight,
        random_state=config.random_state,
        n_jobs=-1,
    )
    selector = BorutaPy(
        estimator=rf,
        n_estimators=config.n_estimators,
        perc=config.perc,
        alpha=config.alpha,
        two_step=config.two_step,
        max_iter=config.max_iter,
        random_state=config.random_state,
        verbose=config.verbose,
    )
    selector.fit(X.to_numpy(), y.to_numpy())
    table = pd.DataFrame(
        {
            "feature": X.columns,
            "confirmed": selector.support_,
            "tentative": selector.support_weak_,
            "ranking": selector.ranking_,
        }
    )
    if config.include_tentative:
        selected_mask = table["confirmed"] | table["tentative"]
    else:
        selected_mask = table["confirmed"]
    selected = table.loc[selected_mask, "feature"].tolist()
    if not selected:
        raise RuntimeError("Boruta-RF selected zero features.")
    table["selected"] = table["feature"].isin(selected)
    table = table.sort_values(["selected", "ranking"], ascending=[False, True])
    return SelectionResult(
        name="boruta_rf",
        selected_features=selected,
        feature_table=table.reset_index(drop=True),
        estimator=selector,
        metadata={
            "include_tentative": config.include_tentative,
            "confirmed_count": int(table["confirmed"].sum()),
            "tentative_count": int(table["tentative"].sum()),
        },
    )
