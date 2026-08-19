"""Shared selector result objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class SelectionResult:
    name: str
    selected_features: list[str]
    feature_table: pd.DataFrame
    estimator: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_selected(self) -> int:
        return len(self.selected_features)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self.selected_features if column not in X.columns]
        if missing:
            raise KeyError(f"Input is missing selected features: {missing[:10]}")
        return X.loc[:, self.selected_features].copy()
