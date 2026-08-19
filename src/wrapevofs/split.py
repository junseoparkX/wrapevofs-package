"""Train/test split helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

from wrapevofs.config import SplitConfig


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_index: pd.Index
    test_index: pd.Index


def parse_split_ratio(ratio: str | float) -> float:
    """Return test_size from a user-facing ratio.

    Examples:
        "7:3" -> 0.3
        "6:4" -> 0.4
        "8:2" -> 0.2
        0.25 -> 0.25
    """

    if isinstance(ratio, (float, int)):
        test_size = float(ratio)
    elif ":" in ratio:
        left, right = ratio.split(":", maxsplit=1)
        train_part = float(left.strip())
        test_part = float(right.strip())
        if train_part <= 0 or test_part <= 0:
            raise ValueError("Split ratio parts must be positive.")
        test_size = test_part / (train_part + test_part)
    else:
        test_size = float(ratio)
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    return test_size


def _combined_strata(
    y: pd.Series,
    extra: pd.DataFrame | None,
    columns: Iterable[str],
) -> pd.Series | None:
    labels = y.astype(str).copy()
    if extra is not None:
        for column in columns:
            if column in extra.columns:
                labels = labels + "|" + extra[column].astype(str)
    if labels.value_counts().min() < 2:
        target_only = y.astype(str)
        if target_only.value_counts().min() >= 2:
            return target_only
        return None
    return labels


def train_test_split_frame(
    df: pd.DataFrame,
    target_column: str,
    config: SplitConfig | None = None,
    feature_columns: list[str] | None = None,
    drop_columns: list[str] | None = None,
    label_mapping: dict[str, int] | None = None,
) -> SplitData:
    """Split a single dataframe into X/y train/test parts."""

    config = config or SplitConfig()
    if target_column not in df.columns:
        raise KeyError(f"target_column not found: {target_column}")

    drop_set = set(drop_columns or [])
    drop_set.add(target_column)
    if feature_columns is None:
        feature_columns = [column for column in df.columns if column not in drop_set]

    X = df.loc[:, feature_columns].copy()
    y = df.loc[:, target_column].copy()
    if label_mapping is not None:
        y = y.map(label_mapping)
        if y.isna().any():
            raise ValueError("label_mapping did not cover every target value.")
        y = y.astype(int)

    stratify = None
    if config.stratify:
        stratify = _combined_strata(y, df, config.stratify_columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=parse_split_ratio(config.ratio),
        random_state=config.random_state,
        shuffle=config.shuffle,
        stratify=stratify,
    )

    return SplitData(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        train_index=X_train.index,
        test_index=X_test.index,
    )
