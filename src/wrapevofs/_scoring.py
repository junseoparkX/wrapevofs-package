"""Scoring helpers shared by wrapper and RFECV selectors."""

from __future__ import annotations

from typing import Any

import numpy as np


def unique_classes(y: Any) -> np.ndarray:
    """Return sorted unique class labels from a pandas or numpy target vector."""

    values = y.to_numpy() if hasattr(y, "to_numpy") else y
    return np.unique(np.asarray(values))


def infer_problem_type(y: Any) -> str:
    """Infer whether a classification target is binary or multiclass."""

    classes = unique_classes(y)
    return "binary" if len(classes) <= 2 else "multiclass"


def resolve_auto_scoring(
    scoring: str,
    y: Any,
    *,
    binary: str,
    multiclass: str,
) -> str:
    """Resolve ``scoring='auto'`` from the observed number of target classes."""

    if scoring != "auto":
        return scoring
    return binary if infer_problem_type(y) == "binary" else multiclass
