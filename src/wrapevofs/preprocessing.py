"""Dataset-agnostic numeric preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from wrapevofs.config import PreprocessingConfig


@dataclass
class PreprocessingReport:
    input_features: int = 0
    after_missingness: int = 0
    after_zero_variance: int = 0
    after_correlation: int = 0
    dropped_missingness: list[str] = field(default_factory=list)
    dropped_zero_variance: list[str] = field(default_factory=list)
    dropped_correlation: list[str] = field(default_factory=list)


class TabularPreprocessor:
    """Fit train-only preprocessing rules and apply them to future data."""

    def __init__(self, config: PreprocessingConfig | None = None):
        self.config = config or PreprocessingConfig()
        self.report = PreprocessingReport()
        self.feature_columns_: list[str] = []
        self.kept_after_missingness_: list[str] = []
        self.kept_after_variance_: list[str] = []
        self.output_features_: list[str] = []
        self.impute_values_: pd.Series | None = None
        self.scale_center_: pd.Series | None = None
        self.scale_scale_: pd.Series | None = None

    def fit(self, X: pd.DataFrame) -> "TabularPreprocessor":
        X_num = self._coerce_numeric(X)
        self.feature_columns_ = list(X_num.columns)
        self.report.input_features = len(self.feature_columns_)

        X_num = X_num.replace([np.inf, -np.inf], np.nan)

        if self.config.missingness_threshold is not None:
            missing_rate = X_num.isna().mean()
            keep_mask = missing_rate <= self.config.missingness_threshold
            self.kept_after_missingness_ = missing_rate.index[keep_mask].tolist()
            self.report.dropped_missingness = missing_rate.index[~keep_mask].tolist()
        else:
            self.kept_after_missingness_ = list(X_num.columns)
        self.report.after_missingness = len(self.kept_after_missingness_)

        X_work = X_num.loc[:, self.kept_after_missingness_]
        self.impute_values_ = self._fit_imputer(X_work)
        X_work = X_work.fillna(self.impute_values_)

        if self.config.drop_zero_variance:
            variance = X_work.var(axis=0, ddof=0)
            keep_mask = variance > 0
            self.kept_after_variance_ = variance.index[keep_mask].tolist()
            self.report.dropped_zero_variance = variance.index[~keep_mask].tolist()
        else:
            self.kept_after_variance_ = list(X_work.columns)
        self.report.after_zero_variance = len(self.kept_after_variance_)

        X_work = X_work.loc[:, self.kept_after_variance_]
        self._fit_scaler(X_work)
        X_work = self._apply_scaler(X_work)

        dropped_corr: list[str] = []
        if self.config.correlation_threshold is not None and X_work.shape[1] > 1:
            corr = X_work.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            dropped_corr = [
                column
                for column in upper.columns
                if bool((upper[column] > self.config.correlation_threshold).any())
            ]
        self.report.dropped_correlation = dropped_corr
        self.output_features_ = [
            column for column in X_work.columns if column not in set(dropped_corr)
        ]
        self.report.after_correlation = len(self.output_features_)

        if not self.output_features_:
            raise ValueError("Preprocessing removed every feature.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_is_fitted()
        X_num = self._coerce_numeric(X)
        missing = [column for column in self.feature_columns_ if column not in X_num.columns]
        if missing and self.config.error_on_missing_features:
            raise KeyError(f"Missing features during transform: {missing[:10]}")

        for column in missing:
            X_num[column] = np.nan

        X_work = X_num.loc[:, self.kept_after_missingness_]
        X_work = X_work.replace([np.inf, -np.inf], np.nan)
        X_work = X_work.fillna(self.impute_values_)
        X_work = X_work.loc[:, self.kept_after_variance_]
        X_work = self._apply_scaler(X_work)
        return X_work.loc[:, self.output_features_].copy()

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def get_feature_names_out(self) -> list[str]:
        self._check_is_fitted()
        return list(self.output_features_)

    def _coerce_numeric(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.config.numeric_only:
            columns = X.select_dtypes(include=[np.number]).columns.tolist()
            return X.loc[:, columns].copy()
        return X.apply(pd.to_numeric, errors="coerce")

    def _fit_imputer(self, X: pd.DataFrame) -> pd.Series:
        strategy = self.config.impute_strategy.lower()
        if strategy == "median":
            values = X.median(axis=0)
        elif strategy == "mean":
            values = X.mean(axis=0)
        elif strategy == "zero":
            values = pd.Series(0.0, index=X.columns)
        else:
            raise ValueError("impute_strategy must be one of: median, mean, zero.")
        return values.fillna(0.0)

    def _fit_scaler(self, X: pd.DataFrame) -> None:
        scaling = self.config.scaling.lower()
        if scaling == "none":
            self.scale_center_ = pd.Series(0.0, index=X.columns)
            self.scale_scale_ = pd.Series(1.0, index=X.columns)
        elif scaling == "standard":
            self.scale_center_ = X.mean(axis=0)
            scale = X.std(axis=0, ddof=0).replace(0, 1.0)
            self.scale_scale_ = scale.fillna(1.0)
        elif scaling == "minmax":
            self.scale_center_ = X.min(axis=0)
            scale = (X.max(axis=0) - X.min(axis=0)).replace(0, 1.0)
            self.scale_scale_ = scale.fillna(1.0)
        else:
            raise ValueError("scaling must be one of: standard, minmax, none.")

    def _apply_scaler(self, X: pd.DataFrame) -> pd.DataFrame:
        return (X - self.scale_center_) / self.scale_scale_

    def _check_is_fitted(self) -> None:
        if self.impute_values_ is None or self.scale_center_ is None:
            raise RuntimeError("TabularPreprocessor is not fitted yet.")
