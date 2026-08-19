import numpy as np
import pandas as pd

from wrapevofs.config import PreprocessingConfig
from wrapevofs.preprocessing import TabularPreprocessor


def test_preprocessor_imputes_and_drops_correlated_features():
    X = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0],
            "b": [1.0, 2.0, 3.0, 4.0],
            "b_copy": [1.0, 2.0, 3.0, 4.0],
            "too_missing": [np.nan, np.nan, np.nan, 1.0],
            "constant": [1.0, 1.0, 1.0, 1.0],
        }
    )
    config = PreprocessingConfig(
        missingness_threshold=0.5,
        impute_strategy="median",
        drop_zero_variance=True,
        scaling="standard",
        correlation_threshold=0.95,
    )
    prep = TabularPreprocessor(config)
    Xt = prep.fit_transform(X)

    assert not Xt.isna().any().any()
    assert "too_missing" in prep.report.dropped_missingness
    assert "constant" in prep.report.dropped_zero_variance
    assert prep.report.dropped_correlation == ["b_copy"]
    assert Xt.columns.tolist() == ["a", "b"]


def test_zero_imputation_is_supported():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [0.0, 1.0, 2.0]})
    prep = TabularPreprocessor(
        PreprocessingConfig(
            missingness_threshold=None,
            impute_strategy="zero",
            scaling="none",
            correlation_threshold=None,
        )
    )
    Xt = prep.fit_transform(X)
    assert Xt.loc[1, "a"] == 0.0


def test_default_preprocessing_only_zero_imputes_without_dropping_columns():
    X = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, 4.0],
            "a_copy": [1.0, np.nan, 3.0, 4.0],
            "constant": [1.0, 1.0, 1.0, 1.0],
            "too_missing": [np.nan, np.nan, np.nan, 2.0],
        }
    )
    Xt = TabularPreprocessor(PreprocessingConfig()).fit_transform(X)

    assert Xt.columns.tolist() == X.columns.tolist()
    assert not Xt.isna().any().any()
    assert Xt.loc[0, "constant"] == 1.0
    assert Xt.loc[0, "a"] == 1.0
    assert Xt.loc[1, "a"] == 0.0
