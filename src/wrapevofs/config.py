"""Configuration objects for the feature-selection package."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


def _replace_dataclass(instance: Any, values: dict[str, Any] | None) -> Any:
    if values is None:
        return instance
    valid = {item.name for item in fields(instance)}
    clean = {key: value for key, value in values.items() if key in valid}
    for key, value in clean.items():
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            setattr(instance, key, _replace_dataclass(current, value))
        elif isinstance(current, dict) and isinstance(value, dict):
            merged = dict(current)
            merged.update(value)
            setattr(instance, key, merged)
        else:
            setattr(instance, key, value)
    return instance


@dataclass
class SplitConfig:
    """Train/test split behavior.

    ratio accepts "7:3", "6:4", "8:2", or a numeric test size such as 0.3.
    """

    ratio: str | float = "7:3"
    random_state: int = 42
    shuffle: bool = True
    stratify: bool = True
    stratify_columns: list[str] = field(default_factory=list)


@dataclass
class PreprocessingConfig:
    """Generic numeric tabular preprocessing.

    impute_strategy is intentionally small and user-facing:
    - "median"
    - "mean"
    - "zero"
    """

    numeric_only: bool = True
    missingness_threshold: float | None = None
    impute_strategy: str = "zero"
    drop_zero_variance: bool = False
    scaling: str = "none"
    correlation_threshold: float | None = None
    error_on_missing_features: bool = True


@dataclass
class XGBoostWrapperConfig:
    enabled: bool = True
    importance_threshold: float = 0.0
    top_k: int | None = 100
    cv_folds: int = 5
    scoring: str = "auto"
    n_jobs: int = -1
    random_state: int = 42
    param_grid: dict[str, list[Any]] = field(
        default_factory=lambda: {
            "subsample": [0.6, 0.7, 0.8],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "n_estimators": [80, 100, 120],
        }
    )
    base_params: dict[str, Any] = field(
        default_factory=lambda: {
            "objective": "binary:logistic",
            "booster": "gbtree",
            "learning_rate": 0.3,
            "gamma": 0,
            "max_depth": 6,
            "reg_lambda": 1,
            "reg_alpha": 0,
            "eval_metric": "logloss",
            "tree_method": "hist",
        }
    )


@dataclass
class SVML1WrapperConfig:
    enabled: bool = True
    coefficient_threshold: float = 1e-6
    standardize: bool = True
    cv_folds: int = 5
    scoring: str = "auto"
    n_jobs: int = -1
    random_state: int = 42
    c_grid: list[float] = field(
        default_factory=lambda: [0.001, 0.01, 0.1, 1.0, 10.0, 30.0, 100.0]
    )
    penalty: str = "l1"
    dual: bool = False
    max_iter: int = 100000
    tol: float = 1e-4
    class_weight: str | None = None


@dataclass
class BorutaRFWrapperConfig:
    enabled: bool = True
    include_tentative: bool = True
    random_state: int = 42
    rf_n_estimators: int = 1000
    rf_max_depth: int | None = 5
    rf_class_weight: str | None = "balanced"
    n_estimators: str | int = "auto"
    perc: int = 95
    alpha: float = 0.05
    two_step: bool = True
    max_iter: int = 150
    verbose: int = 0


@dataclass
class FirstStageConfig:
    enabled_methods: list[str] = field(
        default_factory=lambda: ["xgboost", "svm_l1", "boruta_rf"]
    )
    skip_missing_optional: bool = True
    xgboost: XGBoostWrapperConfig = field(default_factory=XGBoostWrapperConfig)
    svm_l1: SVML1WrapperConfig = field(default_factory=SVML1WrapperConfig)
    boruta_rf: BorutaRFWrapperConfig = field(default_factory=BorutaRFWrapperConfig)


@dataclass
class RFECVConfig:
    """RFECV target-k configuration.

    method_max_features_to_consider implements branch-specific compact rules.
    For example, the default compact caps use XGBoost <= 20 and SVM-L1 <= 20.
    max_features_to_consider is the fallback for methods not listed there.
    """

    estimator: str = "random_forest"
    max_features_to_consider: int | None = 25
    method_max_features_to_consider: dict[str, int | None] = field(
        default_factory=lambda: {
            "xgboost": 20,
            "svm_l1": 20,
            "boruta_rf": 25,
        }
    )
    cv_folds: int = 5
    scoring: str = "auto"
    step: int = 1
    min_features_to_select: int = 1
    n_jobs: int = 1
    random_state: int = 42
    rf_params: dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 150,
            "max_depth": 12,
            "min_samples_leaf": 2,
            "bootstrap": True,
        }
    )


@dataclass
class GAConfig:
    """Genetic Algorithm Random Forest configuration.

    target_k is intentionally not exposed here. It is supplied by RFECV.
    backend controls the Random Forest evaluator used inside GA:
    - "cpu": scikit-learn RandomForestClassifier
    - "gpu": cuML RandomForestClassifier
    - "auto": cuML when available, otherwise scikit-learn
    """

    enabled: bool = True
    backend: str = "auto"
    population_size: int = 50
    n_generations: int = 50
    n_runs: int = 5
    crossover_rate: float = 0.8
    mutation_rate: float = 0.05
    elitism_count: int = 2
    initial_off_ratio: float = 0.30
    size_penalty_lambda: float = 0.015
    fitness_mode: str = "legacy_zero_truncated_linear"
    sampling_epsilon: float = 1e-12
    legacy_zero_fraction_warning: float = 0.50
    target_deviation_warning_threshold: int | None = None
    minimum_unique_masks_warning: int = 2
    top_k: int = 5
    cv_folds: int = 5
    fitness_metric: str = "accuracy"
    random_state: int = 42
    n_jobs: int = 1
    verbose: bool = False
    progress_interval: int = 10
    checkpoint_dir: str | None = None
    rf_params: dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 150,
            "max_depth": 12,
            "min_samples_leaf": 2,
            "bootstrap": True,
        }
    )
    gpu_rf_params: dict[str, Any] = field(
        default_factory=lambda: {
            "n_bins": 16,
        }
    )


@dataclass
class LockingConfig:
    """Development-only representative-run locking configuration.

    The disabled, top-k defaults preserve the package's archived artifact
    behavior. Recommended new analyses explicitly enable
    ``regret_constrained_medoid`` in a separate configuration.
    """

    enabled: bool = False
    strategy: str = "top_k_jaccard_medoid"
    top_k: int = 3
    tolerance_mode: str = "absolute"
    regret_tolerance: float = 0.01
    minimum_pool_size: int = 1
    fallback_rule: str = "strict_eligible_only"
    tie_breakers: list[str] = field(
        default_factory=lambda: [
            "higher_locking_score",
            "smaller_feature_count",
            "stable_mask_hash",
        ]
    )
    epsilon: float = 1e-12
    metric_orientation: str = "larger_is_better"
    locking_metric: str = "auto"
    cv_folds: int = 5
    random_state: int = 42


@dataclass
class ScoringConfig:
    """Optional future alignment of compatible development-stage metrics."""

    unified_metric: str | None = None


@dataclass
class PipelineConfig:
    split: SplitConfig = field(default_factory=SplitConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    first_stage: FirstStageConfig = field(default_factory=FirstStageConfig)
    rfecv: RFECVConfig = field(default_factory=RFECVConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    locking: LockingConfig = field(default_factory=LockingConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "PipelineConfig":
        config = cls()
        if not values:
            return config
        for section in (
            "split",
            "preprocessing",
            "first_stage",
            "rfecv",
            "ga",
            "locking",
            "scoring",
        ):
            if section in values:
                _replace_dataclass(getattr(config, section), values[section])
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        return cls.from_dict(values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
