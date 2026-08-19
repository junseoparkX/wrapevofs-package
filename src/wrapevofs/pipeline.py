"""High-level pipeline API."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

from wrapevofs.config import PipelineConfig
from wrapevofs.locking import (
    LockingResult,
    lock_representative_run,
    score_candidate_feature_sets,
)
from wrapevofs.preprocessing import TabularPreprocessor
from wrapevofs.selectors._result import SelectionResult
from wrapevofs.selectors.boruta_rf import select_boruta_rf
from wrapevofs.selectors.genetic_rf import GeneticRFResult, run_genetic_rf
from wrapevofs.selectors.rfecv_target import RFECVTargetResult, find_rfecv_target
from wrapevofs.selectors.svm_l1_wrapper import select_svm_l1
from wrapevofs.selectors.xgboost_wrapper import select_xgboost
from wrapevofs.split import SplitData, train_test_split_frame


@dataclass
class PreparedData:
    split: SplitData
    preprocessor: TabularPreprocessor
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


@dataclass
class PipelineResult:
    prepared: PreparedData
    first_stage: dict[str, SelectionResult] = field(default_factory=dict)
    rfecv_targets: dict[str, RFECVTargetResult] = field(default_factory=dict)
    ga_results: dict[str, GeneticRFResult] = field(default_factory=dict)
    locking_results: dict[str, LockingResult] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class WrapEvoPipeline:
    """Dataset-agnostic pipeline through RFECV target-k discovery."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.warnings: list[str] = []

    @staticmethod
    def _resolve_unified_metric(metric: str, y: pd.Series) -> str:
        """Map the public metric alias to a scorer accepted by every stage."""

        if metric == "macro_ovr_auroc":
            return "roc_auc" if y.nunique(dropna=False) == 2 else "roc_auc_ovr"
        allowed = {"accuracy", "balanced_accuracy", "roc_auc", "roc_auc_ovr"}
        if metric not in allowed:
            raise ValueError(
                "scoring.unified_metric must be one of: macro_ovr_auroc, accuracy, "
                "balanced_accuracy, roc_auc, roc_auc_ovr."
            )
        return metric

    def _resolved_stage_metrics(self, y: pd.Series) -> tuple[str, str, str]:
        unified = self.config.scoring.unified_metric
        if unified is not None:
            resolved = self._resolve_unified_metric(unified, y)
            return resolved, resolved, resolved
        rfecv = self.config.rfecv.scoring
        if rfecv == "auto":
            rfecv = "roc_auc" if y.nunique(dropna=False) == 2 else "roc_auc_ovr"
        locking = self.config.locking.locking_metric
        if locking in {"auto", "macro_ovr_auroc"}:
            locking = "roc_auc" if y.nunique(dropna=False) == 2 else "roc_auc_ovr"
        return rfecv, self.config.ga.fitness_metric, locking

    def _audit_metric_alignment(self, y: pd.Series) -> None:
        rfecv, ga, locking = self._resolved_stage_metrics(y)
        active = {"rfecv": rfecv, "ga": ga}
        if self.config.locking.enabled:
            active["locking"] = locking
        if len(set(active.values())) > 1:
            message = (
                "Development-stage metric mismatch: "
                + ", ".join(f"{stage}={metric}" for stage, metric in active.items())
                + ". Set scoring.unified_metric to align future analyses."
            )
            if message not in self.warnings:
                self.warnings.append(message)

    def prepare_from_frame(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: list[str] | None = None,
        drop_columns: list[str] | None = None,
        label_mapping: dict[str, int] | None = None,
    ) -> PreparedData:
        split = train_test_split_frame(
            df=df,
            target_column=target_column,
            config=self.config.split,
            feature_columns=feature_columns,
            drop_columns=drop_columns,
            label_mapping=label_mapping,
        )
        preprocessor = TabularPreprocessor(self.config.preprocessing)
        X_train = preprocessor.fit_transform(split.X_train)
        X_test = preprocessor.transform(split.X_test)
        return PreparedData(
            split=split,
            preprocessor=preprocessor,
            X_train=X_train,
            X_test=X_test,
            y_train=split.y_train,
            y_test=split.y_test,
        )

    def run_first_stage(
        self,
        prepared: PreparedData,
        methods: list[str] | None = None,
    ) -> dict[str, SelectionResult]:
        methods = methods or self.config.first_stage.enabled_methods
        results: dict[str, SelectionResult] = {}
        for method in methods:
            try:
                if method == "xgboost" and self.config.first_stage.xgboost.enabled:
                    results[method] = select_xgboost(
                        prepared.X_train,
                        prepared.y_train,
                        self.config.first_stage.xgboost,
                    )
                elif method == "svm_l1" and self.config.first_stage.svm_l1.enabled:
                    results[method] = select_svm_l1(
                        prepared.X_train,
                        prepared.y_train,
                        self.config.first_stage.svm_l1,
                    )
                elif method == "boruta_rf" and self.config.first_stage.boruta_rf.enabled:
                    results[method] = select_boruta_rf(
                        prepared.X_train,
                        prepared.y_train,
                        self.config.first_stage.boruta_rf,
                    )
                else:
                    self.warnings.append(f"Unknown or disabled method skipped: {method}")
            except ImportError as exc:
                if self.config.first_stage.skip_missing_optional:
                    self.warnings.append(f"{method} skipped: {exc}")
                    continue
                raise
        return results

    def find_rfecv_targets(
        self,
        prepared: PreparedData,
        first_stage: dict[str, SelectionResult],
    ) -> dict[str, RFECVTargetResult]:
        self._audit_metric_alignment(prepared.y_train)
        targets: dict[str, RFECVTargetResult] = {}
        for name, selection in first_stage.items():
            X_selected = selection.transform(prepared.X_train)
            targets[name] = find_rfecv_target(
                X_selected,
                prepared.y_train,
                self._rfecv_config_for_data(name, prepared.y_train),
            )
        return targets

    def _rfecv_config_for_method(self, method: str):
        method_cap = self.config.rfecv.method_max_features_to_consider.get(method)
        if method in self.config.rfecv.method_max_features_to_consider:
            return replace(self.config.rfecv, max_features_to_consider=method_cap)
        return self.config.rfecv

    def _rfecv_config_for_data(self, method: str, y: pd.Series):
        config = self._rfecv_config_for_method(method)
        if self.config.scoring.unified_metric is not None:
            config = replace(
                config,
                scoring=self._resolve_unified_metric(
                    self.config.scoring.unified_metric,
                    y,
                ),
            )
        return config

    def run_ga(
        self,
        prepared: PreparedData,
        first_stage: dict[str, SelectionResult],
        rfecv_targets: dict[str, RFECVTargetResult],
    ) -> dict[str, GeneticRFResult]:
        if not self.config.ga.enabled:
            return {}

        results: dict[str, GeneticRFResult] = {}
        for name, selection in first_stage.items():
            if name not in rfecv_targets:
                self.warnings.append(f"GA skipped for {name}: RFECV target missing.")
                continue
            X_selected = selection.transform(prepared.X_train)
            target = rfecv_targets[name]
            ga_config = self.config.ga
            if self.config.scoring.unified_metric is not None:
                ga_config = replace(
                    ga_config,
                    fitness_metric=self._resolve_unified_metric(
                        self.config.scoring.unified_metric,
                        prepared.y_train,
                    ),
                )
            if ga_config.checkpoint_dir:
                ga_config = replace(
                    ga_config,
                    checkpoint_dir=str(Path(ga_config.checkpoint_dir) / "ga" / name),
                )
            results[name] = run_genetic_rf(
                X_selected,
                prepared.y_train,
                target_k=target.target_k,
                name=f"{name}_ga_rf",
                config=ga_config,
            )
        return results

    def run_locking(
        self,
        prepared: PreparedData,
        first_stage: dict[str, SelectionResult],
        ga_results: dict[str, GeneticRFResult],
    ) -> dict[str, LockingResult]:
        """Score and lock retained GA candidates using development data only."""

        if not self.config.locking.enabled:
            return {}
        results: dict[str, LockingResult] = {}
        for name, ga_result in ga_results.items():
            if name not in first_stage:
                self.warnings.append(f"Locking skipped for {name}: Direct branch missing.")
                continue
            candidate_sets = {
                int(solution.run_id): list(solution.selected_features)
                for solution in ga_result.top_solutions
            }
            if len(candidate_sets) < self.config.ga.n_runs:
                self.warnings.append(
                    f"Locking for {name} received {len(candidate_sets)} retained candidates "
                    f"for {self.config.ga.n_runs} GA runs; set ga.top_k >= ga.n_runs "
                    "to audit every run."
                )
            X_development = first_stage[name].transform(prepared.X_train)
            run_seeds = {
                int(solution.run_id): self.config.ga.random_state + int(solution.run_id)
                for solution in ga_result.top_solutions
            }
            candidates = score_candidate_feature_sets(
                X_development,
                prepared.y_train,
                candidate_sets,
                locking_metric=(
                    self._resolve_unified_metric(
                        self.config.scoring.unified_metric,
                        prepared.y_train,
                    )
                    if self.config.scoring.unified_metric is not None
                    else self.config.locking.locking_metric
                ),
                cv_folds=self.config.locking.cv_folds,
                random_state=self.config.locking.random_state,
                rf_params=self.config.ga.rf_params,
                n_jobs=self.config.ga.n_jobs,
                run_seeds=run_seeds,
            )
            results[name] = lock_representative_run(
                candidates,
                self.config.locking,
                full_configuration=self.config.to_dict(),
                seeds={
                    "locking_cv_seed": self.config.locking.random_state,
                    "ga_base_seed": self.config.ga.random_state,
                },
            )
        return results

    def run_until_rfecv(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: list[str] | None = None,
        drop_columns: list[str] | None = None,
        label_mapping: dict[str, int] | None = None,
        methods: list[str] | None = None,
    ) -> PipelineResult:
        prepared = self.prepare_from_frame(
            df=df,
            target_column=target_column,
            feature_columns=feature_columns,
            drop_columns=drop_columns,
            label_mapping=label_mapping,
        )
        first_stage = self.run_first_stage(prepared, methods=methods)
        rfecv_targets = self.find_rfecv_targets(prepared, first_stage)
        return PipelineResult(
            prepared=prepared,
            first_stage=first_stage,
            rfecv_targets=rfecv_targets,
            warnings=list(self.warnings),
        )

    def run_full(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: list[str] | None = None,
        drop_columns: list[str] | None = None,
        label_mapping: dict[str, int] | None = None,
        methods: list[str] | None = None,
    ) -> PipelineResult:
        prepared = self.prepare_from_frame(
            df=df,
            target_column=target_column,
            feature_columns=feature_columns,
            drop_columns=drop_columns,
            label_mapping=label_mapping,
        )
        first_stage = self.run_first_stage(prepared, methods=methods)
        rfecv_targets = self.find_rfecv_targets(prepared, first_stage)
        ga_results = self.run_ga(prepared, first_stage, rfecv_targets)
        locking_results = self.run_locking(prepared, first_stage, ga_results)
        return PipelineResult(
            prepared=prepared,
            first_stage=first_stage,
            rfecv_targets=rfecv_targets,
            ga_results=ga_results,
            locking_results=locking_results,
            warnings=list(self.warnings),
        )

    def run_until_ga(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: list[str] | None = None,
        drop_columns: list[str] | None = None,
        label_mapping: dict[str, int] | None = None,
        methods: list[str] | None = None,
    ) -> PipelineResult:
        return self.run_full(
            df=df,
            target_column=target_column,
            feature_columns=feature_columns,
            drop_columns=drop_columns,
            label_mapping=label_mapping,
            methods=methods,
        )
