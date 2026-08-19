# Configuration reference

Configuration schema version: `1.0`. The top-level sections are `split`, `preprocessing`, `first_stage`, `rfecv`, `ga`, `locking`, and `scoring`. Files without `config_schema_version` are accepted as legacy schema 1.0; any explicit unsupported version is rejected.

## Split and preprocessing

- `split.ratio`: presets such as `7:3`, `6:4`, `8:2`, or a numeric test fraction.
- `random_state`, `shuffle`, `stratify`, `stratify_columns`: deterministic split controls.
- `preprocessing.numeric_only`: retain numeric features.
- `missingness_threshold`: optional train-fitted feature filter.
- `impute_strategy`: `zero`, `mean`, or `median`; fitted on development data.
- `drop_zero_variance`, `correlation_threshold`, `scaling`: optional train-fitted transformations.
- `error_on_missing_features`: reject transform-time feature mismatch.

Preprocessing is fitted on the development partition and then applied to held-out data. Held-out statistics do not choose preprocessing parameters.

## Direct branches

`first_stage.enabled_methods` may contain `xgboost`, `svm_l1`, and `boruta_rf`. Each branch has model, scoring, CV, seed, and threshold settings. Optional packages may be skipped only when `skip_missing_optional` is true; this is reported rather than silently interpreted as a completed branch.

## RFECV

- `estimator`: current implementation uses Random Forest.
- `method_max_features_to_consider`: branch-specific compact cap.
- `max_features_to_consider`: fallback cap.
- `cv_folds`, `scoring`, `step`, `min_features_to_select`, `n_jobs`, `random_state`.
- `rf_params`: estimator parameters.

`target_k` is produced by RFECV for a branch and is not a user-facing GA field.

## GA

- `backend`: `cpu`, `gpu`, or `auto`.
- `population_size`, `n_generations`, `n_runs`.
- `crossover_rate`, `mutation_rate`, `elitism_count`, `initial_off_ratio`.
- `size_penalty_lambda`: penalty per unit absolute deviation from RFECV `target_k`.
- `fitness_mode`: `legacy_zero_truncated_linear` or recommended `untruncated_shifted_linear`.
- `sampling_epsilon`: positive offset used only in recommended sampling weights.
- diagnostic thresholds: `legacy_zero_fraction_warning`, `target_deviation_warning_threshold`, `minimum_unique_masks_warning`.
- `top_k`, `cv_folds`, `fitness_metric`, `random_state`, `n_jobs`.
- `verbose`, `progress_interval`, `checkpoint_dir`.
- `resume_from_checkpoint`: require and validate an existing resumable state.
- `rf_params` and optional `gpu_rf_params`.

Fields that affect scientific execution are included in the checkpoint configuration fingerprint. Output location, verbosity, progress interval, and the resume switch are excluded because they do not alter the search trajectory.

## Locking

- `enabled`.
- `strategy`: legacy `top_k_jaccard_medoid` or `regret_constrained_medoid`.
- `tolerance_mode`: `absolute`, `relative`, or `best_run_se_scaled`; `one_se` is a deprecated alias for the last and is not the conventional paired one-standard-error rule.
- `regret_tolerance`: nonnegative score-gap or relative-gap tolerance.
- `minimum_pool_size`: strict regret mode requires exactly 1.
- `fallback_rule`: strict regret mode requires `strict_eligible_only`.
- canonical `tie_breakers`: higher score, smaller feature count, stable mask hash.
- `metric_orientation`: package locking requires `larger_is_better`.
- `locking_metric`, `cv_folds`, `random_state`.

Lower-is-better losses must be transformed into a documented larger-is-better locking score before use. Passing a lower-is-better orientation is rejected.

## Scoring

`scoring.unified_metric` may be `accuracy`, `balanced_accuracy`, `roc_auc`, or `macro_ovr_auroc`. The last resolves to binary `roc_auc` or multiclass `roc_auc_ovr`. If omitted, historical stage-specific objectives are preserved and metric disagreement is warned.
