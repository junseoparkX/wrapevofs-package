# Common errors and warnings

- `Unsupported configuration schema version`: use schema 1.0 or omit the field for a legacy schema-1.0 file.
- `metric_orientation must be 'larger_is_better'`: transform a loss into a documented larger-is-better score before locking.
- `strict regret_constrained_medoid requires minimum_pool_size=1`: strict mode never expands beyond the declared tolerance.
- `best_run_se_scaled ... requires fold_locking_scores`: use fixed-CV rescoring with at least two finite folds or choose an explicit absolute/relative mode.
- `Resume requested, but checkpoint state does not exist`: remove `--resume` only if a fresh search is intended; otherwise restore the matching checkpoint.
- `Resume checkpoint mismatch for ...`: do not bypass it. Restore the exact package, configuration, input, feature order, and backend used to create the state.
- `Resume checkpoint is corrupt or unreadable`: use an earlier trusted checkpoint or start a separately documented fresh run.
- GPU requested but unavailable: install a compatible Linux CUDA/RAPIDS stack or select `cpu`; do not interpret `auto` fallback as GPU validation.
- `uniform parent sampling fallback`: inspect the reason and objective diagnostics. In recommended mode this can arise from all-equal or unusable raw objectives; the event is audited.
- high `fraction_legacy_zero`: this is a diagnostic in recommended mode, not proof that recommended parent sampling was uniform.
- mixed development metrics warning: set a compatible `scoring.unified_metric` for new analyses or preserve the warning for intentional legacy reproduction.
