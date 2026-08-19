# Artifact schema

The upgraded schema version is `2.0`. Existing core paths are retained; new columns and locking directories are additive.

## GA artifacts

`ga/<branch>/top_solutions.csv` includes:

- `rank`, `run_id`, `score`, `base_score`, `n_features`, `selected_features`;
- `raw_objective`, `legacy_truncated_fitness`;
- `target_deviation`, `penalty_amount`, `stable_mask_hash`.

`ga/<branch>/history.csv` retains historical best-score fields and adds:

- `base_score`, `raw_objective`, `legacy_truncated_fitness`, `sampling_weight`;
- `feature_count`, `target_count`, `target_deviation`, `penalty_amount`;
- `fraction_legacy_zero`, `number_unique_raw_objectives`;
- `uniform_sampling_fallback`, `uniform_sampling_fallback_reason`;
- `population_unique_masks`, `best_raw_objective`, `median_raw_objective`;
- `best_base_score_audit`, `fitness_mode`, `generation_warning_count`.

`summary.json` records `artifact_schema_version`, requested/actual backend, fitness mode, parameters, and warnings. `warnings.json` aggregates pipeline and GA warnings.

## Locking artifacts

`locking/<branch>/locking_candidate_audit.csv` contains one row per retained candidate for the executed lock:

- identity and score: `run_id`, `feature_count`, `canonical_features`, `canonical_mask`, `stable_mask_hash`, `candidate_universe_sha256`, `duplicate_mask_multiplicity`, `locking_score`, `fold_locking_scores`, `score_sd`, `score_se`;
- regret: `absolute_regret`, `relative_regret`;
- eligibility: `eligible`, `eligibility_reason`, `fallback_added`, `eligibility_threshold`, `minimum_pool_size`;
- agreement and decision: `pairwise_jaccard`, `mean_jaccard`, `selected`, `selected_feature_set`, `tie_break_path`;
- provenance: `strategy`, `tolerance_mode`, `regret_tolerance`, `locking_metric`, `seeds`, `software_version`, `configuration_hash`.

`pairwise_jaccard.csv` contains `run_i`, `run_j`, `stable_mask_hash_i`,
`stable_mask_hash_j`, and `jaccard` for the executed eligible pool.
`selected_features.csv`, `selected_features.npy`, and `summary.json` record the
final lock.

For `regret_constrained_medoid`, `eligible` denotes strict membership under the
declared threshold. `fallback_added` is always false, `minimum_pool_size` must
be 1, and `summary.json` records `strict_regret_constraint`,
`selected_absolute_regret`, `selected_relative_regret`, and
`selected_within_declared_tolerance`. Singleton eligible pools are valid.
Their mean Jaccard is undefined and is serialized as missing rather than as
zero or one. Absolute eligibility uses the exact configured comparison
`absolute_regret <= regret_tolerance`, without an additional numerical slack.

The recommended tie path ends with `stable_mask_hash`, computed by the same
package rule used for GA masks: SHA-256 over the canonical binary mask encoded
as `uint8` bytes. `summary.json` records the ordered candidate universe, its
SHA-256 digest, the selected mask digest, the metric orientation, and the
duplicate-mask policy. Duplicate masks are retained as multiple voting
candidates; source-run multiplicity is therefore part of the candidate bank.
If exact duplicates remain tied, `selected_run_id` is a provenance label only,
while `selected_feature_set` identifies every source record carrying the
scientifically selected mask.

## Final feature sets

When locking is enabled, `final_feature_sets.npy` and `locked_feature_sets.npy` contain one selected list per branch. When locking is disabled, the historical GA/RFECV fallback behavior is preserved.

## Schema migration

Schema 2.0 is additive for core GA paths. Consumers should select columns by name rather than position and should check `artifact_schema_version` before requiring new audit columns.
