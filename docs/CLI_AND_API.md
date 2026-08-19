# CLI and Python API

## CLI

```bash
wrapevofs run --help
wrapevofs --version
```

New methodological options:

- `--ga-fitness-mode`: legacy zero-truncated or recommended untruncated shifted-linear mode;
- `--locking-strategy`: legacy top-k or regret-constrained medoid;
- `--locking-tolerance-mode`: absolute, relative, best-run-SE-scaled, or the deprecated `one_se` alias;
- `--regret-tolerance`: nonnegative development-score tolerance;
- `--minimum-pool-size`: required eligible-pool size;
- `--unified-metric`: compatible metric applied across RFECV, GA, and locking.
- `--resume`: require matching atomic GA resume state; never silently restart.

Passing `--locking-strategy` enables locking. Locking executes only when `--run-ga` is also used because retained GA candidates are required. `--resume` also requires `--run-ga`.

## Python configuration

`PipelineConfig` contains `ga`, `locking`, and `scoring` sections. New fields are serializable through `PipelineConfig.from_dict`, `from_yaml`, and `to_dict`.

`WrapEvoPipeline.run_locking(prepared, first_stage, ga_results)` rescored retained masks on `prepared.X_train` and `prepared.y_train` only. The public `score_candidate_feature_sets` and `lock_representative_run` functions have no held-out arguments.

`validate_locking_artifact_directory(path)` checks required CSV/JSON fields, package/schema versions, selected-feature consistency, and strict selected eligibility.

## Best-run-SE-scaled requirements

Best-run-SE-scaled eligibility requires at least two finite development-fold locking scores for the best run. If fold vectors are unavailable, the API raises an actionable error. It does not reconstruct folds from an SD or invent an SE. The `one_se` spelling is retained only as a compatibility alias; this mode is not the conventional paired one-standard-error rule.

For the recommended absolute regret-constrained medoid, the eligible pool is never expanded: the highest-scoring candidate guarantees a nonempty pool, and singleton pools are valid. The final deterministic tie path uses higher locking score, smaller feature count, and the stable hash of the canonical feature mask. Source run IDs remain provenance labels.

## Metric alignment

`scoring.unified_metric: macro_ovr_auroc` resolves to `roc_auc` for binary outcomes and `roc_auc_ovr` for multiclass outcomes. Without a unified metric, the pipeline preserves configured legacy objectives and emits a warning when active development-stage metrics differ.
