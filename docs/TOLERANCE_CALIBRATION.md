# Multi-bank development-only tolerance calibration

The opt-in 0.3.0 API fits one absolute tolerance for multiple candidate banks.
It accepts development summaries only, performs no fitting or file access, and
does not change existing pipeline settings or single-bank locking behavior.

## API and inputs

```python
from wrapevofs import LockingCandidate, calibrate_tolerance

banks = {
    "a": [LockingCandidate(0, ["f1"], 0.90, [0.88, 0.92])],
    "b": [LockingCandidate(0, ["f2"], 0.89, [0.87, 0.91])],
}
result = calibrate_tolerance(banks, fold_ids=["split_0", "split_1"])
# Optional comparison on a declared grid:
grid_result = calibrate_tolerance(banks, delta_grid=[0, 0.0025, 0.005, 0.01])
```

`banks` maps at least two nonempty string bank IDs to nonempty sequences of
`LockingCandidate`. Run IDs are integers unique within a bank; they may recur
in other banks. Features must be unique, nonempty strings within each mask.
Duplicate masks in different candidate records retain their voting multiplicity,
as in the existing selector. Optional integer seeds are provenance only.

All candidates must have finite real locking scores and finite one-dimensional
fold vectors with a common length K >= 2. A locking score must equal its
unweighted fold mean within absolute `1e-12`. Every vector must describe the
same development splits, metric, larger-is-better orientation, and fold order.
The caller must establish split alignment; score vectors alone cannot verify it.
Optional unique string `fold_ids` label the shared order, without adding data.
They cannot substitute for establishing that the underlying splits match.

Supply the same ordered `candidate_universe` on every candidate, or omit it
everywhere to use the sorted union of supplied features across all banks.
The universe affects canonical mask hashes and hence exact hash ties. For
reproducing an earlier audit, retain its original complete ordered universe.
Unknown features, mixed or inconsistent universes, duplicate IDs/names, missing
or inconsistent folds, nonfinite values, booleans as scores/IDs and inconsistent
score/fold means are rejected with `ValueError`.

## Exact rule

1. For candidate i in bank b compute the absolute development gap
   `g[b,i] = max_j score[b,j] - score[b,i]`.
2. By default, exhaustively search `D = {0} union {g[b,i] for all b,i}`.
   An explicit `delta_grid` replaces D with its sorted distinct values and must
   be finite, nonnegative and include zero. Use full binary floating-point
   scores and gaps; do not round eligibility thresholds or add epsilon slack.
3. At each delta, apply `lock_representative_run` with strict
   `regret_constrained_medoid` and absolute tolerance independently in every
   bank. Only gaps <= delta are eligible. Maximize mean eligible-peer Jaccard;
   resolve ties by higher locking score, fewer features and canonical mask hash.
   Exact scientifically equivalent records use the lowest run ID as provenance.
4. Let `m0` be the mean score of the signatures selected at delta zero. Average
   their fold vectors over banks first, obtaining a K-vector v. Compute
   `SE0 = sample_SD(v, ddof=1) / sqrt(K)`. Retain deltas whose mean selected
   development score is >= `m0 - SE0`. Delta zero is always feasible. This uses
   a single fixed SE0, without recalculating the filter at other deltas.
5. Among feasible deltas maximize the unweighted mean pairwise Jaccard between
   selected bank signatures. Break ties by higher mean development score, then
   smaller delta. These comparisons use exact computed values, without rounding.

The SE filter is a training heuristic, not a generalization or noninferiority
guarantee. Cross-bank pairs share banks and must not be treated as independent
replicates. Fitted-bank Jaccard is an optimized training objective. Independent
seed banks, external data or a prospectively frozen evaluation are required
for stronger reproducibility or predictive claims.

## Results and applying a frozen allowance

`ToleranceCalibrationResult` returns:

- `selected_delta`: the fitted full-precision absolute allowance;
- `curve`: one row per delta, containing `calibration_Jaccard`,
  `calibration_mean_score`, `bank_averaged_fold_scores`, `delta0_score_SE`,
  `score_threshold`, `score_feasible`, `mean_score_loss_vs_delta0`,
  `selected_run_ids`, `selected_mask_hashes` and the winning `selected` flag;
- `bank_results`: bank ID to existing `LockingResult`, including complete
  candidate audits, pairwise tables, masks, hashes, source IDs and regrets at
  the fitted allowance;
- `metadata`: schema/software versions, search and exact deltas, bank sizes,
  fold labels, canonical universe and hash, baseline score, SE0, fitted overlap,
  selection ties, and SHA-256 of normalized candidate summaries plus search
  settings. `held_out_used` is false. Calibration schema `1.0` is separate
  from the unchanged single-bank artifact schema `2.1`.

Bank IDs and candidate records are canonically ordered for deterministic audits.
Changing a supplied universe can change exact mask-hash ties. Inputs are not
mutated. The API has no participant matrices, labels, predictions or held-out
results argument, and accepts no arbitrary scoring callback or extra keywords.
The caller remains responsible for the development origin of supplied summaries.

```python
from wrapevofs import LockingConfig, lock_representative_run

frozen_delta = result.selected_delta
subsequent = lock_representative_run(new_development_bank, LockingConfig(
    enabled=True,
    strategy="regret_constrained_medoid",
    tolerance_mode="absolute",
    regret_tolerance=frozen_delta,
))
# Save with full precision; nested dictionaries/lists remain auditable.
import json
with open("calibration_curve.json", "w", encoding="utf-8") as stream:
    json.dump(result.curve.to_dict("records"), stream, indent=2, allow_nan=False)
```

Python's `json.dump` also preserves the precision of `result.metadata`.
Avoid decimal rounding when reloading eligibility thresholds. The synthetic example in
`examples/toy_tolerance_calibration.py` runs without participant data.

## Scope of the motivating exploratory evidence

The method was designed retrospectively after earlier results were available.
The motivating comparison used 240 candidate masks on the same historical TCGA
392-development/99-test partition, with 16 five-candidate banks per branch.
Development-only selection did not use test AUROC to choose delta. This does
not make the later method a prospectively specified or external validation.

Across the three branches under the original scorer, full-240 fitted-bank
Jaccard was 0.131759. Mean historical test AUROC was 0.846078, versus 0.847383
for earlier grid calibration, 0.846256 for fixed delta 0.01, and 0.846580 for
highest development score. All three differences' simultaneous 95% intervals
included zero. These aggregate exploratory results do not establish improved
AUROC, optimal defaults or independently validated reproducibility. The package
provides the rule as an option, without promoting it as best-performing.
