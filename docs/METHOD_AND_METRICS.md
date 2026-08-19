# Methods, metrics, and scientific boundaries

## Development-only execution

The package accepts already extracted tabular features. It does not process raw imaging, omics assays, or participant source files. Split, preprocessing fit, Direct selection, RFECV target discovery, GA search, fixed-CV candidate rescoring, regret eligibility, and locking occur on development data. Held-out metrics are for evaluation after selection and cannot be passed to the locking API.

## Direct branches and RFECV target

XGBoost positive-importance, SVM-L1 nonzero-coefficient, and Boruta-RF all-relevant selections define branch-specific candidate universes. RFECV estimates a development-CV feature count `target_k`. The GA penalty encourages proximity to that count but does not impose it as a hard subset-size constraint.

## Recommended GA objective

For base development-CV score `B(m)` of mask `m`, selected-feature count `|m|`, RFECV target `k`, and penalty `lambda`, the raw objective is

`Q(m) = B(m) - lambda * ||m| - k|`.

Recommended mode ranks by `Q` without truncation. For parent sampling only, finite unequal objectives are shifted by their finite minimum and a positive epsilon. This preserves their order while producing nonnegative weights. If objectives are all nonfinite, effectively equal, or produce an unusable total, the implementation uses uniform sampling and records the reason. The legacy value `max(0, Q)` is retained as a diagnostic and compatibility value.

## Fixed-CV rescoring and regret locking

For larger-is-better locking score `L_r`, define `L_best = max_r L_r`, absolute empirical regret `R_r = L_best - L_r`, and absolute eligible pool `E_delta = {r: R_r <= delta}`. The pool is nonempty for nonnegative `delta` because every best-scoring candidate has zero regret. Strict locking selects only within this pool. A singleton is selected directly; otherwise the candidate maximizing mean Jaccard similarity to other eligible candidates is the medoid.

Relative regret divides the absolute gap by `max(|L_best|, epsilon)`. Best-run-SE-scaled mode uses the best run's finite fold-score standard error as an absolute threshold. It is not a paired one-standard-error comparison.

## Stable tie-breaking and duplicates

Feature names are encoded against one ordered candidate universe. The mask is cast to canonical `uint8` bytes and hashed with SHA-256 using the finalized package rule. The digest provides stable ordering; it is not a random probability or scientific score. Collision resistance is an engineering assumption and is not the source of the regret guarantee.

Duplicate masks from separate source runs remain separate voting candidates. Their multiplicity therefore changes eligible-pool Jaccard averages by design. When exact duplicate records remain tied, they are the same scientific feature set; source run ID is stored only as provenance after mask selection. Input-row order is irrelevant.

## Supported metrics and orientation

- `accuracy`: binary or multiclass.
- `balanced_accuracy`: binary or multiclass.
- `roc_auc`: binary probability/ranking score.
- `roc_auc_ovr` / `macro_ovr_auroc`: multiclass macro one-versus-rest probability score.

Locking uses a larger-is-better convention. A loss must be transformed before use and that transformation must be documented in downstream work.

## Claim boundary

The method guarantees only configured empirical development-CV score-gap feasibility for the candidate bank supplied. It does not guarantee globally optimal feature selection, expected predictive risk, unbiased generalization error, uncertainty calibration, predictive superiority, biomarker validity, external validity, or stability under participant resampling.
