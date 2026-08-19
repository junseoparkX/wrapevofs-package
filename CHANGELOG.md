# Changelog

## 0.2.0

- Added development-only `regret_constrained_medoid` locking with absolute, relative, and best-run-SE-scaled eligibility.
- Enforced strict eligible-only pools, including singleton pools, without tolerance expansion.
- Added canonical-universe mask encoding and deterministic stable-mask-hash tie-breaking; source run IDs are provenance rather than scientific tie-breaks.
- Documented and audited the retained-multiplicity policy for duplicate masks.
- Added the recommended `untruncated_shifted_linear` GA mode with separate raw objective and sampling weights.
- Preserved explicit `top_k_jaccard_medoid` and `legacy_zero_truncated_linear` compatibility modes.
- Added GA flattening, sampling-fallback, target-deviation, and mask-diversity diagnostics.
- Added versioned locking and GA audit artifacts, configuration hashes, and aggregated warnings.
- Added atomic, fingerprint-validated GA checkpoint resume without silent restart.
- Added package, configuration, and artifact schema validation helpers.
- Added compatible unified metric support and mixed-development-metric warnings.
- Added CLI and Python API options, regression tests, a synthetic one-command example, and schema documentation.
- Added BSD-3-Clause licensing and release-ready package metadata.

## 0.1.0 - 2026-07-04

- Rebuilt the repository as an installable Python package.
- Added train/test split, preprocessing, first-stage wrappers, RFECV target discovery, and GA-RF feature selection.
- Added CSV and `.npy` artifact export for wrapper, RFECV target, GA, and final feature sets.
- Added configuration files, examples, tests, and procedure documentation.
- Added branch-specific RFECV compact caps: XGBoost <= 20, SVM-L1 <= 15, Boruta-RF <= 25.
- Added a synthetic tabular sample dataset for quick package validation.
- Added a compact de-identified radiomics demo CSV and notebook walkthrough.
