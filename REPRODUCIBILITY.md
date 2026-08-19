# Reproducibility

## Environment

WrapEvoFS supports Python 3.10–3.12. Create an isolated environment and install the package with its test tools:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

`pyproject.toml` is the authoritative compatibility and dependency specification.

## Tests

```bash
python -m pytest -q
```

The locking tests cover nonempty eligibility, strict regret feasibility, zero-tolerance ties, the all-candidate limit, duplicate masks, feature-order canonicalization, process-stable hashing, deterministic audits, serialization, and candidate-order permutation invariance.

## Synthetic development-only execution

```bash
python scripts/run_development_only_smoke.py --out runs/synthetic_regret_smoke
```

This command uses the synthetic `examples/sample_radiomics.csv` input and `configs/synthetic_regret_smoke.yaml`. RFECV, genetic search, candidate rescoring, and locking use the development partition only.

## Recommended and legacy modes

New analyses should use `configs/recommended_regret_constrained.yaml`. Its core settings are:

- untruncated raw objective for genetic-search ranking and reporting;
- shifted nonnegative weights only for proportional parent sampling;
- absolute development-score regret tolerance of 0.01;
- strict eligible-only locking, with singleton pools allowed;
- deterministic Jaccard-medoid tie-breaking by score, feature count, and stable mask hash;
- compatible macro one-vs-rest AUROC scoring where applicable.

The 0.01 tolerance is a prespecified package operating default on the configured metric scale. It must not be selected by examining held-out results. Other datasets or metrics may require a separately justified, prospectively configured tolerance.

Archived configurations can explicitly retain:

```yaml
ga:
  fitness_mode: legacy_zero_truncated_linear
locking:
  strategy: top_k_jaccard_medoid
```

Legacy modes are retained to reproduce archived analyses. They are not silently replaced by the corrected objective or regret-constrained rule.

## Determinism and audit trail

Configuration objects record split, RFECV, genetic-search, and locking seeds. Canonical feature masks are encoded against one ordered candidate universe and hashed with SHA-256. Stable hashes order scientific ties; they are not probabilistic scores. Source run IDs are retained as provenance labels after feature-set selection.

When `ga.checkpoint_dir` is configured, `resume_state.npz` records the next population, random-generator state, completed history, and fingerprints of the package, configuration, ordered feature universe, and exact development inputs. Resume requires all fingerprints to match and never falls back to a fresh search. See `docs/CHECKPOINT_RESUME.md`.

When distinct source runs contain the same mask, the records remain separate voting candidates. Their multiplicity is part of the supplied candidate bank and is recorded in the locking audit.

See `docs/ARTIFACT_SCHEMA.md` for the complete output schema.

## Data boundary

Do not commit participant-level, controlled-access, or provider-restricted data to this repository. The bundled smoke-test CSV is synthetic. WrapEvoFS begins from already extracted tabular features and does not claim raw imaging or assay reconstruction.
