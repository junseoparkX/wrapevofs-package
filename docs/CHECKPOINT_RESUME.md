# GA checkpoint and resume

## Files

When `ga.checkpoint_dir` is configured, each branch writes human-readable live files (`checkpoint.json`, `history_live.csv`, `top_solutions_live.csv`, live masks/features/scores) and `resume_state.npz`.

Only `resume_state.npz` is authoritative for continuation. It is a single compressed NumPy archive written to a same-directory temporary file and atomically replaced. It contains numeric population state and JSON text; loading uses `allow_pickle=False`.

## Resume point

A state is written after a generation's evaluation, audit row, parent selection, crossover, and mutation have produced the next population. It stores the next generation index and NumPy generator state. Therefore a resumed run evaluates the same next population with the same deterministic generation seed as an uninterrupted run.

## Validation

Resume requires exact agreement on:

- checkpoint-state, package, and artifact-schema versions;
- branch name and RFECV target;
- requested and actual backend;
- scientific GA configuration fingerprint;
- exact ordered development input fingerprint;
- ordered feature-universe fingerprint and labels;
- population size, run count, generation count, and population shape;
- serialized solution masks, feature labels, and authoritative stable-mask hashes;
- compatible NumPy random-generator state.

Corrupt archives, missing members/fields, or any mismatch raise `ValueError`. A requested resume never falls back to a fresh GA.

## Commands

```bash
wrapevofs run ... --run-ga
wrapevofs run ... --run-ga --resume
```

The same input, configuration, feature order, package, schema, and backend must be used. Moving the checkpoint directory is allowed because output paths do not affect the scientific trajectory.

## Security

Resume state is not a signed or authenticated format. Use it only from a trusted workspace. ZIP integrity and structural/fingerprint checks detect ordinary damage and mismatch, but they are not a defense against a malicious party who can rewrite both data and metadata. Human-facing legacy `.npy` exports may use object arrays and should likewise be loaded only from trusted sources.
