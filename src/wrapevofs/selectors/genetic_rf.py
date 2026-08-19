"""Size-aware Genetic Algorithm Random Forest feature selection."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from wrapevofs.config import GAConfig
from wrapevofs._scoring import infer_problem_type, unique_classes


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


@dataclass
class GASolution:
    rank: int
    run_id: int
    score: float
    base_score: float
    n_features: int
    mask: np.ndarray
    selected_features: list[str]
    raw_objective: float | None = None
    legacy_truncated_fitness: float | None = None
    target_deviation: int | None = None
    penalty_amount: float | None = None
    stable_mask_hash: str | None = None


@dataclass
class GeneticRFResult:
    name: str
    target_k: int
    top_solutions: list[GASolution]
    history: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def best_solution(self) -> GASolution:
        if not self.top_solutions:
            raise RuntimeError("No GA solutions are available.")
        return self.top_solutions[0]

    def transform(self, X: pd.DataFrame, rank: int = 1) -> pd.DataFrame:
        solution = next((item for item in self.top_solutions if item.rank == rank), None)
        if solution is None:
            raise KeyError(f"No GA solution with rank={rank}.")
        return X.loc[:, solution.selected_features].copy()


@dataclass(frozen=True)
class _RFBackend:
    name: str
    model_cls: type[Any]
    fallback_reason: str | None = None


@dataclass(frozen=True)
class _ChromosomeEvaluation:
    base_score: float
    raw_objective: float
    legacy_truncated_fitness: float
    ranking_score: float
    feature_count: int
    target_deviation: int
    penalty_amount: float
    stable_mask_hash: str


@dataclass
class _PopulationEvaluation:
    population: list[np.ndarray]
    ranking_scores: np.ndarray
    base_scores: np.ndarray
    raw_objectives: np.ndarray
    legacy_truncated_fitness: np.ndarray
    feature_counts: np.ndarray
    target_deviations: np.ndarray
    penalty_amounts: np.ndarray
    stable_mask_hashes: list[str]


_CUML_RF_ALLOWED_PARAMS = {
    "n_estimators",
    "split_criterion",
    "bootstrap",
    "max_samples",
    "max_depth",
    "max_leaves",
    "max_features",
    "n_bins",
    "n_streams",
    "min_samples_leaf",
    "min_samples_split",
    "random_state",
    "verbose",
    "output_type",
}


def _cuda_device_count() -> int | None:
    try:
        import cupy as cp

        return int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        return None


def _load_cuml_rf() -> tuple[type[Any] | None, str | None]:
    try:
        from cuml.ensemble import RandomForestClassifier as CURF
    except Exception as exc:
        return None, str(exc)

    device_count = _cuda_device_count()
    if device_count == 0:
        return None, "cuML is importable, but no CUDA GPU was detected."
    return CURF, None


def _resolve_backend(config: GAConfig) -> _RFBackend:
    requested = config.backend.lower()
    if requested in {"cpu", "sklearn"}:
        return _RFBackend("cpu", RandomForestClassifier)
    if requested not in {"auto", "gpu", "cuda", "cuml"}:
        raise ValueError("ga.backend must be one of: auto, cpu, gpu.")

    model_cls, error = _load_cuml_rf()
    if model_cls is not None:
        return _RFBackend("gpu", model_cls)

    if requested in {"gpu", "cuda", "cuml"}:
        raise ImportError(
            "GA GPU backend requested, but cuML is unavailable. "
            "Install RAPIDS cuML in a CUDA-enabled environment, or set ga.backend='cpu'. "
            f"Original error: {error}"
        )

    return _RFBackend("cpu", RandomForestClassifier, fallback_reason=error)


def _to_numpy(values: Any) -> np.ndarray:
    if hasattr(values, "get"):
        values = values.get()
    elif hasattr(values, "to_numpy"):
        values = values.to_numpy()
    return np.asarray(values)


def _to_numpy_1d(values: Any) -> np.ndarray:
    return _to_numpy(values).reshape(-1)


def _make_rf(config: GAConfig, seed: int, backend: _RFBackend) -> Any:
    params = dict(config.rf_params)
    params.update({"random_state": seed})
    if backend.name == "gpu":
        params.update(config.gpu_rf_params)
        params = {
            key: value
            for key, value in params.items()
            if key in _CUML_RF_ALLOWED_PARAMS and value is not None
        }
        return backend.model_cls(**params)

    params.update({"n_jobs": config.n_jobs})
    return backend.model_cls(**params)


def _initial_population(
    population_size: int,
    n_features: int,
    off_ratio: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    if population_size < 2:
        raise ValueError("population_size must be at least 2.")
    if n_features < 1:
        raise ValueError("GA requires at least one feature.")

    off_count = int(off_ratio * n_features)
    population = []
    for _ in range(population_size):
        chromosome = np.ones(n_features, dtype=bool)
        if off_count > 0:
            chromosome[:off_count] = False
        rng.shuffle(chromosome)
        if not chromosome.any():
            chromosome[rng.integers(0, n_features)] = True
        population.append(chromosome)
    return population


def _score_fold(
    clf: Any,
    X_te: np.ndarray,
    y_te: np.ndarray,
    metric: str,
) -> float:
    if metric == "accuracy":
        return float(accuracy_score(y_te, _to_numpy_1d(clf.predict(X_te))))
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_te, _to_numpy_1d(clf.predict(X_te))))
    if metric in {"roc_auc", "roc_auc_ovr"}:
        if not hasattr(clf, "predict_proba"):
            pred = _to_numpy_1d(clf.predict(X_te))
            return float(roc_auc_score(y_te, pred))

        proba = _to_numpy(clf.predict_proba(X_te))
        if proba.ndim == 2 and proba.shape[1] > 2:
            labels = _to_numpy_1d(getattr(clf, "classes_", np.unique(y_te)))
            return float(
                roc_auc_score(
                    y_te,
                    proba,
                    multi_class="ovr",
                    labels=labels,
                )
            )
        pred = proba[:, 1] if proba.ndim == 2 else proba
        return float(roc_auc_score(y_te, pred))
    raise ValueError(
        "fitness_metric must be one of: accuracy, balanced_accuracy, roc_auc, roc_auc_ovr."
    )


def _stable_mask_hash(chromosome: np.ndarray) -> str:
    values = np.asarray(chromosome, dtype=np.uint8)
    return hashlib.sha256(values.tobytes()).hexdigest()


def fitness_components(
    *,
    base_score: float,
    feature_count: int,
    target_k: int,
    size_penalty_lambda: float,
    fitness_mode: str,
) -> dict[str, float | int]:
    """Return the scientific objective and legacy compatibility value."""

    if fitness_mode not in {
        "legacy_zero_truncated_linear",
        "untruncated_shifted_linear",
    }:
        raise ValueError(
            "ga.fitness_mode must be 'legacy_zero_truncated_linear' or "
            "'untruncated_shifted_linear'."
        )
    target_deviation = abs(int(feature_count) - int(target_k))
    penalty_amount = float(size_penalty_lambda) * target_deviation
    raw_objective = float(base_score) - penalty_amount
    legacy_fitness = max(0.0, raw_objective)
    ranking_score = (
        legacy_fitness
        if fitness_mode == "legacy_zero_truncated_linear"
        else raw_objective
    )
    return {
        "raw_objective": raw_objective,
        "legacy_truncated_fitness": legacy_fitness,
        "ranking_score": ranking_score,
        "target_deviation": target_deviation,
        "penalty_amount": penalty_amount,
    }


def shifted_sampling_weights(
    raw_objectives: np.ndarray,
    *,
    epsilon: float,
) -> tuple[np.ndarray, bool, str | None]:
    """Transform raw objectives for parent sampling without changing ranking."""

    raw = np.asarray(raw_objectives, dtype=float)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("raw_objectives must be a nonempty one-dimensional array.")
    if epsilon <= 0:
        raise ValueError("sampling epsilon must be positive.")
    finite = np.isfinite(raw)
    if not finite.any():
        return np.ones(raw.size, dtype=float), True, "no_finite_raw_objectives"
    finite_values = raw[finite]
    if np.max(finite_values) - np.min(finite_values) <= epsilon:
        return np.ones(raw.size, dtype=float), True, "all_raw_objectives_effectively_equal"
    weights = np.zeros(raw.size, dtype=float)
    weights[finite] = finite_values - np.min(finite_values) + epsilon
    total = float(np.sum(weights))
    if not np.isfinite(total):
        return np.ones(raw.size, dtype=float), True, "nonfinite_sampling_weight_total"
    if total <= epsilon:
        return np.ones(raw.size, dtype=float), True, "sampling_weight_total_near_zero"
    return weights, False, None


def _evaluate_chromosome_details(
    chromosome: np.ndarray,
    X_np: np.ndarray,
    y_np: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    target_k: int,
    config: GAConfig,
    seed: int,
    backend: _RFBackend,
) -> _ChromosomeEvaluation:
    col_idx = np.where(chromosome)[0]
    if col_idx.size == 0:
        raise ValueError("GA cannot evaluate an empty chromosome.")

    fold_scores = []
    for fold_id, (train_idx, test_idx) in enumerate(folds):
        clf = _make_rf(config, seed + fold_id, backend)
        X_tr = X_np[train_idx][:, col_idx]
        X_te = X_np[test_idx][:, col_idx]
        y_tr = y_np[train_idx]
        y_te = y_np[test_idx]
        clf.fit(X_tr, y_tr)
        fold_scores.append(_score_fold(clf, X_te, y_te, config.fitness_metric))

    base_score = float(np.mean(fold_scores))
    components = fitness_components(
        base_score=base_score,
        feature_count=int(col_idx.size),
        target_k=target_k,
        size_penalty_lambda=config.size_penalty_lambda,
        fitness_mode=config.fitness_mode,
    )
    return _ChromosomeEvaluation(
        base_score=base_score,
        raw_objective=float(components["raw_objective"]),
        legacy_truncated_fitness=float(components["legacy_truncated_fitness"]),
        ranking_score=float(components["ranking_score"]),
        feature_count=int(col_idx.size),
        target_deviation=int(components["target_deviation"]),
        penalty_amount=float(components["penalty_amount"]),
        stable_mask_hash=_stable_mask_hash(chromosome),
    )


def _evaluate_chromosome(
    chromosome: np.ndarray,
    X_np: np.ndarray,
    y_np: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    target_k: int,
    config: GAConfig,
    seed: int,
    backend: _RFBackend,
) -> tuple[float, float]:
    """Compatibility wrapper returning the mode-specific score and base score."""

    result = _evaluate_chromosome_details(
        chromosome,
        X_np,
        y_np,
        folds,
        target_k,
        config,
        seed,
        backend,
    )
    return result.ranking_score, result.base_score


def _evaluate_population_detailed(
    population: list[np.ndarray],
    X_np: np.ndarray,
    y_np: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    target_k: int,
    config: GAConfig,
    seed: int,
    backend: _RFBackend,
) -> _PopulationEvaluation:
    evaluations: list[_ChromosomeEvaluation] = []
    for idx, chromosome in enumerate(population):
        evaluation = _evaluate_chromosome_details(
            chromosome,
            X_np,
            y_np,
            folds,
            target_k,
            config,
            seed + idx * 1009,
            backend,
        )
        evaluations.append(evaluation)
    ranking_scores = np.asarray([item.ranking_score for item in evaluations], dtype=float)
    if config.fitness_mode == "legacy_zero_truncated_linear":
        order = list(np.argsort(ranking_scores)[::-1])
    else:
        order = sorted(
            range(len(population)),
            key=lambda idx: (
                -evaluations[idx].raw_objective,
                -evaluations[idx].base_score,
                evaluations[idx].target_deviation,
                evaluations[idx].feature_count,
                evaluations[idx].stable_mask_hash,
            ),
        )
    return _PopulationEvaluation(
        population=[population[idx] for idx in order],
        ranking_scores=ranking_scores[order],
        base_scores=np.asarray([item.base_score for item in evaluations], dtype=float)[order],
        raw_objectives=np.asarray([item.raw_objective for item in evaluations], dtype=float)[order],
        legacy_truncated_fitness=np.asarray(
            [item.legacy_truncated_fitness for item in evaluations], dtype=float
        )[order],
        feature_counts=np.asarray([item.feature_count for item in evaluations], dtype=int)[order],
        target_deviations=np.asarray(
            [item.target_deviation for item in evaluations], dtype=int
        )[order],
        penalty_amounts=np.asarray([item.penalty_amount for item in evaluations], dtype=float)[order],
        stable_mask_hashes=[evaluations[idx].stable_mask_hash for idx in order],
    )


def _evaluate_population(
    population: list[np.ndarray],
    X_np: np.ndarray,
    y_np: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    target_k: int,
    config: GAConfig,
    seed: int,
    backend: _RFBackend,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Compatibility wrapper preserving the historical three-value return."""

    result = _evaluate_population_detailed(
        population,
        X_np,
        y_np,
        folds,
        target_k,
        config,
        seed,
        backend,
    )
    return result.population, result.ranking_scores, result.base_scores


def _selection_indices_with_audit(
    ranking_scores: np.ndarray,
    raw_objectives: np.ndarray,
    k: int,
    rng: np.random.Generator,
    config: GAConfig,
) -> tuple[np.ndarray, np.ndarray, bool, str | None]:
    if config.fitness_mode == "legacy_zero_truncated_linear":
        weights = np.clip(np.asarray(ranking_scores, dtype=float), 0.0, None)
        if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
            weights = np.ones_like(weights, dtype=float)
            fallback = True
            reason = "legacy_nonfinite_or_zero_weight_total"
        else:
            fallback = False
            reason = None
    else:
        weights, fallback, reason = shifted_sampling_weights(
            raw_objectives,
            epsilon=config.sampling_epsilon,
        )
    probabilities = weights / float(weights.sum())
    indices = rng.choice(
        np.arange(len(ranking_scores)),
        size=k,
        replace=True,
        p=probabilities,
    )
    return indices, weights, fallback, reason


def _selection_indices(scores: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Historical roulette selection retained for compatibility tests."""

    config = GAConfig(fitness_mode="legacy_zero_truncated_linear")
    indices, _, _, _ = _selection_indices_with_audit(scores, scores, k, rng, config)
    return indices


def _crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    crossover_rate: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    c1 = p1.copy()
    c2 = p2.copy()
    if rng.random() < crossover_rate and len(p1) > 2:
        point = int(rng.integers(1, len(p1) - 1))
        c1 = np.concatenate([p1[:point], p2[point:]])
        c2 = np.concatenate([p2[:point], p1[point:]])
    return c1.astype(bool), c2.astype(bool)


def _mutate(
    chromosome: np.ndarray,
    mutation_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    mutated = chromosome.copy()
    flip_mask = rng.random(len(mutated)) < mutation_rate
    mutated[flip_mask] = ~mutated[flip_mask]
    if not mutated.any():
        mutated[int(rng.integers(0, len(mutated)))] = True
    return mutated


def _update_top_solutions(
    existing: list[GASolution],
    candidate: GASolution,
    top_k: int,
    fitness_mode: str = "legacy_zero_truncated_linear",
) -> list[GASolution]:
    by_run: dict[int, GASolution] = {item.run_id: item for item in existing}
    current = by_run.get(candidate.run_id)
    if fitness_mode == "legacy_zero_truncated_linear":
        replace_current = current is None or candidate.score > current.score
    else:
        candidate_key = (
            -float(candidate.raw_objective),
            -candidate.base_score,
            int(candidate.target_deviation),
            candidate.n_features,
            str(candidate.stable_mask_hash),
        )
        current_key = (
            float("inf"),
            float("inf"),
            2**31,
            2**31,
            "",
        ) if current is None else (
            -float(current.raw_objective),
            -current.base_score,
            int(current.target_deviation),
            current.n_features,
            str(current.stable_mask_hash),
        )
        replace_current = candidate_key < current_key
    if replace_current:
        by_run[candidate.run_id] = candidate
    if fitness_mode == "legacy_zero_truncated_linear":
        top = sorted(by_run.values(), key=lambda item: item.score, reverse=True)[:top_k]
    else:
        top = sorted(
            by_run.values(),
            key=lambda item: (
                -float(item.raw_objective),
                -item.base_score,
                int(item.target_deviation),
                item.n_features,
                str(item.stable_mask_hash),
            ),
        )[:top_k]
    for idx, solution in enumerate(top, start=1):
        solution.rank = idx
    return top


def _solution_rows(top_solutions: list[GASolution]) -> list[dict[str, Any]]:
    return [
        {
            "rank": solution.rank,
            "run_id": solution.run_id,
            "score": solution.score,
            "base_score": solution.base_score,
            "n_features": solution.n_features,
            "selected_features": "|".join(solution.selected_features),
            "raw_objective": solution.raw_objective,
            "legacy_truncated_fitness": solution.legacy_truncated_fitness,
            "target_deviation": solution.target_deviation,
            "penalty_amount": solution.penalty_amount,
            "stable_mask_hash": solution.stable_mask_hash,
        }
        for solution in top_solutions
    ]


def _replace_text(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _replace_csv(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _replace_npy(path: Path, values: np.ndarray, allow_pickle: bool = False) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("wb") as handle:
        np.save(handle, values, allow_pickle=allow_pickle)
    tmp.replace(path)


def _write_live_checkpoint(
    checkpoint_dir: Path,
    *,
    name: str,
    target_k: int,
    config: GAConfig,
    backend: _RFBackend,
    top_solutions: list[GASolution],
    history_rows: list[dict[str, Any]],
    problem_type: str,
    classes: np.ndarray,
    run_id: int,
    completed_generation: int,
    is_complete: bool,
    warnings: list[str],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows = _solution_rows(top_solutions)
    _replace_csv(checkpoint_dir / "history_live.csv", pd.DataFrame(history_rows))
    _replace_csv(checkpoint_dir / "top_solutions_live.csv", pd.DataFrame(rows))

    top_feature_sets = np.asarray(
        [np.asarray(item.selected_features, dtype=str) for item in top_solutions],
        dtype=object,
    )
    top_masks = np.asarray([item.mask for item in top_solutions], dtype=bool)
    _replace_npy(
        checkpoint_dir / "top_feature_sets_live.npy",
        top_feature_sets,
        allow_pickle=True,
    )
    _replace_npy(checkpoint_dir / "top_masks_live.npy", top_masks)
    _replace_npy(
        checkpoint_dir / "scores_live.npy",
        np.asarray([item.score for item in top_solutions], dtype=float),
    )

    best = top_solutions[0] if top_solutions else None
    if best is not None:
        _replace_npy(
            checkpoint_dir / "best_feature_set_live.npy",
            np.asarray(best.selected_features, dtype=str),
        )
        _replace_npy(checkpoint_dir / "best_mask_live.npy", best.mask)

    summary = {
        "status": "complete" if is_complete else "running",
        "name": name,
        "target_k": target_k,
        "current_run": run_id + 1,
        "n_runs": config.n_runs,
        "current_generation": completed_generation,
        "n_generations": config.n_generations,
        "completed_history_rows": len(history_rows),
        "top_solution_count": len(top_solutions),
        "best_score": best.score if best is not None else None,
        "best_base_score": best.base_score if best is not None else None,
        "best_n_features": best.n_features if best is not None else None,
        "problem_type": problem_type,
        "n_classes": int(len(classes)),
        "classes": classes.tolist(),
        "population_size": config.population_size,
        "fitness_metric": config.fitness_metric,
        "size_penalty_lambda": config.size_penalty_lambda,
        "fitness_mode": config.fitness_mode,
        "artifact_schema_version": "2.0",
        "warnings": warnings,
        "top_k": config.top_k,
        "progress_interval": config.progress_interval,
        "requested_backend": config.backend,
        "actual_backend": backend.name,
        "backend_fallback_reason": backend.fallback_reason,
    }
    _replace_text(
        checkpoint_dir / "checkpoint.json",
        json.dumps(summary, indent=2, default=_json_default),
    )


def run_genetic_rf(
    X: pd.DataFrame,
    y: pd.Series,
    target_k: int,
    name: str = "ga_rf",
    config: GAConfig | None = None,
) -> GeneticRFResult:
    """Run GA-RF on a fixed candidate feature space.

    target_k is supplied by RFECV. Users tune GA behavior through GAConfig, but
    the target feature count remains model-derived by design.
    """

    config = config or GAConfig()
    classes = unique_classes(y)
    problem_type = infer_problem_type(y)
    if target_k < 1 or target_k > X.shape[1]:
        raise ValueError("target_k must be between 1 and the number of candidate features.")
    if config.elitism_count < 1 or config.elitism_count >= config.population_size:
        raise ValueError("elitism_count must be >= 1 and smaller than population_size.")
    if config.fitness_mode not in {
        "legacy_zero_truncated_linear",
        "untruncated_shifted_linear",
    }:
        raise ValueError(
            "ga.fitness_mode must be 'legacy_zero_truncated_linear' or "
            "'untruncated_shifted_linear'."
        )
    if config.sampling_epsilon <= 0:
        raise ValueError("sampling_epsilon must be positive.")
    if not 0 <= config.legacy_zero_fraction_warning <= 1:
        raise ValueError("legacy_zero_fraction_warning must be between 0 and 1.")
    if config.minimum_unique_masks_warning < 1:
        raise ValueError("minimum_unique_masks_warning must be at least 1.")

    backend = _resolve_backend(config)
    if backend.name == "gpu":
        X_np = X.to_numpy(dtype=np.float32)
        y_np = y.to_numpy(dtype=np.int32)
    else:
        X_np = X.to_numpy()
        y_np = y.to_numpy()
    cv = StratifiedKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )
    folds = list(cv.split(X_np, y_np))

    top_solutions: list[GASolution] = []
    history_rows: list[dict[str, Any]] = []
    ga_warnings: list[str] = []
    feature_names = list(X.columns)
    progress_interval = max(1, int(config.progress_interval))
    checkpoint_dir = Path(config.checkpoint_dir) if config.checkpoint_dir else None

    for run_id in range(config.n_runs):
        if config.verbose:
            print(
                f"[GA-RF:{name}] run {run_id + 1}/{config.n_runs} started "
                f"(backend={backend.name}, target_k={target_k})",
                flush=True,
            )
        rng = np.random.default_rng(config.random_state + run_id)
        population = _initial_population(
            config.population_size,
            X.shape[1],
            config.initial_off_ratio,
            rng,
        )

        for generation in range(config.n_generations):
            evaluation = _evaluate_population_detailed(
                population,
                X_np,
                y_np,
                folds,
                target_k,
                config,
                config.random_state + run_id * 10000 + generation * 100,
                backend,
            )
            population = evaluation.population
            scores = evaluation.ranking_scores
            base_scores = evaluation.base_scores
            best_mask = population[0].copy()
            selected = [feature_names[idx] for idx in np.where(best_mask)[0]]
            candidate = GASolution(
                rank=0,
                run_id=run_id,
                score=float(scores[0]),
                base_score=float(base_scores[0]),
                n_features=int(best_mask.sum()),
                mask=best_mask,
                selected_features=selected,
                raw_objective=float(evaluation.raw_objectives[0]),
                legacy_truncated_fitness=float(
                    evaluation.legacy_truncated_fitness[0]
                ),
                target_deviation=int(evaluation.target_deviations[0]),
                penalty_amount=float(evaluation.penalty_amounts[0]),
                stable_mask_hash=evaluation.stable_mask_hashes[0],
            )
            top_solutions = _update_top_solutions(
                top_solutions,
                candidate,
                config.top_k,
                config.fitness_mode,
            )

            elites = [item.copy() for item in population[: config.elitism_count]]
            n_children_needed = config.population_size - config.elitism_count
            parent_indices, sampling_weights, uniform_fallback, fallback_reason = (
                _selection_indices_with_audit(
                    scores,
                    evaluation.raw_objectives,
                    n_children_needed,
                    rng,
                    config,
                )
            )
            parents = [population[idx] for idx in parent_indices]

            fraction_legacy_zero = float(
                np.mean(evaluation.legacy_truncated_fitness == 0.0)
            )
            number_unique_raw = int(np.unique(evaluation.raw_objectives).size)
            population_unique_masks = len(
                {item.astype(np.uint8).tobytes() for item in population}
            )
            warning_prefix = f"run={run_id} generation={generation}"
            generation_warnings: list[str] = []
            if fraction_legacy_zero > config.legacy_zero_fraction_warning:
                generation_warnings.append(
                    f"{warning_prefix}: fraction_legacy_zero={fraction_legacy_zero:.3f} "
                    f"exceeds {config.legacy_zero_fraction_warning:.3f}."
                )
            if fraction_legacy_zero == 1.0:
                generation_warnings.append(
                    f"{warning_prefix}: all legacy chromosome fitness values are zero."
                )
            if uniform_fallback:
                generation_warnings.append(
                    f"{warning_prefix}: uniform parent sampling fallback ({fallback_reason})."
                )
            if (
                config.target_deviation_warning_threshold is not None
                and int(evaluation.target_deviations[0])
                > config.target_deviation_warning_threshold
            ):
                generation_warnings.append(
                    f"{warning_prefix}: best target deviation "
                    f"{int(evaluation.target_deviations[0])} exceeds "
                    f"{config.target_deviation_warning_threshold}."
                )
            if population_unique_masks < config.minimum_unique_masks_warning:
                generation_warnings.append(
                    f"{warning_prefix}: population has {population_unique_masks} unique masks, "
                    f"below {config.minimum_unique_masks_warning}."
                )
            ga_warnings.extend(generation_warnings)
            history_rows.append(
                {
                    "run_id": run_id,
                    "generation": generation,
                    "best_score": float(scores[0]),
                    "best_base_score": float(base_scores[0]),
                    "best_n_features": int(best_mask.sum()),
                    "base_score": float(base_scores[0]),
                    "raw_objective": float(evaluation.raw_objectives[0]),
                    "legacy_truncated_fitness": float(
                        evaluation.legacy_truncated_fitness[0]
                    ),
                    "sampling_weight": float(sampling_weights[0]),
                    "feature_count": int(evaluation.feature_counts[0]),
                    "target_count": int(target_k),
                    "target_deviation": int(evaluation.target_deviations[0]),
                    "penalty_amount": float(evaluation.penalty_amounts[0]),
                    "fraction_legacy_zero": fraction_legacy_zero,
                    "number_unique_raw_objectives": number_unique_raw,
                    "uniform_sampling_fallback": bool(uniform_fallback),
                    "uniform_sampling_fallback_reason": fallback_reason,
                    "population_unique_masks": population_unique_masks,
                    "best_raw_objective": float(evaluation.raw_objectives[0]),
                    "median_raw_objective": float(
                        np.median(evaluation.raw_objectives)
                    ),
                    "best_base_score_audit": float(base_scores[0]),
                    "fitness_mode": config.fitness_mode,
                    "generation_warning_count": len(generation_warnings),
                }
            )
            completed_generation = generation + 1
            should_checkpoint = (
                checkpoint_dir is not None
                and (
                    completed_generation % progress_interval == 0
                    or completed_generation == config.n_generations
                )
            )
            if should_checkpoint:
                _write_live_checkpoint(
                    checkpoint_dir,
                    name=name,
                    target_k=target_k,
                    config=config,
                    backend=backend,
                    top_solutions=top_solutions,
                    history_rows=history_rows,
                    problem_type=problem_type,
                    classes=classes,
                    run_id=run_id,
                    completed_generation=completed_generation,
                    is_complete=(
                        run_id == config.n_runs - 1
                        and completed_generation == config.n_generations
                    ),
                    warnings=ga_warnings,
                )
            if config.verbose and (
                completed_generation % progress_interval == 0
                or completed_generation == config.n_generations
            ):
                print(
                    f"[GA-RF:{name}] run {run_id + 1}/{config.n_runs} "
                    f"gen {completed_generation}/{config.n_generations} "
                    f"score={float(scores[0]):.4f} "
                    f"base={float(base_scores[0]):.4f} "
                    f"n_features={int(best_mask.sum())}",
                    flush=True,
                )

            children: list[np.ndarray] = []
            for idx in range(0, len(parents), 2):
                p1 = parents[idx]
                p2 = parents[(idx + 1) % len(parents)]
                for child in _crossover(p1, p2, config.crossover_rate, rng):
                    children.append(_mutate(child, config.mutation_rate, rng))

            population = elites
            for child in children:
                if len(population) < config.population_size:
                    population.append(child)
        if config.verbose and top_solutions:
            run_best = next(
                (solution for solution in top_solutions if solution.run_id == run_id),
                None,
            )
            if run_best is not None:
                print(
                    f"[GA-RF:{name}] run {run_id + 1}/{config.n_runs} complete "
                    f"best_score={run_best.score:.4f} "
                    f"best_base={run_best.base_score:.4f} "
                    f"n_features={run_best.n_features}",
                    flush=True,
                )
            else:
                print(
                    f"[GA-RF:{name}] run {run_id + 1}/{config.n_runs} complete",
                    flush=True,
                )

    return GeneticRFResult(
        name=name,
        target_k=target_k,
        top_solutions=top_solutions,
        history=pd.DataFrame(history_rows),
        metadata={
            "population_size": config.population_size,
            "n_generations": config.n_generations,
            "n_runs": config.n_runs,
            "crossover_rate": config.crossover_rate,
            "mutation_rate": config.mutation_rate,
            "elitism_count": config.elitism_count,
            "initial_off_ratio": config.initial_off_ratio,
            "size_penalty_lambda": config.size_penalty_lambda,
            "fitness_mode": config.fitness_mode,
            "sampling_epsilon": config.sampling_epsilon,
            "legacy_zero_fraction_warning": config.legacy_zero_fraction_warning,
            "target_deviation_warning_threshold": (
                config.target_deviation_warning_threshold
            ),
            "minimum_unique_masks_warning": config.minimum_unique_masks_warning,
            "artifact_schema_version": "2.0",
            "top_k": config.top_k,
            "cv_folds": config.cv_folds,
            "fitness_metric": config.fitness_metric,
            "problem_type": problem_type,
            "n_classes": int(len(classes)),
            "classes": classes.tolist(),
            "n_jobs": config.n_jobs,
            "verbose": config.verbose,
            "progress_interval": config.progress_interval,
            "checkpoint_dir": config.checkpoint_dir,
            "requested_backend": config.backend,
            "actual_backend": backend.name,
            "backend_fallback_reason": backend.fallback_reason,
            "rf_params": config.rf_params,
            "gpu_rf_params": config.gpu_rf_params,
        },
        warnings=ga_warnings,
    )
