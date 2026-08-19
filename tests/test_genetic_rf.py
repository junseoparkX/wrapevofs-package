import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

import wrapevofs.selectors.genetic_rf as genetic_rf
from wrapevofs.config import GAConfig
from wrapevofs.selectors.genetic_rf import (
    GASolution,
    _mutate,
    _update_top_solutions,
    fitness_components,
    run_genetic_rf,
    shifted_sampling_weights,
)
from wrapevofs._version import ARTIFACT_SCHEMA_VERSION


def test_gpu_request_fails_actionably_and_auto_falls_back(monkeypatch):
    monkeypatch.setattr(
        genetic_rf,
        "_load_cuml_rf",
        lambda: (None, "synthetic CUDA unavailability"),
    )

    with pytest.raises(
        ImportError,
        match="Install RAPIDS cuML.*set ga.backend='cpu'",
    ):
        genetic_rf._resolve_backend(GAConfig(backend="gpu"))

    backend = genetic_rf._resolve_backend(GAConfig(backend="auto"))
    assert backend.name == "cpu"
    assert backend.model_cls is genetic_rf.RandomForestClassifier
    assert backend.fallback_reason == "synthetic CUDA unavailability"


def test_untruncated_fitness_preserves_negative_objective_for_ranking():
    components = fitness_components(
        base_score=0.60,
        feature_count=20,
        target_k=3,
        size_penalty_lambda=0.05,
        fitness_mode="untruncated_shifted_linear",
    )

    assert components["raw_objective"] == pytest.approx(-0.25)
    assert components["ranking_score"] == pytest.approx(-0.25)
    assert components["legacy_truncated_fitness"] == 0.0
    assert components["target_deviation"] == 17
    assert components["penalty_amount"] == pytest.approx(0.85)


def test_shifted_sampling_weights_preserve_order_and_report_equal_fallback():
    raw = np.asarray([-0.8, -0.3, 0.2])
    weights, fallback, reason = shifted_sampling_weights(raw, epsilon=1e-12)

    assert not fallback
    assert reason is None
    assert np.array_equal(np.argsort(raw), np.argsort(weights))
    equal, fallback, reason = shifted_sampling_weights(
        np.asarray([-0.4, -0.4, -0.4]),
        epsilon=1e-12,
    )
    assert fallback
    assert reason == "all_raw_objectives_effectively_equal"
    assert np.array_equal(equal, np.ones(3))


def test_shifted_sampling_weights_handle_nonfinite_objectives_actionably():
    all_bad, fallback, reason = shifted_sampling_weights(
        np.asarray([np.nan, np.inf, -np.inf]), epsilon=1e-12
    )
    assert fallback
    assert reason == "no_finite_raw_objectives"
    assert np.array_equal(all_bad, np.ones(3))

    mixed, fallback, reason = shifted_sampling_weights(
        np.asarray([-1.0, 0.5, np.nan]), epsilon=1e-12
    )
    assert not fallback
    assert reason is None
    assert mixed[1] > mixed[0] > mixed[2]


def _solution(run_id, features, raw=-0.2, base=0.8, stable_hash="a"):
    mask = np.asarray([feature in features for feature in ["a", "b", "c"]])
    return GASolution(
        rank=0,
        run_id=run_id,
        score=raw,
        base_score=base,
        n_features=len(features),
        mask=mask,
        selected_features=list(features),
        raw_objective=raw,
        legacy_truncated_fitness=max(0.0, raw),
        target_deviation=1,
        penalty_amount=base - raw,
        stable_mask_hash=stable_hash,
    )


def test_untruncated_ties_are_deterministic_and_prefer_smaller_mask():
    larger = _solution(2, ["a", "b"], stable_hash="b")
    smaller = _solution(1, ["a"], stable_hash="c")
    ranked = _update_top_solutions(
        [larger],
        smaller,
        top_k=2,
        fitness_mode="untruncated_shifted_linear",
    )

    assert [item.run_id for item in ranked] == [1, 2]


def test_recommended_solution_ties_follow_raw_base_deviation_count_and_hash():
    lower_raw = _solution(0, ["a"], raw=-0.3, base=0.99, stable_hash="0")
    higher_raw = _solution(1, ["a", "b"], raw=-0.2, base=0.70, stable_hash="f")
    higher_base = _solution(2, ["a", "b"], raw=-0.2, base=0.80, stable_hash="f")
    lower_deviation = _solution(3, ["a", "b"], raw=-0.2, base=0.80, stable_hash="e")
    higher_base.target_deviation = 2
    lower_deviation.target_deviation = 1
    earlier_hash = _solution(4, ["a", "b"], raw=-0.2, base=0.80, stable_hash="a")
    earlier_hash.target_deviation = 1

    ranked: list[GASolution] = []
    for item in [lower_raw, higher_raw, higher_base, lower_deviation, earlier_hash]:
        ranked = _update_top_solutions(
            ranked, item, top_k=5, fitness_mode="untruncated_shifted_linear"
        )

    assert [item.run_id for item in ranked] == [4, 3, 2, 1, 0]


def test_mutation_never_returns_an_empty_mask():
    rng = np.random.default_rng(3)
    mutated = _mutate(np.asarray([True]), mutation_rate=1.0, rng=rng)
    assert mutated.any()


def test_genetic_rf_uses_model_supplied_target_k():
    X, y = make_classification(
        n_samples=50,
        n_features=6,
        n_informative=3,
        random_state=42,
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    result = run_genetic_rf(
        X,
        pd.Series(y),
        target_k=3,
        config=GAConfig(
            backend="cpu",
            population_size=8,
            n_generations=2,
            n_runs=2,
            top_k=2,
            cv_folds=2,
            rf_params={"n_estimators": 5, "max_depth": 3},
        ),
    )

    assert result.target_k == 3
    assert result.metadata["requested_backend"] == "cpu"
    assert result.metadata["actual_backend"] == "cpu"
    assert result.metadata["problem_type"] == "binary"
    assert 1 <= len(result.top_solutions) <= 2
    assert result.best_solution.n_features >= 1
    assert set(result.best_solution.selected_features).issubset(set(X.columns))


def test_genetic_rf_supports_multiclass_accuracy():
    X, y = make_classification(
        n_samples=75,
        n_features=7,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        random_state=9,
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(7)])

    result = run_genetic_rf(
        X,
        pd.Series(y),
        target_k=4,
        config=GAConfig(
            backend="cpu",
            fitness_metric="accuracy",
            population_size=8,
            n_generations=2,
            n_runs=1,
            top_k=2,
            cv_folds=3,
            rf_params={"n_estimators": 8, "max_depth": 4},
        ),
    )

    assert result.target_k == 4
    assert result.metadata["problem_type"] == "multiclass"
    assert result.metadata["n_classes"] == 3
    assert result.best_solution.score >= 0.0
    assert result.best_solution.n_features >= 1


def test_genetic_rf_supports_multiclass_roc_auc_ovr():
    X, y = make_classification(
        n_samples=90,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        random_state=13,
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])

    result = run_genetic_rf(
        X,
        pd.Series(y),
        target_k=4,
        config=GAConfig(
            backend="cpu",
            fitness_metric="roc_auc_ovr",
            population_size=8,
            n_generations=1,
            n_runs=1,
            top_k=1,
            cv_folds=3,
            rf_params={"n_estimators": 8, "max_depth": 4},
        ),
    )

    assert result.metadata["fitness_metric"] == "roc_auc_ovr"
    assert result.metadata["problem_type"] == "multiclass"
    assert result.best_solution.base_score >= 0.0


def test_genetic_rf_writes_live_checkpoints(tmp_path):
    X, y = make_classification(
        n_samples=50,
        n_features=6,
        n_informative=3,
        random_state=21,
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])

    result = run_genetic_rf(
        X,
        pd.Series(y),
        target_k=3,
        name="svm_l1_ga_rf",
        config=GAConfig(
            backend="cpu",
            population_size=8,
            n_generations=2,
            n_runs=1,
            top_k=2,
            cv_folds=2,
            progress_interval=1,
            checkpoint_dir=str(tmp_path),
            rf_params={"n_estimators": 5, "max_depth": 3},
        ),
    )

    assert result.metadata["checkpoint_dir"] == str(tmp_path)
    assert (tmp_path / "checkpoint.json").exists()
    assert (tmp_path / "history_live.csv").exists()
    assert (tmp_path / "top_solutions_live.csv").exists()
    assert (tmp_path / "top_feature_sets_live.npy").exists()
    assert (tmp_path / "best_feature_set_live.npy").exists()
    assert (tmp_path / "resume_state.npz").exists()
    assert not list(tmp_path.glob("*.tmp"))

    live_sets = np.load(tmp_path / "top_feature_sets_live.npy", allow_pickle=True)
    assert len(live_sets) == len(result.top_solutions)


def test_implicit_and_explicit_legacy_modes_are_regression_identical():
    X, y = make_classification(
        n_samples=48,
        n_features=6,
        n_informative=3,
        random_state=31,
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    common = {
        "backend": "cpu",
        "population_size": 6,
        "n_generations": 2,
        "n_runs": 2,
        "top_k": 2,
        "cv_folds": 2,
        "random_state": 11,
        "rf_params": {"n_estimators": 4, "max_depth": 3},
    }
    implicit = run_genetic_rf(X, pd.Series(y), 3, config=GAConfig(**common))
    explicit = run_genetic_rf(
        X,
        pd.Series(y),
        3,
        config=GAConfig(
            **common,
            fitness_mode="legacy_zero_truncated_linear",
        ),
    )

    assert [item.selected_features for item in implicit.top_solutions] == [
        item.selected_features for item in explicit.top_solutions
    ]
    assert [item.score for item in implicit.top_solutions] == pytest.approx(
        [item.score for item in explicit.top_solutions]
    )
    pd.testing.assert_frame_equal(implicit.history, explicit.history)


def test_untruncated_history_exposes_required_audit_fields():
    X, y = make_classification(
        n_samples=44,
        n_features=5,
        n_informative=3,
        random_state=17,
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    result = run_genetic_rf(
        X,
        pd.Series(y),
        2,
        config=GAConfig(
            backend="cpu",
            fitness_mode="untruncated_shifted_linear",
            population_size=6,
            n_generations=1,
            n_runs=1,
            top_k=1,
            cv_folds=2,
            random_state=19,
            rf_params={"n_estimators": 3, "max_depth": 2},
        ),
    )

    required = {
        "base_score",
        "raw_objective",
        "legacy_truncated_fitness",
        "sampling_weight",
        "feature_count",
        "target_count",
        "target_deviation",
        "penalty_amount",
        "fraction_legacy_zero",
        "number_unique_raw_objectives",
        "uniform_sampling_fallback",
    }
    assert required.issubset(result.history.columns)
    assert result.metadata["fitness_mode"] == "untruncated_shifted_linear"
    assert result.metadata["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION


def test_zero_legacy_diagnostic_does_not_control_recommended_sampling(monkeypatch):
    import wrapevofs.selectors.genetic_rf as genetic_rf

    original = genetic_rf._evaluate_population_detailed

    def force_distinct_negative_raw(*args, **kwargs):
        result = original(*args, **kwargs)
        raw = -np.arange(1, len(result.population) + 1, dtype=float)
        result.raw_objectives = raw
        result.ranking_scores = raw.copy()
        result.legacy_truncated_fitness = np.zeros_like(raw)
        return result

    monkeypatch.setattr(
        genetic_rf, "_evaluate_population_detailed", force_distinct_negative_raw
    )
    X, y = make_classification(
        n_samples=36, n_features=5, n_informative=3, random_state=47
    )
    result = run_genetic_rf(
        pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]),
        pd.Series(y),
        2,
        config=GAConfig(
            backend="cpu",
            fitness_mode="untruncated_shifted_linear",
            population_size=4,
            n_generations=1,
            n_runs=1,
            top_k=1,
            cv_folds=2,
            rf_params={"n_estimators": 2, "max_depth": 2},
        ),
    )
    row = result.history.iloc[0]
    assert row["fraction_legacy_zero"] == 1.0
    assert not bool(row["uniform_sampling_fallback"])


def test_actual_uniform_fallback_is_logged(monkeypatch):
    import wrapevofs.selectors.genetic_rf as genetic_rf

    original = genetic_rf._evaluate_population_detailed

    def force_equal_raw(*args, **kwargs):
        result = original(*args, **kwargs)
        raw = np.full(len(result.population), -0.5)
        result.raw_objectives = raw
        result.ranking_scores = raw.copy()
        result.legacy_truncated_fitness = np.zeros_like(raw)
        return result

    monkeypatch.setattr(genetic_rf, "_evaluate_population_detailed", force_equal_raw)
    X, y = make_classification(
        n_samples=36, n_features=5, n_informative=3, random_state=53
    )
    result = run_genetic_rf(
        pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]),
        pd.Series(y),
        2,
        config=GAConfig(
            backend="cpu",
            fitness_mode="untruncated_shifted_linear",
            population_size=4,
            n_generations=1,
            n_runs=1,
            top_k=1,
            cv_folds=2,
            rf_params={"n_estimators": 2, "max_depth": 2},
        ),
    )
    row = result.history.iloc[0]
    assert bool(row["uniform_sampling_fallback"])
    assert row["uniform_sampling_fallback_reason"] == (
        "all_raw_objectives_effectively_equal"
    )
    assert any("uniform parent sampling fallback" in item for item in result.warnings)
