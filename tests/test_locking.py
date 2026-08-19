import itertools
import json
import random
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from wrapevofs import LockingCandidate, LockingConfig, lock_representative_run


def candidate(run_id, score, features, folds=None, universe=None):
    return LockingCandidate(
        run_id=run_id,
        locking_score=score,
        features=features,
        fold_locking_scores=folds,
        seed=40 + run_id,
        candidate_universe=universe,
    )


def base_candidates():
    return [
        candidate(1, 0.900, ["a", "b"], [0.88, 0.92, 0.90, 0.91, 0.89]),
        candidate(2, 0.895, ["a", "b", "c"], [0.88, 0.91, 0.89, 0.90, 0.895]),
        candidate(3, 0.870, ["x", "y"], [0.85, 0.88, 0.86, 0.87, 0.89]),
    ]


def test_absolute_regret_pool_and_selection():
    result = lock_representative_run(
        base_candidates(),
        LockingConfig(
            enabled=True,
            strategy="regret_constrained_medoid",
            tolerance_mode="absolute",
            regret_tolerance=0.01,
            minimum_pool_size=1,
            fallback_rule="strict_eligible_only",
        ),
    )

    assert result.metadata["eligible_run_ids"] == [1, 2]
    assert result.selected_run_id == 1
    assert result.metadata["selected_within_declared_tolerance"] is True
    assert result.metadata["selected_absolute_regret"] <= 0.01
    assert result.metadata["fallback_expansion_occurred"] is False
    assert result.candidate_audit.set_index("run_id").loc[2, "absolute_regret"] == pytest.approx(0.005)


def test_relative_regret_pool():
    result = lock_representative_run(
        base_candidates(),
        LockingConfig(
            enabled=True,
            strategy="regret_constrained_medoid",
            tolerance_mode="relative",
            regret_tolerance=0.01,
            minimum_pool_size=1,
        ),
    )

    assert result.metadata["eligible_run_ids"] == [1, 2]


def test_best_run_se_scaled_uses_best_run_fold_scores():
    result = lock_representative_run(
        base_candidates(),
        LockingConfig(
            enabled=True,
            strategy="regret_constrained_medoid",
            tolerance_mode="best_run_se_scaled",
            minimum_pool_size=1,
            fallback_rule="strict_eligible_only",
        ),
    )

    expected_se = np.std([0.88, 0.92, 0.90, 0.91, 0.89], ddof=1) / np.sqrt(5)
    assert result.metadata["eligibility_threshold"] == pytest.approx(expected_se)
    assert result.metadata["eligible_run_ids"] == [1, 2]


def test_one_eligible_candidate_is_supported():
    result = lock_representative_run(
        base_candidates(),
        LockingConfig(
            enabled=True,
            strategy="regret_constrained_medoid",
            tolerance_mode="absolute",
            regret_tolerance=0.0,
            minimum_pool_size=1,
        ),
    )

    assert result.selected_run_id == 1
    assert result.metadata["eligible_run_ids"] == [1]
    assert np.isnan(result.candidate_audit.set_index("run_id").loc[1, "mean_jaccard"])


def test_all_candidates_can_be_eligible():
    result = lock_representative_run(
        base_candidates(),
        LockingConfig(
            enabled=True,
            strategy="regret_constrained_medoid",
            tolerance_mode="absolute",
            regret_tolerance=1.0,
            minimum_pool_size=1,
        ),
    )

    assert result.metadata["eligible_run_ids"] == [1, 2, 3]


def test_strict_regret_lock_rejects_minimum_pool_expansion():
    with pytest.raises(ValueError, match="requires minimum_pool_size=1"):
        lock_representative_run(
            base_candidates(),
            LockingConfig(
                enabled=True,
                strategy="regret_constrained_medoid",
                tolerance_mode="absolute",
                regret_tolerance=0.0,
                minimum_pool_size=2,
                fallback_rule="expand_by_score",
            ),
        )


def test_strict_regret_lock_rejects_relaxing_fallback_rule():
    with pytest.raises(ValueError, match="strict_eligible_only"):
        lock_representative_run(
            base_candidates(),
            LockingConfig(
                enabled=True,
                strategy="regret_constrained_medoid",
                tolerance_mode="absolute",
                regret_tolerance=0.0,
                minimum_pool_size=1,
                fallback_rule="expand_by_score",
            ),
        )


def test_equal_score_and_jaccard_tie_prefers_smaller_feature_count():
    result = lock_representative_run(
        [candidate(1, 0.9, ["a", "b", "c"]), candidate(2, 0.9, ["a", "b"])],
        LockingConfig(
            enabled=True,
            strategy="regret_constrained_medoid",
            tolerance_mode="absolute",
            regret_tolerance=0.0,
            minimum_pool_size=1,
            fallback_rule="strict_eligible_only",
        ),
    )

    assert result.selected_run_id == 2


def test_final_scientific_tie_break_is_stable_mask_hash():
    result = lock_representative_run(
        [
            candidate(2, 0.9, ["a"], universe=["a", "b"]),
            candidate(1, 0.9, ["b"], universe=["a", "b"]),
        ],
        LockingConfig(
            enabled=True,
            strategy="regret_constrained_medoid",
            tolerance_mode="absolute",
            regret_tolerance=0.0,
            minimum_pool_size=1,
            fallback_rule="strict_eligible_only",
        ),
    )

    audit = result.candidate_audit.set_index("run_id")
    selected_hash = audit.loc[result.selected_run_id, "stable_mask_hash"]
    assert selected_hash == audit["stable_mask_hash"].min()
    assert result.metadata["tie_break_path"].endswith("stable_mask_hash")


def test_empty_candidate_bank_is_rejected():
    with pytest.raises(ValueError, match="At least one locking candidate"):
        lock_representative_run([])


def test_negative_delta_is_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        lock_representative_run(
            [candidate(1, 0.9, ["a"])],
            LockingConfig(
                enabled=True,
                strategy="regret_constrained_medoid",
                regret_tolerance=-0.01,
            ),
        )


def test_singleton_candidate_bank_is_selected():
    result = lock_representative_run(
        [candidate(7, 0.75, ["b", "a"], universe=["a", "b", "c"])],
        LockingConfig(enabled=True, strategy="regret_constrained_medoid"),
    )

    assert result.selected_run_id == 7
    assert result.selected_features == ["a", "b"]
    assert result.metadata["eligible_run_ids"] == [7]
    assert result.metadata["selected_absolute_regret"] == 0.0


def test_unique_best_candidate_at_zero_tolerance_is_selected():
    result = lock_representative_run(
        [candidate(1, 0.8, ["a"]), candidate(2, 0.9, ["b"])],
        LockingConfig(
            enabled=True,
            strategy="regret_constrained_medoid",
            regret_tolerance=0.0,
        ),
    )

    assert result.metadata["eligible_run_ids"] == [2]
    assert result.selected_features == ["b"]


def test_exact_best_score_ties_use_zero_regret_pool_medoid():
    universe = ["a", "b"]
    result = lock_representative_run(
        [
            candidate(1, 0.9, ["a"], universe=universe),
            candidate(2, 0.9, ["a", "b"], universe=universe),
            candidate(3, 0.9, ["b"], universe=universe),
            candidate(4, 0.8, ["a"], universe=universe),
        ],
        LockingConfig(
            enabled=True,
            strategy="regret_constrained_medoid",
            regret_tolerance=0.0,
        ),
    )

    assert result.metadata["eligible_run_ids"] == [1, 2, 3]
    assert result.selected_features == ["a", "b"]


def test_absolute_pool_does_not_apply_hidden_epsilon_relaxation():
    result = lock_representative_run(
        [candidate(1, 1.0, ["a"]), candidate(2, 0.9899999999995, ["b"])],
        LockingConfig(
            enabled=True,
            strategy="regret_constrained_medoid",
            regret_tolerance=0.01,
            epsilon=1e-12,
        ),
    )

    assert result.metadata["eligible_run_ids"] == [1]
    assert result.metadata["selected_absolute_regret"] <= 0.01


def test_lower_is_better_orientation_is_rejected():
    with pytest.raises(ValueError, match="larger_is_better"):
        lock_representative_run(
            [candidate(1, 0.3, ["a"])],
            LockingConfig(
                enabled=True,
                strategy="regret_constrained_medoid",
                metric_orientation="lower_is_better",
            ),
        )


def test_inconsistent_candidate_universes_are_rejected():
    with pytest.raises(ValueError, match="same canonical candidate_universe"):
        lock_representative_run(
            [
                candidate(1, 0.9, ["a"], universe=["a", "b"]),
                candidate(2, 0.8, ["b"], universe=["b", "a"]),
            ]
        )


def test_feature_order_is_canonicalized_before_hashing():
    config = LockingConfig(enabled=True, strategy="regret_constrained_medoid")
    first = lock_representative_run(
        [candidate(1, 0.9, ["c", "a"], universe=["a", "b", "c"])],
        config,
    )
    second = lock_representative_run(
        [candidate(1, 0.9, ["a", "c"], universe=["a", "b", "c"])],
        config,
    )

    assert first.selected_features == second.selected_features == ["a", "c"]
    assert (
        first.metadata["selected_stable_mask_hash"]
        == second.metadata["selected_stable_mask_hash"]
    )


def test_duplicate_masks_are_retained_as_multiple_voting_candidates():
    universe = ["a", "b"]
    result = lock_representative_run(
        [
            candidate(4, 0.9, ["a"], universe=universe),
            candidate(3, 0.9, ["a"], universe=universe),
            candidate(2, 0.9, ["a", "b"], universe=universe),
            candidate(1, 0.9, ["b"], universe=universe),
        ],
        LockingConfig(
            enabled=True,
            strategy="regret_constrained_medoid",
            regret_tolerance=0.0,
        ),
    )

    assert result.selected_features == ["a"]
    assert result.metadata["duplicate_mask_policy"] == "retain_multiplicity_as_voting_candidates"
    selected_rows = result.candidate_audit[result.candidate_audit["selected_feature_set"]]
    assert set(selected_rows["run_id"]) == {3, 4}
    assert set(selected_rows["duplicate_mask_multiplicity"]) == {2}


def test_identical_masks_and_scores_have_one_provenance_record_but_one_feature_set():
    universe = ["a", "b"]
    result = lock_representative_run(
        [
            candidate(9, 0.9, ["a"], universe=universe),
            candidate(2, 0.9, ["a"], universe=universe),
        ],
        LockingConfig(enabled=True, strategy="regret_constrained_medoid"),
    )

    assert result.selected_run_id == 2
    assert result.metadata["selected_source_run_ids"] == [2, 9]
    assert result.candidate_audit["selected"].sum() == 1
    assert result.candidate_audit["selected_feature_set"].sum() == 2


def test_stable_mask_hash_is_reproducible_across_python_processes():
    code = (
        "from wrapevofs import LockingCandidate, LockingConfig, lock_representative_run;"
        "c=LockingCandidate(1,['c','a'],0.9,candidate_universe=['a','b','c']);"
        "print(lock_representative_run([c],LockingConfig(enabled=True,"
        "strategy='regret_constrained_medoid')).metadata['selected_stable_mask_hash'])"
    )
    observed = [
        subprocess.check_output([sys.executable, "-c", code], text=True).strip()
        for _ in range(2)
    ]

    assert observed[0] == observed[1]


def test_candidate_record_permutations_preserve_feature_set_and_audit():
    universe = ["a", "b", "c"]
    candidates = [
        candidate(30, 0.90, ["a"], universe=universe),
        candidate(10, 0.90, ["a", "b"], universe=universe),
        candidate(20, 0.90, ["b"], universe=universe),
    ]
    config = LockingConfig(
        enabled=True,
        strategy="regret_constrained_medoid",
        regret_tolerance=0.0,
    )
    results = [lock_representative_run(list(order), config) for order in itertools.permutations(candidates)]
    audit_csv = [result.candidate_audit.to_csv(index=False) for result in results]
    pairwise_csv = [result.pairwise_jaccard.to_csv(index=False) for result in results]

    assert {tuple(result.selected_features) for result in results} == {("a", "b")}
    assert len(set(audit_csv)) == 1
    assert len(set(pairwise_csv)) == 1


@pytest.mark.parametrize("seed", range(25))
def test_generated_banks_always_select_within_absolute_regret(seed):
    rng = random.Random(seed)
    universe = [f"f{i}" for i in range(6)]
    candidates = []
    for run_id in range(1, 7):
        features = [feature for feature in universe if rng.random() < 0.45]
        if not features:
            features = [universe[rng.randrange(len(universe))]]
        candidates.append(
            candidate(run_id, rng.uniform(-1.0, 1.0), features, universe=universe)
        )
    delta = rng.uniform(0.0, 0.5)
    result = lock_representative_run(
        candidates,
        LockingConfig(
            enabled=True,
            strategy="regret_constrained_medoid",
            regret_tolerance=delta,
        ),
    )

    assert result.metadata["selected_absolute_regret"] <= delta
    assert result.metadata["selected_within_declared_tolerance"] is True


@pytest.mark.parametrize("seed", range(12))
def test_generated_bank_permutations_are_invariant_and_repeated_runs_deterministic(seed):
    rng = random.Random(seed)
    universe = [f"f{i}" for i in range(5)]
    candidates = []
    for run_id in range(1, 6):
        features = [feature for feature in universe if rng.random() < 0.5] or [universe[0]]
        candidates.append(
            candidate(run_id, round(rng.uniform(0.0, 1.0), 2), features, universe=universe)
        )
    config = LockingConfig(
        enabled=True,
        strategy="regret_constrained_medoid",
        regret_tolerance=0.2,
    )
    baseline = lock_representative_run(candidates, config)
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    repeat = lock_representative_run(shuffled, config)

    assert repeat.selected_features == baseline.selected_features
    pd.testing.assert_frame_equal(repeat.candidate_audit, baseline.candidate_audit)
    pd.testing.assert_frame_equal(repeat.pairwise_jaccard, baseline.pairwise_jaccard)
    assert repeat.metadata == baseline.metadata


def test_serialization_reload_and_deterministic_audit_reproduction(tmp_path):
    config = LockingConfig(
        enabled=True,
        strategy="regret_constrained_medoid",
        regret_tolerance=0.01,
    )
    first = lock_representative_run(base_candidates(), config)
    second = lock_representative_run(base_candidates(), config)
    audit_path = tmp_path / "locking_candidate_audit.csv"
    pairwise_path = tmp_path / "pairwise_jaccard.csv"
    summary_path = tmp_path / "summary.json"
    first.candidate_audit.to_csv(audit_path, index=False)
    first.pairwise_jaccard.to_csv(pairwise_path, index=False)
    summary_path.write_text(json.dumps(first.metadata, sort_keys=True), encoding="utf-8")

    assert first.candidate_audit.to_csv(index=False) == second.candidate_audit.to_csv(index=False)
    assert first.pairwise_jaccard.to_csv(index=False) == second.pairwise_jaccard.to_csv(index=False)
    reloaded_audit = pd.read_csv(audit_path)
    reloaded_pairwise = pd.read_csv(pairwise_path)
    reloaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert reloaded_audit.loc[reloaded_audit["selected"], "stable_mask_hash"].item() == first.metadata["selected_stable_mask_hash"]
    assert len(reloaded_pairwise) == len(first.pairwise_jaccard)
    assert reloaded_summary["selected_stable_mask_hash"] == first.metadata["selected_stable_mask_hash"]


def test_legacy_top_k_mode_is_explicit_and_deterministic():
    candidates = [
        candidate(1, 0.90, ["a", "b"]),
        candidate(2, 0.89, ["a", "b", "c"]),
        candidate(3, 0.88, ["b", "c"]),
        candidate(4, 0.87, ["x", "y"]),
    ]
    implicit = lock_representative_run(candidates, LockingConfig(enabled=True))
    explicit = lock_representative_run(
        candidates,
        LockingConfig(enabled=True, strategy="top_k_jaccard_medoid", top_k=3),
    )

    assert implicit.selected_run_id == explicit.selected_run_id == 2
    assert implicit.selected_features == explicit.selected_features
    assert len(implicit.selected_features) == len(explicit.selected_features) == 3
    assert implicit.metadata["eligible_run_ids"] == [1, 2, 3]
    implicit_audit = implicit.candidate_audit.set_index("run_id")
    explicit_audit = explicit.candidate_audit.set_index("run_id")
    assert implicit_audit.loc[2, "locking_score"] == explicit_audit.loc[2, "locking_score"]
    assert implicit_audit.loc[2, "feature_count"] == explicit_audit.loc[2, "feature_count"]


def test_nonfinite_scores_are_rejected():
    with pytest.raises(ValueError, match="nonfinite locking score"):
        lock_representative_run([candidate(1, np.nan, ["a"])])


def test_missing_fold_scores_fail_actionably_for_best_run_se_scaled():
    with pytest.raises(ValueError, match="requires fold_locking_scores"):
        lock_representative_run(
            [candidate(1, 0.9, ["a"]), candidate(2, 0.89, ["b"])],
            LockingConfig(
                enabled=True,
                strategy="regret_constrained_medoid",
                tolerance_mode="best_run_se_scaled",
                minimum_pool_size=1,
                fallback_rule="strict_eligible_only",
            ),
        )


def test_empty_feature_masks_are_rejected():
    with pytest.raises(ValueError, match="empty feature mask"):
        lock_representative_run([candidate(1, 0.9, [])])


def test_held_out_arguments_are_not_accepted():
    with pytest.raises(TypeError):
        lock_representative_run(
            base_candidates(),
            LockingConfig(enabled=True),
            X_test=np.ones((2, 2)),
            y_test=np.ones(2),
        )


def test_complete_audit_schema_is_exported():
    result = lock_representative_run(
        base_candidates(),
        LockingConfig(
            enabled=True,
            strategy="regret_constrained_medoid",
            tolerance_mode="absolute",
            regret_tolerance=0.01,
            minimum_pool_size=1,
            fallback_rule="strict_eligible_only",
            locking_metric="roc_auc",
        ),
        seeds={"cv_seed": 42},
    )
    required = {
        "run_id",
        "feature_count",
        "canonical_features",
        "canonical_mask",
        "stable_mask_hash",
        "candidate_universe_sha256",
        "duplicate_mask_multiplicity",
        "locking_score",
        "fold_locking_scores",
        "score_sd",
        "score_se",
        "absolute_regret",
        "relative_regret",
        "eligible",
        "eligibility_reason",
        "fallback_added",
        "pairwise_jaccard",
        "mean_jaccard",
        "selected",
        "selected_feature_set",
        "tie_break_path",
        "tolerance_mode",
        "regret_tolerance",
        "minimum_pool_size",
        "locking_metric",
        "seeds",
        "software_version",
        "configuration_hash",
    }

    assert required.issubset(result.candidate_audit.columns)
    assert result.metadata["held_out_used"] is False
    assert result.metadata["metric_orientation"] == "larger_is_better"
