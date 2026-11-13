"""Calibration contract tests using synthetic development summaries only."""

from dataclasses import replace
from itertools import combinations
import inspect

import numpy as np
import pytest

from wrapevofs import (
    LockingCandidate, LockingConfig, calibrate_tolerance, lock_representative_run,
)
from wrapevofs.locking import jaccard


def candidate(run, features, score, spread=0.02):
    return LockingCandidate(run, features, score, [score - spread, score + spread])


def banks(spread=0.02):
    return {
        "a": [candidate(0, ["a"], 0.9, spread),
              candidate(1, ["shared"], 0.89), candidate(2, ["shared"], 0.885)],
        "b": [candidate(0, ["b"], 0.9, spread),
              candidate(1, ["shared"], 0.89), candidate(2, ["shared"], 0.885)],
    }


def test_breakpoints_filter_and_selection_provenance():
    result = calibrate_tolerance(banks(), fold_ids=["split_a", "split_b"])
    assert result.selected_delta == 0.9 - 0.885
    assert result.metadata["delta0_score_SE"] == pytest.approx(0.02)
    assert result.metadata["score_threshold"] == pytest.approx(0.88)
    assert result.metadata["fitted_bank_Jaccard"] == 1
    assert result.metadata["held_out_used"] is False
    assert result.metadata["fold_ids"] == ["split_a", "split_b"]
    assert result.curve.delta.tolist() == [0, 0.9 - 0.89, 0.9 - 0.885]
    assert result.curve.selected.sum() == 1
    assert result.curve.iloc[0].score_feasible
    for bank in result.bank_results.values():
        assert bank.selected_run_id == 1
        assert bank.selected_features == ["shared"]
        assert bank.metadata["selected_within_declared_tolerance"]
        assert bank.metadata["regret_tolerance"] == result.selected_delta
    selected = result.curve.loc[result.curve.selected].iloc[0]
    assert selected.selected_run_ids == {"a": 1, "b": 1}
    assert selected.selected_mask_hashes == {
        key: bank.metadata["selected_stable_mask_hash"] for key, bank in result.bank_results.items()
    }


def test_se_is_from_bank_averaged_folds_and_filter_can_reject_best_overlap():
    data = banks(spread=0)
    result = calibrate_tolerance(data)
    assert result.selected_delta == 0
    assert result.metadata["delta0_score_SE"] == 0
    assert not result.curve.iloc[-1].score_feasible
    # Opposing fold fluctuations cancel across banks; per-run SE averaging would not.
    data = banks()
    data["b"][0] = replace(data["b"][0], fold_locking_scores=[0.92, 0.88])
    assert calibrate_tolerance(data).metadata["delta0_score_SE"] == 0


def test_exact_breakpoint_and_next_float_below_are_not_interchangeable():
    gap = 0.9 - 0.885
    result = calibrate_tolerance(banks(), delta_grid=[gap, 0, np.nextafter(gap, 0)])
    assert result.selected_delta == gap
    assert result.curve.iloc[1].calibration_Jaccard == 0
    assert result.curve.iloc[2].calibration_Jaccard == 1
    assert result.metadata["search"] == "explicit_grid"


def test_tied_curve_uses_higher_score_then_smaller_delta():
    # Same fitted Jaccard at every delta: retain the higher-score baseline,
    # and pick zero from the remaining exact score ties.
    data = {key: [candidate(0, ["shared"], 0.9), candidate(1, ["shared"], 0.89)]
            for key in ["a", "b"]}
    result = calibrate_tolerance(data, delta_grid=[1, 0.01, 0, 1])
    assert result.selected_delta == 0
    assert result.metadata["delta_values"] == [0, 0.01, 1]


def test_fold_count_is_general_and_singleton_banks_are_valid():
    data = {key: [replace(candidate(0, [key], 0.5), fold_locking_scores=[0.4, 0.5, 0.6])]
            for key in ["a", "b"]}
    result = calibrate_tolerance(data)
    assert result.metadata["fold_count"] == 3
    assert result.selected_delta == 0
    assert result.metadata["delta0_score_SE"] == pytest.approx(0.1 / np.sqrt(3))


def test_order_invariance_and_input_is_not_mutated():
    data = banks()
    original = repr(data)
    result = calibrate_tolerance(data)
    permuted = {key: list(reversed(data[key])) for key in reversed(data)}
    other = calibrate_tolerance(permuted)
    assert other.selected_delta == result.selected_delta
    assert other.metadata["input_sha256"] == result.metadata["input_sha256"]
    assert other.curve.equals(result.curve)
    assert repr(data) == original


@pytest.mark.parametrize("change,match", [
    (dict(run_id=0.5), "integer"), (dict(run_id=True), "integer"),
    (dict(features=[]), "nonempty"), (dict(features=["x", "x"]), "duplicate"),
    (dict(features="abc"), "sequence"), (dict(features=[1]), "strings"),
    (dict(locking_score=float("nan")), "finite"),
    (dict(locking_score=float("inf")), "finite"), (dict(locking_score=True), "real"),
    (dict(locking_score=0.1), "fold mean"),
    (dict(fold_locking_scores=None), "required"),
    (dict(fold_locking_scores=[0.9]), "two common"),
    (dict(fold_locking_scores=[0.9, 0.9, 0.9]), "same fold count"),
    (dict(fold_locking_scores=[[0.9], [0.9]]), "real"),
    (dict(fold_locking_scores=[float("nan"), 0.9]), "finite"),
    (dict(fold_locking_scores=[0.9, float("inf")]), "finite"),
    (dict(seed=2.5), "seed"),
])
def test_invalid_candidate_is_rejected(change, match):
    data = banks()
    data["b"][1] = replace(data["b"][1], **change)
    with pytest.raises(ValueError, match=match):
        calibrate_tolerance(data)


@pytest.mark.parametrize("data,match", [
    ({}, "two"), ({"a": banks()["a"]}, "two"),
    ({"a": [], "b": banks()["b"]}, "empty"),
    ({0: banks()["a"], "b": banks()["b"]}, "Bank IDs"),
    ({"a": [None], "b": banks()["b"]}, "LockingCandidate"),
    ({"a": banks()["a"] * 2, "b": banks()["b"]}, "duplicate run_id"),
])
def test_invalid_bank_is_rejected(data, match):
    with pytest.raises(ValueError, match=match):
        calibrate_tolerance(data)


@pytest.mark.parametrize("grid", [[], [0.01], [0, -1], [0, float("nan")],
                                  [0, float("inf")], [0, True], [0, "0.01"], 0.1])
def test_invalid_grid_is_rejected(grid):
    with pytest.raises(ValueError, match="delta_grid"):
        calibrate_tolerance(banks(), delta_grid=grid)


@pytest.mark.parametrize("ids", [["one"], ["one", "one"], [0, 1], "ab"])
def test_invalid_fold_ids_are_rejected(ids):
    with pytest.raises(ValueError, match="fold_ids"):
        calibrate_tolerance(banks(), fold_ids=ids)


def test_supplied_universe_is_shared_and_respected():
    data = {key: [replace(c, candidate_universe=["shared", "b", "a", "unused"]) for c in group]
            for key, group in banks().items()}
    result = calibrate_tolerance(data)
    assert result.metadata["candidate_universe"] == ["shared", "b", "a", "unused"]
    data["b"][1] = replace(data["b"][1], candidate_universe=None)
    with pytest.raises(ValueError, match="same ordered"):
        calibrate_tolerance(data)
    data = {key: [replace(c, candidate_universe=["shared"]) for c in group]
            for key, group in banks().items()}
    with pytest.raises(ValueError, match="outside"):
        calibrate_tolerance(data)


def test_public_signature_has_no_heldout_channel():
    assert list(inspect.signature(calibrate_tolerance).parameters) == ["banks", "delta_grid", "fold_ids"]
    with pytest.raises(TypeError):
        calibrate_tolerance(banks(), held_out_auc=0.99)


def test_randomized_curve_matches_uncached_single_bank_locking():
    rng = np.random.default_rng(20261112)
    for _ in range(8):
        universe = [f"f{i}" for i in range(7)]
        data = {}
        for bank in range(3):
            group = []
            for run in range(4):
                features = rng.choice(universe, size=int(rng.integers(1, 6)), replace=False).tolist()
                folds = rng.uniform(0.65, 0.95, size=4).tolist()
                group.append(LockingCandidate(run, features, float(np.mean(folds)), folds,
                                              candidate_universe=universe))
            data[str(bank)] = group
        result = calibrate_tolerance(data)
        for row in result.curve.to_dict("records"):
            selected = []
            for key, group in data.items():
                locked = lock_representative_run(group, LockingConfig(
                    enabled=True, strategy="regret_constrained_medoid", regret_tolerance=row["delta"],
                ))
                assert locked.selected_run_id == row["selected_run_ids"][key]
                selected.append(next(c for c in group if c.run_id == locked.selected_run_id))
            expected = np.mean([jaccard(a.features, b.features) for a, b in combinations(selected, 2)])
            assert row["calibration_Jaccard"] == expected
        feasible = result.curve[result.curve.score_feasible].to_dict("records")
        expected = min(feasible, key=lambda r: (-r["calibration_Jaccard"], -r["calibration_mean_score"], r["delta"]))
        assert result.selected_delta == expected["delta"]
