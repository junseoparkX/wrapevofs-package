"""Opt-in multi-bank tolerance calibration using development summaries only.

The fold-SE filter is a training heuristic, not a generalization or
noninferiority guarantee. No held-out values or participant data are accepted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from itertools import combinations
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from wrapevofs._version import __version__
from wrapevofs.config import LockingConfig
from wrapevofs.locking import (
    LockingCandidate,
    LockingResult,
    _candidate_universe_hash,
    jaccard,
    lock_representative_run,
)


@dataclass
class ToleranceCalibrationResult:
    """Fitted allowance, complete search curve, and per-bank locking audits."""

    selected_delta: float
    curve: pd.DataFrame
    bank_results: dict[str, LockingResult]
    metadata: dict[str, Any]


def _names(values: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of unique nonempty strings.")
    try:
        names = tuple(values)
    except TypeError as error:
        raise ValueError(f"{label} must be a sequence of strings.") from error
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"{label} must contain nonempty strings.")
    if len(names) != len(set(names)):
        raise ValueError(f"{label} contains duplicate names.")
    return names


def _finite_real(value: Real, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite real number.")
    try:
        number = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a finite real number.") from error
    if not np.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return number


def _prepare_banks(
    banks: Mapping[str, Sequence[LockingCandidate]],
) -> tuple[dict[str, list[LockingCandidate]], tuple[str, ...], int]:
    if not isinstance(banks, Mapping) or len(banks) < 2:
        raise ValueError("At least two development banks are required.")
    if any(not isinstance(key, str) or not key for key in banks):
        raise ValueError("Bank IDs must be nonempty strings.")
    prepared: dict[str, list[LockingCandidate]] = {}
    universes: list[tuple[str, ...] | None] = []
    fold_count: int | None = None
    for bank_id in sorted(banks):
        try:
            candidates = list(banks[bank_id])
        except TypeError as error:
            raise ValueError(f"Bank {bank_id!r} must contain locking candidates.") from error
        if not candidates:
            raise ValueError(f"Bank {bank_id!r} is empty.")
        seen: set[int] = set()
        prepared[bank_id] = []
        for candidate in candidates:
            if not isinstance(candidate, LockingCandidate):
                raise ValueError("Every bank entry must be a LockingCandidate.")
            run_id = candidate.run_id
            label = f"bank {bank_id!r}, run {run_id!r}"
            if isinstance(run_id, (bool, np.bool_)) or not isinstance(run_id, Integral):
                raise ValueError(f"{label}: run_id must be an integer.")
            if run_id in seen:
                raise ValueError(f"{label}: duplicate run_id within bank.")
            seen.add(run_id)
            features = _names(candidate.features, f"{label} features")
            universe = (
                None if candidate.candidate_universe is None
                else _names(candidate.candidate_universe, "candidate_universe")
            )
            universes.append(universe)
            score = _finite_real(candidate.locking_score, f"{label} locking_score")
            if candidate.fold_locking_scores is None:
                raise ValueError(f"{label}: fold_locking_scores are required.")
            try:
                folds = tuple(
                    _finite_real(value, f"{label} fold score")
                    for value in candidate.fold_locking_scores
                )
            except TypeError as error:
                raise ValueError(f"{label}: fold scores must be a one-dimensional sequence.") from error
            if len(folds) < 2:
                raise ValueError("At least two common development folds are required.")
            if fold_count is None:
                fold_count = len(folds)
            if len(folds) != fold_count:
                raise ValueError("All candidates in all banks must have the same fold count.")
            mean = float(np.mean(folds))
            if not np.isfinite(mean) or abs(mean - score) > 1e-12:
                raise ValueError(f"{label}: locking_score must equal the fold mean within 1e-12.")
            if candidate.seed is not None and (
                isinstance(candidate.seed, (bool, np.bool_))
                or not isinstance(candidate.seed, Integral)
            ):
                raise ValueError(f"{label}: seed must be an integer or None.")
            prepared[bank_id].append(replace(
                candidate, run_id=int(run_id), features=features, locking_score=score,
                fold_locking_scores=folds,
                seed=None if candidate.seed is None else int(candidate.seed),
            ))
    supplied = [universe for universe in universes if universe is not None]
    if supplied:
        universe = supplied[0]
        if len(supplied) != len(universes) or any(item != universe for item in supplied):
            raise ValueError("Supply the same ordered candidate_universe for every candidate or none.")
    else:
        universe = tuple(sorted({f for bank in prepared.values() for c in bank for f in c.features}))
    universe_set = set(universe)
    for bank_id, candidates in prepared.items():
        normalized = []
        for candidate in candidates:
            selected = set(candidate.features)
            if not selected <= universe_set:
                raise ValueError(f"Bank {bank_id!r}: features outside candidate_universe.")
            normalized.append(replace(
                candidate, features=tuple(f for f in universe if f in selected),
                candidate_universe=universe,
            ))
        prepared[bank_id] = sorted(normalized, key=lambda c: c.run_id)
    return prepared, universe, int(fold_count)


def calibrate_tolerance(
    banks: Mapping[str, Sequence[LockingCandidate]],
    *,
    delta_grid: Sequence[float] | None = None,
    fold_ids: Sequence[str] | None = None,
) -> ToleranceCalibrationResult:
    """Fit one absolute regret allowance across development candidate banks.

    Each score must be the unweighted mean of its finite fold vector (absolute
    tolerance 1e-12). All candidates must use the SAME development splits,
    metric, orientation (larger is better), and fold order. Fold identities
    cannot be inferred from scores; the caller is responsible for alignment.
    Optional unique ``fold_ids`` label this shared order in the audit.

    By default, search zero and every distinct observed within-bank score gap.
    An explicit finite, nonnegative ``delta_grid`` must include zero; it opts
    into grid comparison instead. Eligibility always uses full precision.

    At each delta, use the unchanged regret-constrained medoid and its tie
    rules. Let m0 be the mean delta-zero selected score and SE0 the sample SD
    of its bank-averaged fold vector divided by sqrt(K). Retain deltas with
    mean selected score >= m0 - SE0. Maximize mean pairwise selected-bank
    Jaccard, breaking ties by higher mean score and then smaller delta.

    This optimizes fitted overlap under a training heuristic. It provides no
    generalization, predictive superiority, or noninferiority guarantee and
    does not change pipeline defaults or single-bank locking behavior.
    """
    banks, universe, fold_count = _prepare_banks(banks)
    ids = (
        tuple(f"fold_{index}" for index in range(fold_count))
        if fold_ids is None else _names(fold_ids, "fold_ids")
    )
    if len(ids) != fold_count:
        raise ValueError("fold_ids must match the common fold count.")
    regrets = {
        bank_id: tuple(max(c.locking_score for c in bank) - c.locking_score for c in bank)
        for bank_id, bank in banks.items()
    }
    if any(not np.isfinite(gap) for gaps in regrets.values() for gap in gaps):
        raise ValueError("Within-bank score gaps must be finite.")
    if delta_grid is None:
        deltas = sorted({0.0} | {gap for gaps in regrets.values() for gap in gaps})
        search = "observed_breakpoints"
    else:
        try:
            deltas = sorted({_finite_real(value, "delta_grid value") for value in delta_grid})
        except TypeError as error:
            raise ValueError("delta_grid must be a sequence of nonnegative finite numbers.") from error
        if not deltas or deltas[0] < 0 or 0.0 not in deltas:
            raise ValueError("delta_grid must be nonnegative and include zero.")
        search = "explicit_grid"

    def lock(bank: list[LockingCandidate], delta: float) -> LockingResult:
        return lock_representative_run(bank, LockingConfig(
            enabled=True, strategy="regret_constrained_medoid",
            tolerance_mode="absolute", regret_tolerance=delta,
            minimum_pool_size=1, fallback_rule="strict_eligible_only",
        ))

    # Reuse a selection only while its exact eligible pool is unchanged.
    cache: dict[tuple[str, tuple[int, ...]], tuple[LockingCandidate, str]] = {}
    rows: list[dict[str, Any]] = []
    for delta in deltas:
        chosen: list[LockingCandidate] = []
        hashes: dict[str, str] = {}
        for bank_id, bank in banks.items():
            eligible = tuple(c.run_id for c, gap in zip(bank, regrets[bank_id]) if gap <= delta)
            key = (bank_id, eligible)
            if key not in cache:
                locked = lock(bank, delta)
                candidate = next(c for c in bank if c.run_id == locked.selected_run_id)
                cache[key] = (candidate, locked.metadata["selected_stable_mask_hash"])
            candidate, hashes[bank_id] = cache[key]
            chosen.append(candidate)
        folds = np.asarray([c.fold_locking_scores for c in chosen], dtype=float)
        rows.append({
            "delta": delta,
            "calibration_Jaccard": float(np.mean([
                jaccard(a.features, b.features) for a, b in combinations(chosen, 2)
            ])),
            "calibration_mean_score": float(np.mean([c.locking_score for c in chosen])),
            "bank_averaged_fold_scores": folds.mean(axis=0).tolist(),
            "selected_run_ids": dict(zip(banks, [c.run_id for c in chosen])),
            "selected_mask_hashes": hashes,
        })
    baseline = rows[0]
    se0 = float(np.std(baseline["bank_averaged_fold_scores"], ddof=1) / np.sqrt(fold_count))
    threshold = baseline["calibration_mean_score"] - se0
    if not np.isfinite(se0) or not np.isfinite(threshold) or any(
        not np.isfinite(row["calibration_mean_score"])
        or not np.isfinite(row["bank_averaged_fold_scores"]).all()
        for row in rows
    ):
        raise ValueError("Development score aggregation and SE0 must remain finite.")
    for row in rows:
        row.update({
            "delta0_score_SE": se0,
            "score_threshold": threshold,
            "score_feasible": row["calibration_mean_score"] >= threshold,
            "mean_score_loss_vs_delta0": baseline["calibration_mean_score"] - row["calibration_mean_score"],
        })
    winner = min(
        (row for row in rows if row["score_feasible"]),
        key=lambda row: (-row["calibration_Jaccard"], -row["calibration_mean_score"], row["delta"]),
    )
    for row in rows:
        row["selected"] = row is winner
    delta = winner["delta"]
    payload = {
        "banks": {key: [asdict(c) for c in bank] for key, bank in banks.items()},
        "fold_ids": ids, "search": search, "deltas": deltas,
    }
    digest = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")).hexdigest()
    return ToleranceCalibrationResult(
        selected_delta=delta,
        curve=pd.DataFrame(rows),
        bank_results={key: lock(bank, delta) for key, bank in banks.items()},
        metadata={
            "calibration_schema_version": "1.0", "software_version": __version__,
            "search": search, "delta_values": deltas,
            "bank_ids": list(banks), "bank_count": len(banks),
            "candidate_counts": {key: len(bank) for key, bank in banks.items()},
            "fold_count": fold_count, "fold_ids": list(ids),
            "fold_alignment": "caller_supplied_common_splits_and_order",
            "candidate_universe": list(universe),
            "candidate_universe_sha256": _candidate_universe_hash(universe),
            "input_sha256": digest, "selected_delta": delta,
            "baseline_mean_score": baseline["calibration_mean_score"],
            "delta0_score_SE": se0, "score_threshold": threshold,
            "fitted_bank_Jaccard": winner["calibration_Jaccard"],
            "mean_selected_development_score": winner["calibration_mean_score"],
            "selection_tie_breakers": ["higher_bank_Jaccard", "higher_mean_development_score", "smaller_delta"],
            "score_filter_role": "training_heuristic_without_generalization_or_noninferiority_guarantee",
            "held_out_used": False,
        },
    )
