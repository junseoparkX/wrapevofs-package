"""Development-only representative-run locking.

This module intentionally accepts only retained run candidates and development
scores. Held-out matrices, labels, predictions, and metrics are not part of the
API and therefore cannot influence eligibility or representative selection.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from wrapevofs.config import LockingConfig
from wrapevofs._scoring import infer_problem_type
from wrapevofs.selectors.genetic_rf import _stable_mask_hash


@dataclass(frozen=True)
class LockingCandidate:
    """One retained run-best subset scored only on development data."""

    run_id: int
    features: tuple[str, ...] | list[str]
    locking_score: float
    fold_locking_scores: tuple[float, ...] | list[float] | None = None
    seed: int | None = None
    candidate_universe: tuple[str, ...] | list[str] | None = None


@dataclass
class LockingResult:
    """Selected representative and complete machine-readable locking audit."""

    selected_run_id: int
    selected_features: list[str]
    candidate_audit: pd.DataFrame
    pairwise_jaccard: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def _canonical_mask(
    features: Iterable[str],
    candidate_universe: Sequence[str],
) -> np.ndarray:
    """Encode a feature set in one common, ordered candidate universe."""

    selected = set(features)
    return np.asarray(
        [feature in selected for feature in candidate_universe],
        dtype=np.uint8,
    )


def _candidate_universe_hash(candidate_universe: Sequence[str]) -> str:
    encoded = json.dumps(
        list(candidate_universe),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _configuration_hash(
    config: LockingConfig,
    full_configuration: Mapping[str, Any] | None,
) -> str:
    payload: dict[str, Any] = {"locking": asdict(config)}
    if full_configuration is not None:
        payload["pipeline"] = dict(full_configuration)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _candidate_statistics(candidate: LockingCandidate) -> tuple[str, float, float]:
    if candidate.fold_locking_scores is None:
        return "", float("nan"), float("nan")
    scores = np.asarray(candidate.fold_locking_scores, dtype=float)
    if scores.ndim != 1 or scores.size < 1 or not np.isfinite(scores).all():
        raise ValueError(
            f"run {candidate.run_id} fold_locking_scores must be a finite one-dimensional sequence."
        )
    if scores.size < 2:
        score_sd = float("nan")
        score_se = float("nan")
    else:
        score_sd = float(np.std(scores, ddof=1))
        score_se = score_sd / float(np.sqrt(scores.size))
    return json.dumps(scores.tolist()), score_sd, score_se


def _validate_candidates(
    candidates: Sequence[LockingCandidate],
) -> tuple[list[LockingCandidate], tuple[str, ...]]:
    if not candidates:
        raise ValueError("At least one locking candidate is required.")
    run_ids = [candidate.run_id for candidate in candidates]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("run_id values must be unique.")
    for candidate in candidates:
        raw_features = tuple(str(feature) for feature in candidate.features)
        if not raw_features:
            raise ValueError(f"run {candidate.run_id} has an empty feature mask.")
        if len(raw_features) != len(set(raw_features)):
            raise ValueError(f"run {candidate.run_id} contains duplicate feature names.")
    supplied_universes = [
        candidate.candidate_universe
        for candidate in candidates
        if candidate.candidate_universe is not None
    ]
    if supplied_universes and len(supplied_universes) != len(candidates):
        raise ValueError(
            "candidate_universe must be supplied for every candidate or for none."
        )
    if supplied_universes:
        universes = [tuple(str(feature) for feature in universe) for universe in supplied_universes]
        candidate_universe = universes[0]
        if any(universe != candidate_universe for universe in universes[1:]):
            raise ValueError("All candidates must use the same canonical candidate_universe.")
    else:
        candidate_universe = tuple(
            sorted(
                {
                    str(feature)
                    for candidate in candidates
                    for feature in candidate.features
                }
            )
        )
    if not candidate_universe:
        raise ValueError("The canonical candidate_universe cannot be empty.")
    if len(candidate_universe) != len(set(candidate_universe)):
        raise ValueError("candidate_universe contains duplicate feature names.")

    universe_set = set(candidate_universe)
    validated: list[LockingCandidate] = []
    for candidate in candidates:
        raw_features = tuple(str(feature) for feature in candidate.features)
        features = tuple(feature for feature in candidate_universe if feature in set(raw_features))
        if not features:
            raise ValueError(f"run {candidate.run_id} has an empty feature mask.")
        if len(raw_features) != len(set(raw_features)):
            raise ValueError(f"run {candidate.run_id} contains duplicate feature names.")
        missing = sorted(set(raw_features) - universe_set)
        if missing:
            raise ValueError(
                f"run {candidate.run_id} contains features outside candidate_universe: "
                f"{missing[:5]}"
            )
        score = float(candidate.locking_score)
        if not np.isfinite(score):
            raise ValueError(f"run {candidate.run_id} has a nonfinite locking score.")
        validated.append(
            LockingCandidate(
                run_id=int(candidate.run_id),
                features=features,
                locking_score=score,
                fold_locking_scores=(
                    None
                    if candidate.fold_locking_scores is None
                    else tuple(float(value) for value in candidate.fold_locking_scores)
                ),
                seed=candidate.seed,
                candidate_universe=candidate_universe,
            )
        )
    return validated, candidate_universe


def _pairwise_table(
    candidates: Sequence[LockingCandidate],
    stable_mask_hashes: Mapping[int, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ordered = sorted(
        candidates,
        key=lambda candidate: (stable_mask_hashes[candidate.run_id], candidate.run_id),
    )
    for left in ordered:
        for right in ordered:
            rows.append(
                {
                    "run_i": left.run_id,
                    "run_j": right.run_id,
                    "stable_mask_hash_i": stable_mask_hashes[left.run_id],
                    "stable_mask_hash_j": stable_mask_hashes[right.run_id],
                    "jaccard": jaccard(left.features, right.features),
                }
            )
    return pd.DataFrame(rows)


def _mean_jaccard(
    candidate: LockingCandidate,
    pool: Sequence[LockingCandidate],
) -> tuple[float, dict[str, float]]:
    peer_values = {
        str(peer.run_id): jaccard(candidate.features, peer.features)
        for peer in pool
        if peer.run_id != candidate.run_id
    }
    if not peer_values:
        return float("nan"), peer_values
    return float(np.mean(list(peer_values.values()))), peer_values


def lock_representative_run(
    candidates: Sequence[LockingCandidate],
    config: LockingConfig | None = None,
    *,
    software_version: str = "0.2.0",
    full_configuration: Mapping[str, Any] | None = None,
    seeds: Mapping[str, Any] | None = None,
) -> LockingResult:
    """Lock one representative run using development-only scores and features."""

    config = config or LockingConfig(enabled=True)
    candidates, candidate_universe = _validate_candidates(candidates)
    if config.strategy not in {"top_k_jaccard_medoid", "regret_constrained_medoid"}:
        raise ValueError(
            "locking.strategy must be 'top_k_jaccard_medoid' or "
            "'regret_constrained_medoid'."
        )
    if config.minimum_pool_size < 1 or config.minimum_pool_size > len(candidates):
        raise ValueError("minimum_pool_size must be between 1 and the candidate count.")
    if config.epsilon <= 0:
        raise ValueError("locking.epsilon must be positive.")
    if config.regret_tolerance < 0:
        raise ValueError("regret_tolerance must be nonnegative.")
    if config.metric_orientation != "larger_is_better":
        raise ValueError(
            "locking.metric_orientation must be 'larger_is_better'; "
            "lower-is-better scores must be transformed before locking."
        )
    if config.strategy == "regret_constrained_medoid" and config.tie_breakers != [
        "higher_locking_score",
        "smaller_feature_count",
        "stable_mask_hash",
    ]:
        raise ValueError(
            "regret_constrained_medoid requires canonical tie_breakers: "
            "higher_locking_score, smaller_feature_count, stable_mask_hash."
        )

    masks = {
        candidate.run_id: _canonical_mask(candidate.features, candidate_universe)
        for candidate in candidates
    }
    stable_mask_hashes = {
        candidate.run_id: _stable_mask_hash(masks[candidate.run_id])
        for candidate in candidates
    }
    hashes_to_masks: dict[str, bytes] = {}
    for candidate in candidates:
        mask_bytes = masks[candidate.run_id].tobytes()
        prior = hashes_to_masks.setdefault(stable_mask_hashes[candidate.run_id], mask_bytes)
        if prior != mask_bytes:
            raise RuntimeError("A stable-mask SHA-256 collision was detected.")
    mask_multiplicity = {
        mask_hash: list(stable_mask_hashes.values()).count(mask_hash)
        for mask_hash in set(stable_mask_hashes.values())
    }

    by_score = sorted(
        candidates,
        key=lambda candidate: (
            -candidate.locking_score,
            stable_mask_hashes[candidate.run_id],
            candidate.run_id,
        ),
    )
    best_score = max(candidate.locking_score for candidate in candidates)
    absolute_regret = {
        candidate.run_id: best_score - candidate.locking_score for candidate in candidates
    }
    relative_regret = {
        candidate.run_id: absolute_regret[candidate.run_id]
        / max(abs(best_score), config.epsilon)
        for candidate in candidates
    }

    eligibility_reason: dict[int, str] = {}
    fallback_added: dict[int, bool] = {candidate.run_id: False for candidate in candidates}
    eligibility_threshold: float
    effective_tolerance_mode = (
        "best_run_se_scaled" if config.tolerance_mode == "one_se" else config.tolerance_mode
    )
    if config.strategy == "top_k_jaccard_medoid":
        if config.top_k < 1:
            raise ValueError("top_k must be at least 1.")
        pool = by_score[: min(config.top_k, len(by_score))]
        eligibility_threshold = float("nan")
        for candidate in candidates:
            eligibility_reason[candidate.run_id] = (
                "legacy_top_k_by_score"
                if candidate in pool
                else "outside_legacy_top_k"
            )
    else:
        if config.minimum_pool_size != 1:
            raise ValueError(
                "strict regret_constrained_medoid requires minimum_pool_size=1; "
                "candidates outside the declared tolerance are never added."
            )
        if config.fallback_rule != "strict_eligible_only":
            raise ValueError(
                "strict regret_constrained_medoid requires "
                "fallback_rule='strict_eligible_only'."
            )
        if effective_tolerance_mode == "absolute":
            eligibility_threshold = float(config.regret_tolerance)
            pool = [
                candidate
                for candidate in candidates
                if absolute_regret[candidate.run_id]
                <= eligibility_threshold
            ]
            for candidate in candidates:
                eligibility_reason[candidate.run_id] = (
                    "absolute_regret_within_tolerance"
                    if candidate in pool
                    else "absolute_regret_exceeds_tolerance"
                )
        elif effective_tolerance_mode == "relative":
            eligibility_threshold = float(config.regret_tolerance)
            pool = [
                candidate
                for candidate in candidates
                if relative_regret[candidate.run_id]
                <= eligibility_threshold
            ]
            for candidate in candidates:
                eligibility_reason[candidate.run_id] = (
                    "relative_regret_within_tolerance"
                    if candidate in pool
                    else "relative_regret_exceeds_tolerance"
                )
        elif effective_tolerance_mode == "best_run_se_scaled":
            best_candidates = [
                candidate for candidate in candidates if candidate.locking_score == best_score
            ]
            best = min(
                best_candidates,
                key=lambda candidate: (
                    stable_mask_hashes[candidate.run_id],
                    candidate.run_id,
                ),
            )
            if best.fold_locking_scores is None:
                raise ValueError(
                    "best_run_se_scaled eligibility requires fold_locking_scores for the best "
                    "development run; configure an explicit non-one-SE mode or "
                    "regenerate development-fold scores."
                )
            _, _, best_se = _candidate_statistics(best)
            if not np.isfinite(best_se):
                raise ValueError(
                    "best_run_se_scaled eligibility requires at least two finite development-fold "
                    "locking scores for the best run."
                )
            eligibility_threshold = best_se
            pool = [
                candidate
                for candidate in candidates
                if absolute_regret[candidate.run_id]
                <= eligibility_threshold
            ]
            for candidate in candidates:
                eligibility_reason[candidate.run_id] = (
                    "absolute_regret_within_best_run_se_scaled_threshold"
                    if candidate in pool
                    else "absolute_regret_exceeds_best_run_se_scaled_threshold"
                )
        else:
            raise ValueError(
                "tolerance_mode must be 'absolute', 'relative', or "
                "'best_run_se_scaled' ('one_se' is a deprecated alias)."
            )

        # The best-scoring candidate has zero regret, so a nonempty strict pool
        # always exists for nonnegative tolerances. No candidate outside the
        # declared threshold is admitted to representative selection.
        if not pool:
            raise RuntimeError("strict regret eligibility unexpectedly produced an empty pool.")

    pool = sorted(
        pool,
        key=lambda candidate: (stable_mask_hashes[candidate.run_id], candidate.run_id),
    )
    mean_jaccard: dict[int, float] = {}
    peer_jaccard: dict[int, dict[str, float]] = {}
    for candidate in pool:
        mean_value, peer_values = _mean_jaccard(candidate, pool)
        mean_jaccard[candidate.run_id] = mean_value
        peer_jaccard[candidate.run_id] = peer_values

    if len(pool) == 1:
        selected = pool[0]
        if config.strategy == "top_k_jaccard_medoid":
            tie_break_path = "mean_jaccard > higher_locking_score > lower_run_id"
        else:
            tie_break_path = (
                "mean_jaccard > higher_locking_score > smaller_feature_count > "
                "stable_mask_hash"
            )
    elif config.strategy == "top_k_jaccard_medoid":
        selection_order = sorted(
            pool,
            key=lambda candidate: (
                -mean_jaccard[candidate.run_id],
                -candidate.locking_score,
                candidate.run_id,
            ),
        )
        tie_break_path = "mean_jaccard > higher_locking_score > lower_run_id"
        selected = selection_order[0]
    else:
        selection_order = sorted(
            pool,
            key=lambda candidate: (
                -mean_jaccard[candidate.run_id],
                -candidate.locking_score,
                len(candidate.features),
                stable_mask_hashes[candidate.run_id],
            ),
        )
        tie_break_path = (
            "mean_jaccard > higher_locking_score > smaller_feature_count > "
            "stable_mask_hash"
        )
        best_selection_key = (
            -mean_jaccard[selection_order[0].run_id],
            -selection_order[0].locking_score,
            len(selection_order[0].features),
            stable_mask_hashes[selection_order[0].run_id],
        )
        scientifically_equivalent = [
            candidate
            for candidate in selection_order
            if (
                -mean_jaccard[candidate.run_id],
                -candidate.locking_score,
                len(candidate.features),
                stable_mask_hashes[candidate.run_id],
            )
            == best_selection_key
        ]
        # Exact duplicate records represent the same scientific feature set.
        # A source run is retained only as provenance, after mask selection.
        selected = min(scientifically_equivalent, key=lambda candidate: candidate.run_id)

    selected_mask_hash = stable_mask_hashes[selected.run_id]
    selected_source_run_ids = sorted(
        candidate.run_id
        for candidate in candidates
        if stable_mask_hashes[candidate.run_id] == selected_mask_hash
    )
    if config.strategy == "regret_constrained_medoid":
        if config.tolerance_mode == "relative":
            selected_within_tolerance = (
                relative_regret[selected.run_id] <= eligibility_threshold
            )
        else:
            selected_within_tolerance = (
                absolute_regret[selected.run_id] <= eligibility_threshold
            )
        if not selected_within_tolerance:
            raise RuntimeError("Selected candidate violates the declared regret tolerance.")
    else:
        selected_within_tolerance = None

    config_hash = _configuration_hash(config, full_configuration)
    universe_hash = _candidate_universe_hash(candidate_universe)
    seeds_json = json.dumps(dict(seeds or {}), sort_keys=True)
    rows: list[dict[str, Any]] = []
    pool_ids = {candidate.run_id for candidate in pool}
    for candidate in sorted(
        candidates,
        key=lambda item: (stable_mask_hashes[item.run_id], item.run_id),
    ):
        fold_json, score_sd, score_se = _candidate_statistics(candidate)
        candidate_seeds = dict(seeds or {})
        if candidate.seed is not None:
            candidate_seeds["run_seed"] = candidate.seed
        rows.append(
            {
                "run_id": candidate.run_id,
                "feature_count": len(candidate.features),
                "canonical_features": json.dumps(list(candidate.features), ensure_ascii=False),
                "canonical_mask": "".join(str(int(value)) for value in masks[candidate.run_id]),
                "stable_mask_hash": stable_mask_hashes[candidate.run_id],
                "candidate_universe_sha256": universe_hash,
                "duplicate_mask_multiplicity": mask_multiplicity[
                    stable_mask_hashes[candidate.run_id]
                ],
                "locking_score": candidate.locking_score,
                "fold_locking_scores": fold_json,
                "score_sd": score_sd,
                "score_se": score_se,
                "absolute_regret": absolute_regret[candidate.run_id],
                "relative_regret": relative_regret[candidate.run_id],
                "eligible": candidate.run_id in pool_ids,
                "eligibility_reason": eligibility_reason[candidate.run_id],
                "fallback_added": fallback_added[candidate.run_id],
                "pairwise_jaccard": json.dumps(
                    peer_jaccard.get(candidate.run_id, {}), sort_keys=True
                ),
                "mean_jaccard": mean_jaccard.get(candidate.run_id, float("nan")),
                "selected": candidate.run_id == selected.run_id,
                "selected_feature_set": (
                    stable_mask_hashes[candidate.run_id] == selected_mask_hash
                ),
                "tie_break_path": tie_break_path,
                "strategy": config.strategy,
                "tolerance_mode": effective_tolerance_mode,
                "regret_tolerance": config.regret_tolerance,
                "eligibility_threshold": eligibility_threshold,
                "minimum_pool_size": config.minimum_pool_size,
                "locking_metric": config.locking_metric,
                "seeds": json.dumps(candidate_seeds, sort_keys=True)
                if candidate_seeds
                else seeds_json,
                "software_version": software_version,
                "configuration_hash": config_hash,
            }
        )

    return LockingResult(
        selected_run_id=selected.run_id,
        selected_features=list(selected.features),
        candidate_audit=pd.DataFrame(rows),
        pairwise_jaccard=_pairwise_table(pool, stable_mask_hashes),
        metadata={
            "strategy": config.strategy,
            "tolerance_mode": effective_tolerance_mode,
            "regret_tolerance": config.regret_tolerance,
            "eligibility_threshold": eligibility_threshold,
            "minimum_pool_size": config.minimum_pool_size,
            "fallback_expansion_occurred": any(fallback_added.values()),
            "eligible_run_ids": sorted(candidate.run_id for candidate in pool),
            "selected_run_id": selected.run_id,
            "selected_source_run_ids": selected_source_run_ids,
            "selected_stable_mask_hash": selected_mask_hash,
            "selected_source_run_id_role": (
                "provenance_only_after_scientific_feature_set_selection"
            ),
            "selected_absolute_regret": absolute_regret[selected.run_id],
            "selected_relative_regret": relative_regret[selected.run_id],
            "selected_within_declared_tolerance": selected_within_tolerance,
            "strict_regret_constraint": config.strategy == "regret_constrained_medoid",
            "metric_orientation": config.metric_orientation,
            "tie_break_path": tie_break_path,
            "duplicate_mask_policy": "retain_multiplicity_as_voting_candidates",
            "candidate_universe": list(candidate_universe),
            "candidate_universe_sha256": universe_hash,
            "stable_mask_hash_algorithm": "sha256(uint8_canonical_mask_bytes)",
            "locking_metric": config.locking_metric,
            "software_version": software_version,
            "configuration_hash": config_hash,
            "held_out_used": False,
            "artifact_schema_version": "2.0",
        },
    )


def score_candidate_feature_sets(
    X_development: pd.DataFrame,
    y_development: pd.Series,
    candidate_feature_sets: Mapping[int, Sequence[str]],
    *,
    locking_metric: str = "auto",
    cv_folds: int = 5,
    random_state: int = 42,
    rf_params: Mapping[str, Any] | None = None,
    n_jobs: int = 1,
    run_seeds: Mapping[int, int] | None = None,
) -> list[LockingCandidate]:
    """Score retained candidates with fixed development-only CV folds."""

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    if not candidate_feature_sets:
        raise ValueError("candidate_feature_sets cannot be empty.")
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")
    problem_type = infer_problem_type(y_development)
    metric = locking_metric
    if metric == "auto":
        metric = "roc_auc" if problem_type == "binary" else "roc_auc_ovr"
    if metric == "macro_ovr_auroc":
        metric = "roc_auc" if problem_type == "binary" else "roc_auc_ovr"
    allowed = {
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "roc_auc_ovr",
        "roc_auc_ovr_weighted",
    }
    if metric not in allowed:
        raise ValueError(f"Unsupported locking_metric={locking_metric!r}.")

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )
    parameters = dict(rf_params or {})
    parameters.setdefault("n_estimators", 150)
    parameters.setdefault("max_depth", 12)
    parameters.setdefault("min_samples_leaf", 2)
    parameters.setdefault("bootstrap", True)
    candidate_universe = tuple(str(feature) for feature in X_development.columns)
    if len(candidate_universe) != len(set(candidate_universe)):
        raise ValueError("X_development columns must define a unique candidate universe.")
    candidates: list[LockingCandidate] = []
    for run_id in sorted(candidate_feature_sets):
        features = [str(feature) for feature in candidate_feature_sets[run_id]]
        if not features:
            raise ValueError(f"run {run_id} has an empty feature mask.")
        missing = sorted(set(features) - set(X_development.columns))
        if missing:
            raise KeyError(f"run {run_id} has missing development features: {missing[:5]}")
        run_seed = int((run_seeds or {}).get(run_id, random_state + int(run_id)))
        estimator = RandomForestClassifier(
            **parameters,
            random_state=run_seed,
            n_jobs=n_jobs,
        )
        fold_scores = cross_val_score(
            estimator,
            X_development.loc[:, features],
            y_development,
            cv=cv,
            scoring=metric,
            n_jobs=n_jobs,
        )
        if not np.isfinite(fold_scores).all():
            raise ValueError(f"run {run_id} produced nonfinite development-CV scores.")
        candidates.append(
            LockingCandidate(
                run_id=int(run_id),
                features=features,
                locking_score=float(np.mean(fold_scores)),
                fold_locking_scores=tuple(float(value) for value in fold_scores),
                seed=run_seed,
                candidate_universe=candidate_universe,
            )
        )
    return candidates
