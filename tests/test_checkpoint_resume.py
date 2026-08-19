import json

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from wrapevofs.config import GAConfig
from wrapevofs.selectors.genetic_rf import run_genetic_rf


def _data():
    X, y = make_classification(
        n_samples=40,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=101,
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]), pd.Series(y)


def _config(checkpoint_dir, *, resume=False, **overrides):
    values = {
        "backend": "cpu",
        "fitness_mode": "untruncated_shifted_linear",
        "population_size": 4,
        "n_generations": 3,
        "n_runs": 1,
        "top_k": 1,
        "cv_folds": 2,
        "random_state": 17,
        "progress_interval": 1,
        "checkpoint_dir": str(checkpoint_dir),
        "resume_from_checkpoint": resume,
        "rf_params": {"n_estimators": 2, "max_depth": 2},
    }
    values.update(overrides)
    return GAConfig(**values)


def _interrupt_after_first_checkpoint(monkeypatch, checkpoint_dir):
    import wrapevofs.selectors.genetic_rf as genetic_rf

    original = genetic_rf._write_resume_checkpoint

    def write_then_interrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs["next_generation"] == 1:
            raise RuntimeError("simulated interruption")
        return result

    monkeypatch.setattr(genetic_rf, "_write_resume_checkpoint", write_then_interrupt)
    X, y = _data()
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_genetic_rf(X, y, 2, name="resume_test", config=_config(checkpoint_dir))
    monkeypatch.setattr(genetic_rf, "_write_resume_checkpoint", original)
    return X, y


def _rewrite_metadata(path, mutate):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        population = archive["population"].copy()
    mutate(metadata)
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata=np.asarray(
                json.dumps(metadata, sort_keys=True, separators=(",", ":"))
            ),
            population=population,
        )


def test_shortened_uninterrupted_and_resumed_runs_are_identical(
    tmp_path, monkeypatch
):
    X, y = _interrupt_after_first_checkpoint(monkeypatch, tmp_path / "resumed")
    resumed = run_genetic_rf(
        X,
        y,
        2,
        name="resume_test",
        config=_config(tmp_path / "resumed", resume=True),
    )
    uninterrupted = run_genetic_rf(
        X,
        y,
        2,
        name="resume_test",
        config=_config(tmp_path / "uninterrupted"),
    )

    assert [item.selected_features for item in resumed.top_solutions] == [
        item.selected_features for item in uninterrupted.top_solutions
    ]
    assert [item.score for item in resumed.top_solutions] == pytest.approx(
        [item.score for item in uninterrupted.top_solutions]
    )
    pd.testing.assert_frame_equal(resumed.history, uninterrupted.history)
    assert resumed.metadata["resumed_from_checkpoint"] is True
    assert not list((tmp_path / "resumed").glob("*.tmp"))


def test_corrupt_checkpoint_is_rejected(tmp_path):
    X, y = _data()
    path = tmp_path / "resume_state.npz"
    path.write_bytes(b"not an npz archive")
    with pytest.raises(ValueError, match="corrupt or unreadable"):
        run_genetic_rf(X, y, 2, name="resume_test", config=_config(tmp_path, resume=True))


def test_missing_checkpoint_is_rejected(tmp_path):
    X, y = _data()
    with pytest.raises(ValueError, match="does not exist"):
        run_genetic_rf(X, y, 2, name="resume_test", config=_config(tmp_path, resume=True))


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda metadata: metadata.__setitem__("software_version", "0.0.invalid"), "software_version"),
        (lambda metadata: metadata.pop("configuration_sha256"), "configuration_sha256"),
    ],
)
def test_checkpoint_version_or_required_context_mismatch_is_rejected(
    tmp_path, monkeypatch, mutation, match
):
    X, y = _interrupt_after_first_checkpoint(monkeypatch, tmp_path)
    _rewrite_metadata(tmp_path / "resume_state.npz", mutation)
    with pytest.raises(ValueError, match=match):
        run_genetic_rf(X, y, 2, name="resume_test", config=_config(tmp_path, resume=True))


def test_configuration_mismatch_is_rejected(tmp_path, monkeypatch):
    X, y = _interrupt_after_first_checkpoint(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="configuration_sha256"):
        run_genetic_rf(
            X,
            y,
            2,
            name="resume_test",
            config=_config(tmp_path, resume=True, mutation_rate=0.2),
        )


def test_feature_order_and_input_mismatches_are_rejected(tmp_path, monkeypatch):
    X, y = _interrupt_after_first_checkpoint(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="candidate_universe_sha256|input_sha256"):
        run_genetic_rf(
            X.loc[:, list(reversed(X.columns))],
            y,
            2,
            name="resume_test",
            config=_config(tmp_path, resume=True),
        )
    changed = X.copy()
    changed.iloc[0, 0] += 1.0
    with pytest.raises(ValueError, match="input_sha256"):
        run_genetic_rf(
            changed,
            y,
            2,
            name="resume_test",
            config=_config(tmp_path, resume=True),
        )
