import pandas as pd
from sklearn.datasets import make_classification

from wrapevofs.config import RFECVConfig
from wrapevofs import WrapEvoPipeline, PipelineConfig
from wrapevofs.selectors.rfecv_target import find_rfecv_target


def test_rfecv_target_respects_user_max_features():
    X, y = make_classification(
        n_samples=60,
        n_features=8,
        n_informative=3,
        random_state=42,
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    result = find_rfecv_target(
        X,
        pd.Series(y),
        RFECVConfig(
            max_features_to_consider=3,
            cv_folds=3,
            n_jobs=1,
            rf_params={"n_estimators": 5, "max_depth": 4},
        ),
    )
    assert 1 <= result.target_k <= 3
    assert len(result.selected_features_at_target) == result.target_k
    assert set(result.selected_features_at_target).issubset(set(X.columns))
    assert result.metadata["problem_type"] == "binary"
    assert result.metadata["resolved_scoring"] == "roc_auc"


def test_pipeline_uses_method_specific_rfecv_caps():
    config = PipelineConfig()
    pipeline = WrapEvoPipeline(config)

    assert pipeline._rfecv_config_for_method("xgboost").max_features_to_consider == 20
    assert pipeline._rfecv_config_for_method("svm_l1").max_features_to_consider == 20
    assert pipeline._rfecv_config_for_method("boruta_rf").max_features_to_consider == 25


def test_rfecv_target_auto_scoring_supports_multiclass():
    X, y = make_classification(
        n_samples=90,
        n_features=9,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        random_state=11,
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(9)])

    result = find_rfecv_target(
        X,
        pd.Series(y),
        RFECVConfig(
            scoring="auto",
            max_features_to_consider=5,
            cv_folds=3,
            n_jobs=1,
            rf_params={"n_estimators": 10, "max_depth": 4},
        ),
    )

    assert 1 <= result.target_k <= 5
    assert result.metadata["problem_type"] == "multiclass"
    assert result.metadata["resolved_scoring"] == "roc_auc_ovr"
    assert result.metadata["n_classes"] == 3
