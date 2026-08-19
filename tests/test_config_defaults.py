from wrapevofs import PipelineConfig


def test_default_config_matches_legacy_package_defaults():
    config = PipelineConfig()

    assert config.split.ratio == "7:3"
    assert config.preprocessing.impute_strategy == "zero"
    assert config.preprocessing.missingness_threshold is None
    assert config.preprocessing.drop_zero_variance is False
    assert config.preprocessing.scaling == "none"
    assert config.preprocessing.correlation_threshold is None

    assert config.first_stage.enabled_methods == ["xgboost", "svm_l1", "boruta_rf"]
    assert config.first_stage.xgboost.top_k == 100
    assert config.first_stage.xgboost.scoring == "auto"
    assert config.first_stage.xgboost.n_jobs == -1
    assert config.first_stage.xgboost.base_params["reg_lambda"] == 1
    assert config.first_stage.xgboost.base_params["reg_alpha"] == 0

    assert config.first_stage.svm_l1.coefficient_threshold == 1e-6
    assert config.first_stage.svm_l1.standardize is True
    assert config.first_stage.svm_l1.scoring == "auto"
    assert config.first_stage.svm_l1.c_grid == [0.001, 0.01, 0.1, 1.0, 10.0, 30.0, 100.0]
    assert config.first_stage.svm_l1.n_jobs == -1
    assert config.first_stage.svm_l1.penalty == "l1"
    assert config.first_stage.svm_l1.dual is False
    assert config.first_stage.svm_l1.max_iter == 100000

    assert config.rfecv.max_features_to_consider == 25
    assert config.rfecv.scoring == "auto"
    assert config.rfecv.method_max_features_to_consider["xgboost"] == 20
    assert config.rfecv.method_max_features_to_consider["svm_l1"] == 20
    assert config.rfecv.method_max_features_to_consider["boruta_rf"] == 25
    assert config.rfecv.n_jobs == 1
    assert config.ga.backend == "auto"
    assert config.ga.population_size == 50
    assert config.ga.size_penalty_lambda == 0.015
    assert config.ga.fitness_mode == "legacy_zero_truncated_linear"
    assert config.ga.top_k == 5
    assert config.ga.verbose is False
    assert config.ga.progress_interval == 10
    assert config.ga.checkpoint_dir is None
    assert config.ga.gpu_rf_params["n_bins"] == 16
    assert config.locking.enabled is False
    assert config.locking.strategy == "top_k_jaccard_medoid"
    assert config.locking.top_k == 3
    assert config.locking.metric_orientation == "larger_is_better"
    assert config.locking.tie_breakers == [
        "higher_locking_score",
        "smaller_feature_count",
        "stable_mask_hash",
    ]
    assert config.scoring.unified_metric is None


def test_archived_config_without_mode_fields_uses_legacy_behavior():
    config = PipelineConfig.from_dict(
        {
            "ga": {"size_penalty_lambda": 0.015},
            "locking": {"enabled": False},
        }
    )

    assert config.ga.fitness_mode == "legacy_zero_truncated_linear"
    assert config.locking.strategy == "top_k_jaccard_medoid"
