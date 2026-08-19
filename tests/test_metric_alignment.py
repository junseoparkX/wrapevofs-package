import pandas as pd
import pytest

from wrapevofs.config import PipelineConfig
from wrapevofs.pipeline import WrapEvoPipeline


def test_unified_macro_ovr_metric_resolves_for_binary_and_multiclass():
    pipeline = WrapEvoPipeline(
        PipelineConfig.from_dict(
            {"scoring": {"unified_metric": "macro_ovr_auroc"}}
        )
    )

    assert pipeline._resolved_stage_metrics(pd.Series([0, 1, 0, 1])) == (
        "roc_auc",
        "roc_auc",
        "roc_auc",
    )
    assert pipeline._resolved_stage_metrics(pd.Series([0, 1, 2, 0, 1, 2])) == (
        "roc_auc_ovr",
        "roc_auc_ovr",
        "roc_auc_ovr",
    )


def test_existing_metric_mismatch_is_reported_without_changing_defaults():
    pipeline = WrapEvoPipeline(PipelineConfig())
    pipeline._audit_metric_alignment(pd.Series([0, 1, 0, 1]))

    assert pipeline.config.rfecv.scoring == "auto"
    assert pipeline.config.ga.fitness_metric == "accuracy"
    assert len(pipeline.warnings) == 1
    assert "rfecv=roc_auc" in pipeline.warnings[0]
    assert "ga=accuracy" in pipeline.warnings[0]


def test_invalid_unified_metric_fails_explicitly():
    pipeline = WrapEvoPipeline(
        PipelineConfig.from_dict({"scoring": {"unified_metric": "f1"}})
    )

    with pytest.raises(ValueError, match="unified_metric"):
        pipeline._resolved_stage_metrics(pd.Series([0, 1]))
