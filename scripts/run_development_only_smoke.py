"""Run the synthetic development-only regret-locking example."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wrapevofs import PipelineConfig, WrapEvoPipeline
from wrapevofs.artifacts import save_pipeline_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/synthetic_regret_smoke")
    args = parser.parse_args()
    data = pd.read_csv("examples/sample_radiomics.csv")
    config = PipelineConfig.from_yaml("configs/synthetic_regret_smoke.yaml")
    output = Path(args.out)
    config.ga.checkpoint_dir = str(output)
    result = WrapEvoPipeline(config).run_full(
        data,
        target_column="MGMT_binary",
        drop_columns=["patient_id"],
        methods=["svm_l1"],
    )
    save_pipeline_result(result, output)
    print(output / "final_feature_sets.npy")


if __name__ == "__main__":
    main()
