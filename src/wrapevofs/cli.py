"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wrapevofs.artifacts import save_pipeline_result
from wrapevofs.config import PipelineConfig
from wrapevofs.pipeline import WrapEvoPipeline
from wrapevofs._version import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wrapevofs")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run preprocessing through RFECV target discovery.")
    run.add_argument("--csv", required=True, help="Input CSV containing features and target.")
    run.add_argument("--target", required=True, help="Target column name.")
    run.add_argument("--out", required=True, help="Output directory.")
    run.add_argument("--config", help="YAML config path.")
    run.add_argument("--ratio", help="Override split ratio, for example 7:3, 6:4, 8:2.")
    run.add_argument("--impute", choices=["median", "mean", "zero"], help="Override imputation strategy.")
    run.add_argument(
        "--rfecv-max-features",
        type=int,
        help="Override RFECV compact cap for all wrapper branches.",
    )
    run.add_argument("--xgb-rfecv-max", type=int, help="RFECV compact cap for the xgboost branch.")
    run.add_argument("--svm-rfecv-max", type=int, help="RFECV compact cap for the svm_l1 branch.")
    run.add_argument("--boruta-rfecv-max", type=int, help="RFECV compact cap for the boruta_rf branch.")
    run.add_argument("--methods", help="Comma-separated methods: svm_l1,xgboost,boruta_rf.")
    run.add_argument("--drop-columns", help="Comma-separated non-feature columns to drop.")
    run.add_argument(
        "--ga-backend",
        choices=["auto", "cpu", "gpu"],
        help="Override GA Random Forest evaluator backend.",
    )
    run.add_argument(
        "--ga-fitness-mode",
        choices=["legacy_zero_truncated_linear", "untruncated_shifted_linear"],
        help="GA ranking objective; archived configurations use the legacy mode.",
    )
    run.add_argument(
        "--locking-strategy",
        choices=["top_k_jaccard_medoid", "regret_constrained_medoid"],
        help="Enable development-only representative-run locking with this strategy.",
    )
    run.add_argument(
        "--locking-tolerance-mode",
        choices=["absolute", "relative", "best_run_se_scaled", "one_se"],
        help="Regret eligibility mode for regret-constrained locking.",
    )
    run.add_argument("--regret-tolerance", type=float, help="Development-CV regret tolerance.")
    run.add_argument(
        "--minimum-pool-size",
        type=int,
        help="Compatibility option; strict regret-constrained locking requires 1.",
    )
    run.add_argument(
        "--unified-metric",
        choices=["accuracy", "balanced_accuracy", "roc_auc", "macro_ovr_auroc"],
        help="Apply one compatible metric across RFECV, GA, and locking.",
    )
    run.add_argument("--run-ga", action="store_true", help="Continue from RFECV target discovery into GA-RF.")
    run.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume GA state from each branch checkpoint directory. Fails on "
            "missing, corrupt, version-, configuration-, feature-, or input-mismatched state."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        config = PipelineConfig.from_yaml(args.config) if args.config else PipelineConfig()
        if args.ratio:
            config.split.ratio = args.ratio
        if args.impute:
            config.preprocessing.impute_strategy = args.impute
        if args.rfecv_max_features is not None:
            config.rfecv.max_features_to_consider = args.rfecv_max_features
            for method in config.rfecv.method_max_features_to_consider:
                config.rfecv.method_max_features_to_consider[method] = args.rfecv_max_features
        if args.xgb_rfecv_max is not None:
            config.rfecv.method_max_features_to_consider["xgboost"] = args.xgb_rfecv_max
        if args.svm_rfecv_max is not None:
            config.rfecv.method_max_features_to_consider["svm_l1"] = args.svm_rfecv_max
        if args.boruta_rfecv_max is not None:
            config.rfecv.method_max_features_to_consider["boruta_rf"] = args.boruta_rfecv_max
        if args.ga_backend:
            config.ga.backend = args.ga_backend
        if args.ga_fitness_mode:
            config.ga.fitness_mode = args.ga_fitness_mode
        if args.locking_strategy:
            config.locking.enabled = True
            config.locking.strategy = args.locking_strategy
        if args.locking_tolerance_mode:
            config.locking.tolerance_mode = args.locking_tolerance_mode
        if args.regret_tolerance is not None:
            config.locking.regret_tolerance = args.regret_tolerance
        if args.minimum_pool_size is not None:
            config.locking.minimum_pool_size = args.minimum_pool_size
        if args.unified_metric:
            config.scoring.unified_metric = args.unified_metric
        if args.resume:
            if not args.run_ga:
                raise SystemExit("--resume requires --run-ga.")
            config.ga.resume_from_checkpoint = True
        methods = args.methods.split(",") if args.methods else None
        drop_columns = args.drop_columns.split(",") if args.drop_columns else None

        df = pd.read_csv(args.csv)
        output_dir = Path(args.out)
        if args.run_ga:
            config.ga.checkpoint_dir = str(output_dir)
        pipeline = WrapEvoPipeline(config)
        if args.run_ga:
            result = pipeline.run_until_rfecv(
                df=df,
                target_column=args.target,
                drop_columns=drop_columns,
                methods=methods,
            )
            save_pipeline_result(result, output_dir)
            result.ga_results = pipeline.run_ga(
                result.prepared,
                result.first_stage,
                result.rfecv_targets,
            )
            result.locking_results = pipeline.run_locking(
                result.prepared,
                result.first_stage,
                result.ga_results,
            )
            result.warnings = list(pipeline.warnings)
        else:
            result = pipeline.run_until_rfecv(
                df=df,
                target_column=args.target,
                drop_columns=drop_columns,
                methods=methods,
            )
        save_pipeline_result(result, output_dir)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
