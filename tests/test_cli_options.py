from wrapevofs.cli import build_parser


def test_cli_exposes_upgraded_ga_locking_and_metric_options():
    args = build_parser().parse_args(
        [
            "run",
            "--csv",
            "input.csv",
            "--target",
            "target",
            "--out",
            "run",
            "--ga-fitness-mode",
            "untruncated_shifted_linear",
            "--locking-strategy",
            "regret_constrained_medoid",
            "--locking-tolerance-mode",
            "relative",
            "--regret-tolerance",
            "0.02",
            "--minimum-pool-size",
            "1",
            "--unified-metric",
            "macro_ovr_auroc",
            "--run-ga",
            "--resume",
        ]
    )

    assert args.ga_fitness_mode == "untruncated_shifted_linear"
    assert args.locking_strategy == "regret_constrained_medoid"
    assert args.locking_tolerance_mode == "relative"
    assert args.regret_tolerance == 0.02
    assert args.minimum_pool_size == 1
    assert args.unified_metric == "macro_ovr_auroc"
    assert args.run_ga is True
    assert args.resume is True
