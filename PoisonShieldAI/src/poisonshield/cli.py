from __future__ import annotations
import argparse
import json
from pathlib import Path
from .pipeline import ExperimentSession, read_config, run_sweep
from .data.loaders import synthetic_flows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="poisonshield", description="PoisonShieldAI offline attack/defense benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="configs/quick_experiment.yaml", help="Base YAML config")
    common.add_argument("--dataset-config", default=None, help="Optional YAML section overrides")
    common.add_argument("--data-path", default=None, help="Local CSV file or directory")
    common.add_argument("--dataset", choices=["synthetic", "cicids2017", "toniot", "unswnb15"], default=None)
    common.add_argument("--task", choices=["binary", "multiclass"], default=None)
    common.add_argument("--output-dir", default=None)
    common.add_argument("--victim", default=None, help="all, rf, xgboost, cnn, or comma-separated")
    run = sub.add_parser("run", parents=[common], help="One controlled experiment across selected victims")
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--poison-rate", type=float, default=None)
    run.add_argument("--attack-lambda", type=float, default=None)
    sweep = sub.add_parser("sweep", parents=[common], help="Multi-budget, multi-seed experiments")
    sweep.add_argument("--datasets", nargs="+", choices=["synthetic", "cicids2017", "toniot", "unswnb15"])
    sweep.add_argument("--rates", type=float, nargs="+", default=[0.005, 0.01, 0.05, 0.10, 0.20])
    sweep.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    demo = sub.add_parser("make-demo-csv", help="Write synthetic demo CSV (NEVER a real dataset)")
    demo.add_argument("--output", default="examples/synthetic_flows.csv")
    demo.add_argument("--rows", type=int, default=850)
    args = parser.parse_args(argv)
    if args.command == "make-demo-csv":
        filename = Path(args.output)
        filename.parent.mkdir(parents=True, exist_ok=True)
        synthetic_flows(n=args.rows).to_csv(filename, index=False)
        print(f"Synthetic demonstration CSV: {filename}")
        return 0
    cfg = read_config(args.config, args.dataset_config)
    ex = cfg["experiment"]
    for arg, key in [("dataset", "dataset"), ("data_path", "data_path"),
                     ("task", "task"), ("output_dir", "output_dir"), ("victim", "victim")]:
        value = getattr(args, arg)
        if value is not None:
            ex[key] = value
    if args.command == "run":
        if args.attack_lambda is not None:
            cfg["attack"]["lambda_weight"] = args.attack_lambda
        runner = ExperimentSession(cfg)
        results = runner.run(seed=args.seed, rate=args.poison_rate, victim=args.victim)
        print(results[["dataset", "victim", "method", "status", "accuracy", "macro_f1"]].to_string(index=False))
        print(f"Saved under {runner.output.resolve()}")
    else:
        datasets = args.datasets or [ex["dataset"]]
        results = run_sweep(cfg, datasets, ex.get("victim", "all"), args.seeds, args.rates, args.data_path)
        print(f"Completed: {len(results)} method/model result rows")
        print(f"Results: {Path(ex['output_dir']).resolve() / 'all_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
