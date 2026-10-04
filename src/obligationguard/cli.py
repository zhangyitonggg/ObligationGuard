from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path
import sys

from .config import check_judge, load_config, readiness
from .datasets import load_instances, statistics, validate_benchmark
from .errors import ObligationGuardError


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="obligationguard")
    commands = result.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="import and shuffle the trajectory collection")
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--seed", type=int)
    benchmark = commands.add_parser("import-benchmark", help="import and anonymize the supplied benchmark directories")
    benchmark.add_argument("--input", required=True)
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--identity-env")
    validate = commands.add_parser("validate", help="validate a canonical JSONL dataset")
    validate.add_argument("--input", required=True)
    validate.add_argument("--benchmark", action="store_true")
    check = commands.add_parser("check", help="check data and execution requirements without model calls")
    check.add_argument("--paper", default="configs/paper.toml")
    check.add_argument("--runtime", default="configs/runtime.toml")
    check.add_argument("--scope", choices=("positive", "pipeline"), default="positive")
    for name in ("train", "pipeline", "positive"):
        command = commands.add_parser(name)
        command.add_argument("--paper", default="configs/paper.toml")
        command.add_argument("--runtime", default="configs/runtime.toml")
    prediction = commands.add_parser("predict", help="identify obligations once per instance")
    prediction.add_argument("--input", required=True)
    prediction.add_argument("--output", required=True)
    prediction.add_argument("--runtime", default="configs/runtime.toml")
    prediction.add_argument("--model", required=True)
    evaluation = commands.add_parser("evaluate", help="evaluate predictions with semantic one-to-one matching")
    evaluation.add_argument("--input", required=True)
    evaluation.add_argument("--predictions", required=True)
    evaluation.add_argument("--judgments", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--runtime", default="configs/runtime.toml")
    job = commands.add_parser("jobs", help="run synthesis or annotation prompts")
    job.add_argument("--stage", required=True)
    job.add_argument("--input", required=True)
    job.add_argument("--output", required=True)
    job.add_argument("--runtime", default="configs/runtime.toml")
    job.add_argument("--model", required=True)
    report = commands.add_parser("rq2", help="analyze obligation distance and concurrency")
    report.add_argument("--input", required=True)
    report.add_argument("--model-results", required=True, help="JSON object mapping model names to metric files")
    report.add_argument("--output", required=True)
    experiment = commands.add_parser("experiment", help="run an individual research-question experiment")
    experiment.add_argument("--rq", choices=("rq1", "rq2", "rq3", "rq4"), required=True)
    experiment.add_argument("--paper", default="configs/paper.toml")
    experiment.add_argument("--runtime", default="configs/runtime.toml")
    privacy = commands.add_parser("audit", help="check repository files for private identifiers")
    privacy.add_argument("--root", default=".")
    privacy.add_argument("--identity-env", help="environment variable containing private identity terms separated by semicolons")
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "prepare":
            from .prepare import prepare_collection
            seed = args.seed if args.seed is not None else secrets.randbits(32)
            result = prepare_collection(args.manifest, args.output, seed)
            print(json.dumps({"train": result["train"]["instances"], "validation": result["validation"]["instances"], "seed": seed}, ensure_ascii=False))
        elif args.command == "validate":
            instances = load_instances(args.input)
            print(json.dumps(validate_benchmark(instances) if args.benchmark else statistics(instances), ensure_ascii=False))
        elif args.command == "import-benchmark":
            from .benchmark import import_benchmark
            print(json.dumps(import_benchmark(args.input, args.output, args.identity_env), ensure_ascii=False, indent=2))
        elif args.command == "check":
            checks = readiness(load_config(args.paper), load_config(args.runtime), scope=args.scope)
            print(json.dumps(checks, ensure_ascii=False, indent=2))
            return 0 if all(checks.values()) else 2
        elif args.command == "train":
            from .training import train
            runtime = load_config(args.runtime)
            check_judge(runtime["models"]["judge"])
            train(load_config(args.paper), runtime)
        elif args.command == "pipeline":
            from .pipeline import pipeline
            pipeline(load_config(args.paper), load_config(args.runtime))
        elif args.command == "positive":
            from .experiments import run_positive
            run_positive(load_config(args.paper), load_config(args.runtime))
        elif args.command == "predict":
            from .backends import make_backend
            from .evaluation import predict
            runtime = load_config(args.runtime)
            predict(load_instances(args.input), make_backend(runtime["evaluation_models"][args.model]), args.output)
        elif args.command == "evaluate":
            from .backends import make_backend
            from .evaluation import evaluate
            runtime = load_config(args.runtime)
            check_judge(runtime["models"]["judge"])
            result = evaluate(load_instances(args.input), args.predictions, make_backend(runtime["models"]["judge"]), args.judgments, args.output)
            print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
        elif args.command == "jobs":
            from .backends import make_backend
            from .jobs import run_jobs
            runtime = load_config(args.runtime)
            run_jobs(args.stage, args.input, args.output, make_backend(runtime["models"][args.model]))
        elif args.command == "rq2":
            from .analysis import rq2
            from .io import write_json
            paths = json.loads(Path(args.model_results).read_text(encoding="utf-8"))
            expected_models = {"Claude-Opus-4.8", "DeepSeek-V4.1-Flash", "DeepSeek-V4-Pro", "Kimi-K3", "GLM-5.3"}
            if set(paths) != expected_models:
                raise ObligationGuardError("RQ2 requires the five models specified in Section 6.2")
            results = {model: json.loads(Path(path).read_text(encoding="utf-8")) for model, path in paths.items()}
            write_json(args.output, rq2(load_instances(args.input), results))
        elif args.command == "experiment":
            from .experiments import run_rq1, run_rq2, run_rq3
            from .rq4 import run_rq4
            functions = {"rq1": run_rq1, "rq2": run_rq2, "rq3": run_rq3, "rq4": run_rq4}
            functions[args.rq](load_config(args.paper), load_config(args.runtime))
        elif args.command == "audit":
            from .privacy import audit_repository
            report = audit_repository(args.root, args.identity_env)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["passed"] else 2
        return 0
    except (ObligationGuardError, OSError, KeyError, ImportError) as exc:
        print(f"ObligationGuard: {exc}", file=sys.stderr)
        return 2
