from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import random
import subprocess
import sys

from .ablations import model_settings, nested_training_sets
from .analysis import action_observation_steps, rq2, table_csv
from .backends import make_backend
from .benchmark import positive_instances
from .config import check_judge, readiness
from .datasets import load_instances, validate_benchmark, validate_training_split
from .errors import ConfigurationError, DataError
from .identifiers import opaque_instance_id
from .evaluation import evaluate, predict
from .io import file_sha256, fingerprint, read_jsonl, write_json, write_jsonl
from .jobs import run_jobs
from .schema import Instance, Obligation


def evaluate_model(instances, model_config: dict, runtime: dict, directory: Path) -> dict:
    check_judge(runtime["models"]["judge"])
    backend = make_backend(model_config)
    try:
        predict(instances, backend, directory / "predictions.jsonl")
    finally:
        if hasattr(backend, "close"):
            backend.close()
    del backend
    if model_config["backend"] == "local":
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    result = evaluate(instances, directory / "predictions.jsonl", make_backend(runtime["models"]["judge"]), directory / "judgments.jsonl", directory / "metrics.json")
    if all(instance.obligations for instance in instances):
        result["evaluation_scope"] = "positive"
        result["metrics"].pop("ca_negative", None)
        result["metrics"]["counts"].pop("negative_instances", None)
        write_json(directory / "metrics.json", result)
    return result


def launch_training(paper: dict, runtime: dict, directory: Path) -> None:
    identity = fingerprint({"paper": paper, "runtime": runtime, "train": file_sha256(runtime["paths"]["train"]), "validation": file_sha256(runtime["paths"]["validation"])})
    marker = directory / "training_complete.json"
    if marker.is_file():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous["fingerprint"] != identity or not (Path(runtime["paths"]["checkpoints"]) / "best" / "config.json").is_file():
            raise ConfigurationError("saved training run has different settings or an unavailable checkpoint")
        return
    write_json(directory / "paper.json", paper)
    write_json(directory / "runtime.json", runtime)
    command = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(paper["training"]["world_size"]), "-m", "obligationguard", "train", "--paper", str(directory / "paper.json"), "--runtime", str(directory / "runtime.json")]
    subprocess.run(command, check=True)
    write_json(marker, {"fingerprint": identity})


def run_rq1(paper: dict, runtime: dict) -> dict:
    instances = positive_instances(runtime)
    output = Path(runtime["paths"]["results"]) / "positive"
    results = {name: evaluate_model(instances, config, runtime, output / name) for name, config in runtime["evaluation_models"].items()}
    table_csv([{"model": name, "metrics": value["metrics"]} for name, value in results.items()], output / "table.csv")
    return results


def run_positive(paper: dict, runtime: dict) -> dict:
    """One-click entry point: evaluate the positive cohort and return immediately."""
    check_judge(runtime["models"]["judge"])
    checks = readiness(paper, runtime, scope="positive")
    missing = [name for name, valid in checks.items() if not valid]
    if missing:
        raise ConfigurationError("prepare these requirements before starting: " + ", ".join(missing))
    result = run_rq1(paper, runtime)
    write_json(Path(runtime["paths"]["results"]) / "positive" / "summary.json", {"evaluation_scope": "positive", "instances_per_model": 120, "models": {name: row["metrics"] for name, row in result.items()}})
    return result


def annotate_creation_steps(instances, runtime: dict, directory: Path):
    instances = [action_observation_steps(instance) for instance in instances]
    jobs = []
    for instance in instances:
        for index, obligation in enumerate(instance.obligations):
            if obligation.creation_step is None:
                jobs.append({"job_id": f"{instance.instance_id}-O{index + 1}", "inputs": {"USER_TASK": instance.task, "TRAJECTORY": list(instance.trajectory), "GROUND_TRUTH_OBLIGATION": obligation.required_safety_action}})
    if not jobs:
        return instances
    write_jsonl(directory / "jobs.jsonl", jobs)
    run_jobs("creation_step_annotation", directory / "jobs.jsonl", directory / "annotations.jsonl", make_backend(runtime["models"].get("annotator", runtime["models"]["judge"])))
    annotated = {row["job_id"]: row["output"] for row in read_jsonl(directory / "annotations.jsonl")}
    result = []
    for instance in instances:
        obligations = []
        for index, obligation in enumerate(instance.obligations):
            identifier = f"{instance.instance_id}-O{index + 1}"
            obligations.append(replace(obligation, creation_step=annotated[identifier]["creation_step"]) if obligation.creation_step is None else obligation)
        result.append(replace(instance, obligations=tuple(obligations)))
    write_jsonl(directory / "annotated_benchmark.jsonl", (x.to_dict() for x in result))
    return result


def run_rq2(paper: dict, runtime: dict, results: dict | None = None) -> dict:
    instances = positive_instances(runtime)
    output = Path(runtime["paths"]["results"])
    names = paper["rq2"]["models"]
    if results is None:
        results = {name: evaluate_model(instances, runtime["evaluation_models"][name], runtime, output / "positive" / name) for name in names}
    instances = annotate_creation_steps(instances, runtime, output / "rq2" / "creation_steps")
    report = rq2(instances, {name: results[name] for name in names})
    write_json(output / "rq2" / "metrics.json", report)
    from .plots import plot_rq2
    plot_rq2(report, output / "rq2" / "performance.pdf")
    return report


def direct_training_set(runtime: dict, output: Path, seed: int) -> Path:
    provided = runtime["paths"].get("direct_train")
    if provided:
        instances = load_instances(provided)
        if len(instances) != 40000:
            raise DataError("direct synthesis comparison requires 40,000 training examples")
        validation = runtime["paths"].get("direct_validation")
        if validation:
            validate_training_split(instances, load_instances(validation))
        return Path(provided)
    source = runtime["paths"].get("without_templates")
    if source:
        from .direct_data import materialize_without_templates
        return materialize_without_templates(source, output / "without_templates", seed)
    config = runtime["models"].get("generator")
    if not config or config["model"] != "gpt-5.6-sol":
        raise ConfigurationError("direct synthesis requires the GPT-5.6-Sol generator")
    destination = output / "direct_synthesis"
    write_jsonl(destination / "jobs.jsonl", ({"job_id": f"direct-{i:05d}", "inputs": {}} for i in range(1, 40001)))
    run_jobs("direct_synthesis", destination / "jobs.jsonl", destination / "outputs.jsonl", make_backend(config))
    instances = []
    for row in read_jsonl(destination / "outputs.jsonl"):
        generated = row["output"]
        obligations = [Obligation(action).to_dict(i) for i, action in enumerate(generated["ground_truth_obligation_set"], 1)]
        instances.append(Instance.from_dict({"instance_id": opaque_instance_id(row["job_id"], "direct_synthesis"), "task": generated["user_task"], "trajectory": generated["trajectory"], "obligations": obligations}))
    if len(instances) != 40000:
        raise DataError("direct synthesis did not produce 40,000 unique examples")
    random.Random(seed).shuffle(instances)
    path = destination / "train.jsonl"
    write_jsonl(path, (x.to_dict() for x in instances))
    return path


def run_rq3(paper: dict, runtime: dict) -> list[dict]:
    output = Path(runtime["paths"]["results"]) / "rq3"
    instances = positive_instances(runtime)
    subsets = nested_training_sets(runtime["paths"]["train"], output / "training_sets")
    manifest = Path(runtime["paths"]["train"]).parent / "manifest.json"
    seed = runtime["training_runtime"].get("seed")
    if seed is None:
        seed = json.loads(manifest.read_text(encoding="utf-8"))["seed"]
    direct = direct_training_set(runtime, output, seed)
    rows = []
    for run in model_settings(paper):
        name, settings = run["name"], run["settings"]
        variant_runtime = deepcopy(runtime)
        variant_runtime["training_runtime"]["seed"] = seed
        backbone = settings["training"]["model"]
        variant_runtime["training_runtime"]["model_revision"] = runtime["backbone_revisions"][backbone]
        if "Qwen3" not in backbone:
            variant_runtime["training_runtime"]["chat_template_kwargs"] = {}
        variant_runtime["paths"]["train"] = str(direct if "direct-synthesis" in name else subsets[settings["training"]["train_examples"]])
        if "direct-synthesis" in name:
            validation = runtime["paths"].get("direct_validation") or (str(direct.parent / "validation.jsonl") if runtime["paths"].get("without_templates") and not runtime["paths"].get("direct_train") else None)
            if validation:
                variant_runtime["paths"]["validation"] = validation
        variant_runtime["paths"]["checkpoints"] = str(Path(runtime["paths"]["checkpoints"]) / "rq3" / name)
        if name == "Qwen3-8B-40000":
            variant_runtime["paths"]["checkpoints"] = runtime["paths"]["checkpoints"]
        else:
            launch_training(settings, variant_runtime, output / name / "training")
        configuration = {"backend": "local", "model": str(Path(variant_runtime["paths"]["checkpoints"]) / "best"), "max_output_tokens": runtime["training_runtime"]["max_output_tokens"], "max_context_length": paper["evaluation"]["max_context_length"], "chat_template_kwargs": variant_runtime["training_runtime"]["chat_template_kwargs"]}
        evaluated = evaluate_model(instances, configuration, runtime, output / name)
        rows.append({"model": name, "metrics": evaluated["metrics"]})
        if settings["training"]["train_examples"] == 40000 and "direct-synthesis" not in name:
            original_config = {**configuration, "model": backbone, "revision": runtime["backbone_revisions"][backbone]}
            original = evaluate_model(instances, original_config, runtime, output / (backbone.rsplit("/", 1)[-1] + "-original"))
            rows.append({"model": backbone.rsplit("/", 1)[-1] + "-original", "metrics": original["metrics"]})
    table_csv(rows, output / "table.csv")
    write_json(output / "metrics.json", rows)
    from .plots import plot_rq3
    plot_rq3(rows, output / "training_data.pdf")
    return rows
