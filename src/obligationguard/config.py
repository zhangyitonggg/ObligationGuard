from __future__ import annotations

import os
import json
from pathlib import Path
import tomllib

from .errors import ConfigurationError


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    if path.suffix == ".json":
        config = json.loads(path.read_text(encoding="utf-8"))
    else:
        with path.open("rb") as stream:
            config = tomllib.load(stream)
    repository = Path(__file__).resolve().parents[2]
    if "models_config" in config:
        registry = Path(os.path.expandvars(config.pop("models_config").replace("${REPOSITORY}", str(repository))))
        models = json.loads(registry.read_text(encoding="utf-8"))
        def merge(base, changes):
            for key, value in changes.items():
                if isinstance(value, dict) and isinstance(base.get(key), dict):
                    merge(base[key], value)
                else:
                    base[key] = value
            return base
        config = merge(models, config)
    def expand(value):
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, str):
            return os.path.expandvars(value.replace("${REPOSITORY}", str(repository)))
        return value
    config = expand(config)
    deepspeed = config.get("training", {}).get("deepspeed")
    if isinstance(deepspeed, str) and not Path(deepspeed).is_absolute():
        config["training"]["deepspeed"] = str(repository / deepspeed)
    return config


def check_judge(config: dict) -> None:
    if config.get("model") != "gpt-5.6-sol":
        raise ConfigurationError("semantic matching requires the GPT-5.6-Sol judge")
    if config.get("parameters", {}).get("temperature") != 0:
        raise ConfigurationError("semantic matching requires judge temperature 0")


def readiness(paper: dict, runtime: dict, *, scope: str = "pipeline") -> dict:
    import importlib.util
    from .datasets import load_instances, validate_benchmark, validate_training_split
    paths = runtime.get("paths", {})
    checks = {}
    for name in (("benchmark",) if scope == "positive" else ("train", "validation", "benchmark")):
        path = paths.get(name)
        checks[name] = bool(path and Path(path).is_file())
    if checks.get("train") and checks.get("validation"):
        validate_training_split(load_instances(paths["train"]), load_instances(paths["validation"]), paper["training"]["train_examples"], paper["training"]["validation_examples"])
    if checks["benchmark"]:
        validate_benchmark(load_instances(paths["benchmark"]))
    if scope != "positive":
        checks["training_packages"] = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "accelerate", "deepspeed", "flash_attn"))
    elif any(config.get("backend") == "local" for config in runtime.get("evaluation_models", {}).values()):
        checks["local_inference_packages"] = all(importlib.util.find_spec(name) for name in ("torch", "transformers"))
        checkpoint = runtime.get("evaluation_models", {}).get("ObligationGuard", {}).get("model", "")
        checks["guard_checkpoint"] = bool(checkpoint and "$" not in checkpoint and (Path(checkpoint) / "config.json").is_file() and any(Path(checkpoint).glob("*.safetensors")))
    if scope == "positive" and any(config.get("backend") == "chat" and config.get("token_count_backend") not in {"moonshot", "vllm"} for config in runtime.get("evaluation_models", {}).values()):
        checks["tokenizer_packages"] = bool(importlib.util.find_spec("transformers"))
    checks["judge_credentials"] = bool(os.getenv(runtime.get("models", {}).get("judge", {}).get("api_key_env", "OPENAI_API_KEY")))
    checks["run_root"] = bool(os.getenv("OBLIGATIONGUARD_RUN_ROOT"))
    from .catalog import EVALUATED_MODELS
    checks["evaluation_models"] = set(runtime.get("evaluation_models", {})) == set(EVALUATED_MODELS)
    for configuration in runtime.get("evaluation_models", {}).values():
        variable = configuration.get("api_key_env")
        if variable:
            checks[variable] = bool(os.getenv(variable))
        if configuration.get("backend") == "managed_chat":
            executable = configuration.get("server", {}).get("python", "")
            checks["inference_python"] = bool(executable and "$" not in executable and Path(executable).is_file())
        if "base_url" in configuration:
            checks["endpoint_" + configuration["model"]] = "$" not in configuration["base_url"]
    stages = runtime.get("pipeline", {}).get("stages", ["rq1"])
    if scope != "positive" and "rq4" in stages:
        settings = runtime.get("susvibes", {})
        checks["susvibes_tasks"] = bool(settings.get("tasks") and Path(settings["tasks"]).is_file())
        checks["mini_config"] = bool(Path(settings["mini_config"]).is_file()) if settings.get("mini_config") else bool(importlib.util.find_spec("minisweagent"))
        checks["susvibes_checkout"] = bool(settings.get("checkout") and (Path(settings["checkout"]) / "susvibes" / "eval" / "core.py").is_file())
        checks["mini_swe_agent"] = bool(importlib.util.find_spec("minisweagent"))
        import shutil
        checks["docker"] = bool(shutil.which("docker"))
    return checks
