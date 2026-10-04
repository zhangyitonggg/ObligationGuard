from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .datasets import load_instances
from .errors import DataError
from .io import write_jsonl


def nested_training_sets(train_path: str | Path, output: str | Path) -> dict:
    instances = load_instances(train_path)
    if len(instances) != 40000:
        raise DataError("data-scaling ablations require a shuffled 40,000-example training set")
    paths = {}
    for size in (5000, 10000, 20000, 40000):
        path = Path(output) / f"train_{size}.jsonl"
        write_jsonl(path, (instance.to_dict() for instance in instances[:size]))
        paths[size] = str(path)
    return paths


def model_settings(paper: dict) -> list[dict]:
    runs = []
    for model in paper["rq3"]["backbones"]:
        settings = deepcopy(paper)
        settings["training"]["model"] = model
        runs.append({"name": model.rsplit("/", 1)[-1] + "-40000", "settings": settings})
    for size in (5000, 10000, 20000):
        settings = deepcopy(paper)
        settings["training"]["train_examples"] = size
        runs.append({"name": f"Qwen3-8B-{size}", "settings": settings})
    settings = deepcopy(paper)
    runs.append({"name": "Qwen3-8B-direct-synthesis-40000", "settings": settings})
    return runs
