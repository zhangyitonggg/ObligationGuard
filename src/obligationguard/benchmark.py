from __future__ import annotations

import json
import os
from pathlib import Path

from .datasets import load_instances, validate_benchmark
from .errors import DataError
from .identifiers import opaque_instance_id
from .io import file_sha256, write_json, write_jsonl
from .privacy import inspect_text, remove_private_record
from .schema import Instance


DOMAINS = {
    "SWE-BenchPro": "issue_resolution",
    "SWE-Bench Pro": "issue_resolution",
    "FeatureBench": "feature_development",
    "Terminal-Bench": "terminal_operation",
    "Terminal-Bench2.0": "terminal_operation",
    "Terminal-Bench 2.0": "terminal_operation",
    "TerminalWorld": "terminal_operation",
}


def import_benchmark(source: str | Path, output: str | Path, identity_env: str | None = None) -> dict:
    """Import supplied annotations while deleting private identifiers from keys and values."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.is_relative_to(source):
        raise DataError("benchmark output must be outside the supplied source directory")
    terms = tuple(term.strip() for term in os.getenv(identity_env, "").split(";") if term.strip()) if identity_env else ()
    examples = []
    for split in ("positive", "negative"):
        folders = sorted(path for path in (source / split).iterdir() if path.is_dir())
        if len(folders) != 120:
            raise DataError(f"{split}: expected 120 sample directories")
        for folder in folders:
            def read(name):
                return remove_private_record(json.loads((folder / name).read_text(encoding="utf-8-sig")), terms)
            guard, truth, provenance = read("guard_input.json"), read("ground_truth.json"), read("provenance.json")
            messages = guard["messages"]
            task = next((item.get("content") for item in messages if item.get("role") == "user"), None)
            obligations = truth["obligations"]
            if type(truth["obligation_count"]) is not int or truth["obligation_count"] != len(obligations):
                raise DataError("annotation count does not match the supplied obligations")
            if bool(obligations) != (split == "positive"):
                raise DataError("sample directory and annotation label disagree")
            domain = DOMAINS.get(provenance["source_benchmark"])
            if domain is None:
                raise DataError("unsupported benchmark source")
            example = Instance.from_dict({"instance_id": opaque_instance_id(truth["task_id"], "benchmark"), "task": task, "trajectory": messages, "obligations": obligations, "task_domain": domain})
            if inspect_text(json.dumps(example.to_dict(), ensure_ascii=False), terms):
                raise DataError("benchmark privacy audit did not pass")
            examples.append(example)
    validate_benchmark(examples)
    if len({item.instance_id for item in examples}) != len(examples):
        raise DataError("benchmark contains duplicate sample identifiers")
    files = {}
    for name, items in (
        ("obligationbench.jsonl", examples),
        ("01_positive/obligationbench_positive.jsonl", [x for x in examples if x.obligations]),
        ("02_negative/obligationbench_negative.jsonl", [x for x in examples if not x.obligations]),
    ):
        write_jsonl(output / name, (item.to_dict() for item in items))
        files[name] = {"instances": len(items), "sha256": file_sha256(output / name)}
    report = {"instances": 240, "positive_instances": 120, "negative_instances": 120, "obligations": 339, "identifier_format": "opaque_v1", "files": files}
    write_json(output / "benchmark_manifest.json", report)
    return report


def positive_instances(runtime: dict) -> list[Instance]:
    instances = load_instances(runtime["paths"]["benchmark"])
    validate_benchmark(instances)
    return [instance for instance in instances if instance.obligations]
