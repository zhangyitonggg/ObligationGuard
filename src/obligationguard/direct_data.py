from __future__ import annotations

import json
import os
from pathlib import Path
import random

from .datasets import statistics, validate_training_split
from .errors import DataError
from .identifiers import opaque_instance_id
from .io import file_sha256, fingerprint, read_jsonl, write_json, write_jsonl
from .privacy import inspect_text, remove_private_record
from .schema import Instance, Obligation


def direct_instance(row: dict, number: int) -> Instance:
    labels = row.get("ground_truth_obligation_set")
    if not isinstance(labels, list) or any(not isinstance(label, str) or not label.strip() for label in labels):
        raise DataError("direct synthesis requires a ground-truth action string array")
    return Instance.from_dict({"instance_id": opaque_instance_id(f"without-templates-{number:05d}", "direct_synthesis"), "task": row.get("user_task"), "trajectory": row.get("trajectory"), "obligations": [Obligation(label).to_dict(index) for index, label in enumerate(labels, 1)]})


def import_without_templates(source: str | Path, destination: str | Path, identity_env: str | None = None) -> dict:
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise DataError("the supplied source file must remain untouched")
    terms = tuple(term.strip() for term in os.getenv(identity_env, "").split(";") if term.strip()) if identity_env else ()
    count, obligations = 0, 0
    def records():
        nonlocal count, obligations
        for count, original in enumerate(read_jsonl(source), 1):
            row = remove_private_record(original, terms)
            sample = direct_instance(row, count)
            if inspect_text(json.dumps(row, ensure_ascii=False), terms):
                raise DataError("without-templates privacy audit did not pass")
            if len(sample.obligations) != len(original["ground_truth_obligation_set"]):
                raise DataError("privacy removal must preserve the annotation count")
            obligations += len(sample.obligations)
            yield row
        if count != 42000:
            raise DataError("without-templates source must contain 42,000 examples")
    write_jsonl(destination, records())
    report = {"file": destination.name, "format": "direct_synthesis", "records": count, "obligations": obligations, "source_sha256": file_sha256(source), "sha256": file_sha256(destination)}
    write_json(destination.parent / "manifest.json", report)
    return report


def materialize_without_templates(source: str | Path, directory: str | Path, seed: int, expected_train: int = 40000, expected_validation: int = 2000) -> Path:
    """Shuffle and split the supplied direct-synthesis source by trajectory."""
    source, directory = Path(source), Path(directory)
    identity = fingerprint({"source_sha256": file_sha256(source), "seed": seed, "train": expected_train, "validation": expected_validation, "identifier_format": "opaque_v1"})
    manifest_path = directory / "manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["fingerprint"] != identity or any(file_sha256(directory / name) != digest for name, digest in previous["sha256"].items()):
            raise DataError("saved without-templates split uses different data or settings")
        return directory / "train.jsonl"
    samples = [direct_instance(row, index) for index, row in enumerate(read_jsonl(source), 1)]
    if len(samples) != expected_train + expected_validation:
        raise DataError("without-templates source does not match the training and validation sizes")
    randomizer = random.Random(seed)
    randomizer.shuffle(samples)
    validation, train = samples[:expected_validation], samples[expected_validation:]
    randomizer.shuffle(train)
    randomizer.shuffle(validation)
    validate_training_split(train, validation, expected_train, expected_validation)
    write_jsonl(directory / "train.jsonl", (sample.to_dict() for sample in train))
    write_jsonl(directory / "validation.jsonl", (sample.to_dict() for sample in validation))
    write_json(directory / "split_ids.json", {"train": [sample.instance_id for sample in train], "validation": [sample.instance_id for sample in validation]})
    hashes = {name: file_sha256(directory / name) for name in ("train.jsonl", "validation.jsonl", "split_ids.json")}
    write_json(manifest_path, {
        "fingerprint": identity,
        "seed": seed,
        "split_unit": "trajectory",
        "identifier_format": "opaque_v1",
        "sources": [{"file": source.name, "sha256": file_sha256(source), "records": len(samples)}],
        "train": {"file": "train.jsonl", "sha256": hashes["train.jsonl"], **statistics(train)},
        "validation": {"file": "validation.jsonl", "sha256": hashes["validation.jsonl"], **statistics(validation)},
        "sha256": hashes,
    })
    return directory / "train.jsonl"
