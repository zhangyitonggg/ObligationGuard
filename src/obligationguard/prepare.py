from __future__ import annotations

import json
import random
from pathlib import Path

from .datasets import statistics, validate_training_split
from .errors import DataError
from .identifiers import opaque_instance_id
from .io import file_sha256, read_jsonl, write_json, write_jsonl
from .privacy import remove_private_record
from .schema import Instance


def import_trajectory(row: dict) -> Instance:
    """Map source fields and remove private contact addresses and user directories."""
    row = remove_private_record(row)
    task_input = row.get("input")
    target = row.get("target")
    if not isinstance(task_input, dict) or not isinstance(target, dict):
        raise DataError("trajectory requires input and target objects")
    labels = target.get("unfulfilled_obligations")
    if not isinstance(labels, list):
        raise DataError("target.unfulfilled_obligations must be an array")
    obligations = []
    for number, label in enumerate(labels, 1):
        obligations.append({
            "id": f"O{number}",
            "required_safety_action": label["required_action"],
            "safety_consequence": label.get("security_consequence_if_omitted", ""),
            "evidence": label.get("evidence", []),
        })
    scenario = row.get("multi_group_id") or row.get("seed_id")
    return Instance.from_dict({
        "instance_id": opaque_instance_id(row["trajectory_id"], "training"),
        "scenario_id": scenario,
        "source_scenario_ids": row.get("source_seed_ids", []),
        "task": task_input["task"],
        "trajectory": task_input["trajectory"],
        "obligations": obligations,
        "metadata": {"original_target": target, **{key: value for key, value in row.items() if key not in {"input", "target"}}},
    })


def prepare_collection(manifest_path: str | Path, output: str | Path, seed: int) -> dict:
    manifest_path, output = Path(manifest_path), Path(output)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise DataError("seed must be an integer between 0 and 2**32-1")
    if (output / "manifest.json").exists():
        raise DataError(f"{output}: prepared data already exists; select a new output directory")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    specifications = [
        ("single_obligation_trajectory_file", "single_obligation_records", "trajectory_sha256"),
        ("multi_obligation_trajectory_file", "multi_obligation_records", "multi_trajectory_sha256"),
    ]
    records = []
    source_metadata = []
    for file_key, count_key, hash_key in specifications:
        filename = manifest.get(file_key)
        if not isinstance(filename, str):
            raise DataError(f"collection manifest is missing {file_key}")
        source = (manifest_path.parent / filename).resolve()
        digest = file_sha256(source)
        expected_digest = manifest.get(hash_key)
        if expected_digest is not None and digest != expected_digest:
            raise DataError(f"{source.name}: SHA-256 mismatch")
        imported = [import_trajectory(row) for row in read_jsonl(source)]
        if len(imported) != manifest.get(count_key):
            raise DataError(f"{source.name}: record count does not match collection manifest")
        records.extend(imported)
        source_metadata.append({"file": source.name, "sha256": digest, "records": len(imported)})
    if len(records) != manifest.get("total_trajectory_records"):
        raise DataError("collection total does not match manifest")
    if len({x.instance_id for x in records}) != len(records):
        raise DataError("collection contains duplicate trajectory identifiers")
    randomizer = random.Random(seed)
    randomizer.shuffle(records)
    validation, train = records[:2000], records[2000:]
    randomizer.shuffle(train)
    randomizer.shuffle(validation)
    validate_training_split(train, validation)
    write_jsonl(output / "train.jsonl", (x.to_dict() for x in train))
    write_jsonl(output / "validation.jsonl", (x.to_dict() for x in validation))
    write_json(output / "split_ids.json", {"train": [x.instance_id for x in train], "validation": [x.instance_id for x in validation]})
    result = {
        "seed": seed,
        "split_unit": "trajectory",
        "identifier_format": "opaque_v1",
        "sources": source_metadata,
        "train": {"file": "train.jsonl", "sha256": file_sha256(output / "train.jsonl"), **statistics(train)},
        "validation": {"file": "validation.jsonl", "sha256": file_sha256(output / "validation.jsonl"), **statistics(validation)},
    }
    write_json(output / "manifest.json", result)
    return result
