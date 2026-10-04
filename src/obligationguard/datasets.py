from __future__ import annotations

from collections import Counter
from pathlib import Path

from .errors import DataError
from .io import read_jsonl
from .schema import Instance


def load_instances(path: str | Path) -> list[Instance]:
    values = [Instance.from_dict(x) for x in read_jsonl(path)]
    if not values:
        raise DataError(f"{path}: dataset is empty")
    ids = [x.instance_id for x in values]
    if len(ids) != len(set(ids)):
        raise DataError(f"{path}: duplicate instance_id")
    return values


def statistics(instances: list[Instance]) -> dict:
    sizes = Counter(len(x.obligations) for x in instances)
    return {"instances": len(instances), "positive_instances": len(instances) - sizes[0], "negative_instances": sizes[0], "obligations": sum(len(x.obligations) for x in instances), "obligation_set_sizes": dict(sorted(sizes.items())), "task_domains": dict(Counter(x.task_domain for x in instances if x.task_domain)), "safety_categories": dict(Counter(o.safety_category for x in instances for o in x.obligations if o.safety_category))}


def validate_benchmark(instances: list[Instance]) -> dict:
    stats = statistics(instances)
    expected = {"instances": 240, "positive_instances": 120, "negative_instances": 120, "obligations": 339}
    for key, value in expected.items():
        if stats[key] != value:
            raise DataError(f"ObligationBench {key}: expected {value}, received {stats[key]}")
    if any(len(instance.obligations) > 5 for instance in instances):
        raise DataError("ObligationBench supports at most five obligations per instance")
    pairs: dict[str, list[Instance]] = {}
    for instance in instances:
        if instance.pair_id:
            pairs.setdefault(instance.pair_id, []).append(instance)
    for pair, values in pairs.items():
        if len(values) != 2 or len({bool(x.obligations) for x in values}) != 2 or values[0].task != values[1].task:
            raise DataError(f"{pair}: expected one positive and one negative trajectory for the same task")
    if not set(stats["task_domains"]) <= {"issue_resolution", "feature_development", "terminal_operation"}:
        raise DataError("ObligationBench has an unsupported task domain")
    if not set(stats["safety_categories"]) <= {"Data Exposure", "Unauthorized Access", "Asset Tampering", "Identity Forgery", "Others"}:
        raise DataError("ObligationBench has an unsupported safety category")
    return stats


def validate_training_split(train: list[Instance], validation: list[Instance], expected_train: int = 40000, expected_validation: int = 2000, *, split_by_scenario: bool = False) -> None:
    if len(train) != expected_train or len(validation) != expected_validation:
        raise DataError(f"expected {expected_train} training and {expected_validation} validation examples")
    if {x.instance_id for x in train} & {x.instance_id for x in validation}:
        raise DataError("training and validation share instance identifiers")
    if not split_by_scenario:
        return
    def groups(instances):
        result = set()
        for instance in instances:
            if not instance.scenario_id:
                raise DataError(f"{instance.instance_id}: scenario_id is required for training")
            result.add(instance.scenario_id)
            result.update(instance.source_scenario_ids)
        return result
    if groups(train) & groups(validation):
        raise DataError("training and validation share base scenarios")
