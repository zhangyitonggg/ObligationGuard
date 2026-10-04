from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from .errors import DataError
from .metrics import InstanceScore, aggregate
from .schema import Instance


def action_observation_steps(instance: Instance) -> Instance:
    """Count tool actions, preserving message references, rather than counting chat messages."""
    if [item.get("step") for item in instance.trajectory] == list(range(1, len(instance.trajectory) + 1)):
        return instance
    steps, pending, calls = [], [], {}
    for index, message in enumerate(instance.trajectory):
        tool_calls = message.get("tool_calls") if message.get("role") == "assistant" else None
        if tool_calls:
            for number, call in enumerate(tool_calls):
                identifier = call.get("id")
                if not identifier or identifier in calls:
                    raise DataError("RQ2 requires unique identifiers for recorded tool actions")
                step = {"step": len(steps) + 1, "action": {"message_index": index, "tool_call": call}, "observation": []}
                if number == 0:
                    step["context"] = pending + [{"message_index": index, **{key: value for key, value in message.items() if key != "tool_calls"}}]
                    pending = []
                steps.append(step)
                calls[identifier] = step
        elif message.get("role") == "tool":
            identifier = message.get("tool_call_id")
            if identifier not in calls:
                raise DataError("RQ2 tool observation has no corresponding action")
            calls[identifier]["observation"].append({"message_index": index, **message})
        else:
            pending.append({"message_index": index, **message})
    if not steps:
        raise DataError("RQ2 requires recorded action/observation steps")
    steps[-1].setdefault("context", []).extend(pending)
    return replace(instance, trajectory=tuple(steps))


def distance_group(instance: Instance) -> str:
    if not instance.obligations:
        raise DataError("creation distance is defined only for positive instances")
    steps = [x.get("step") for x in instance.trajectory]
    if any(type(x) is not int for x in steps) or steps != list(range(1, len(steps) + 1)):
        raise DataError("RQ2 requires sequential action/observation step numbers")
    creation = [x.creation_step for x in instance.obligations]
    if any(x is None or x > len(steps) for x in creation):
        raise DataError("RQ2 requires a valid creation step for every ground-truth obligation")
    distance = len(steps) - min(creation)
    return "<=4" if distance <= 4 else "5-8" if distance <= 8 else "9-16" if distance <= 16 else ">16"


def rq2(instances: list[Instance], model_results: dict[str, dict]) -> dict:
    if not model_results:
        raise DataError("RQ2 requires evaluated model results")
    positive = [x for x in instances if x.obligations]
    groups = {"creation_distance": defaultdict(list), "obligation_count": defaultdict(list)}
    for instance in positive:
        groups["creation_distance"][distance_group(instance)].append(instance.instance_id)
        groups["obligation_count"][str(len(instance.obligations))].append(instance.instance_id)
    result = {}
    for factor, buckets in groups.items():
        result[factor] = {}
        for bucket, identifiers in buckets.items():
            metrics = {}
            for model, evaluated in model_results.items():
                scores = {row["instance_id"]: InstanceScore(**{key: row[key] for key in ("instance_id", "prediction_count", "truth_count", "matched_count")}) for row in evaluated["instances"]}
                if not set(identifiers) <= scores.keys():
                    raise DataError("RQ2 is missing model scores for a group")
                metrics[model] = aggregate(scores[x] for x in identifiers)
            result[factor][bucket] = {"instances": len(identifiers), "models": metrics, "mean": {key: sum(value[key] for value in metrics.values()) / len(metrics) for key in ("precision", "recall", "exact_match")}}
    return result


def table_csv(rows: list[dict], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "precision", "recall", "exact_match", "ca_positive", "ca_negative"]
    if all("ca_negative" not in row["metrics"] for row in rows):
        fields.remove("ca_negative")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({"model": row["model"], **{key: "" if row["metrics"][key] is None else f"{100 * row['metrics'][key]:.2f}" for key in fields[1:]}})
