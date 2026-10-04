from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import DataError, ModelOutputError
from .identifiers import MODEL_TASK_ID


def nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataError(f"{name}: expected a nonempty string")
    return value


@dataclass(frozen=True)
class Obligation:
    required_safety_action: str
    id: str = ""
    safety_consequence: str = ""
    evidence: tuple = ()
    creation_step: int | None = None
    safety_category: str | None = None

    @classmethod
    def from_dict(cls, value: dict) -> "Obligation":
        if not isinstance(value, dict):
            raise DataError("obligation: expected an object")
        action = nonempty_text(value.get("required_safety_action"), "required_safety_action")
        evidence = value.get("evidence", [])
        if not isinstance(evidence, list) or any(not isinstance(x, (str, dict)) for x in evidence):
            raise DataError("evidence: expected an array of strings or objects")
        step = value.get("creation_step")
        if step is not None and (type(step) is not int or step < 1):
            raise DataError("creation_step: expected a positive integer")
        for key in ("id", "safety_consequence"):
            if key in value and not isinstance(value[key], str):
                raise DataError(f"{key}: expected a string")
        category = value.get("safety_category")
        if category is not None and not isinstance(category, str):
            raise DataError("safety_category: expected a string")
        return cls(action, value.get("id", ""), value.get("safety_consequence", ""), tuple(evidence), step, category)

    def target_dict(self, number: int) -> dict:
        return {
            "id": self.id or f"O{number}",
            "required_safety_action": self.required_safety_action,
            "safety_consequence": self.safety_consequence,
            "evidence": list(self.evidence),
        }

    def to_dict(self, number: int) -> dict:
        value = self.target_dict(number)
        if self.creation_step is not None:
            value["creation_step"] = self.creation_step
        if self.safety_category is not None:
            value["safety_category"] = self.safety_category
        return value


@dataclass(frozen=True)
class Instance:
    instance_id: str
    task: str
    trajectory: tuple[dict, ...]
    obligations: tuple[Obligation, ...]
    scenario_id: str | None = None
    source_scenario_ids: tuple[str, ...] = ()
    task_domain: str | None = None
    pair_id: str | None = None
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict) -> "Instance":
        identifier = nonempty_text(value.get("instance_id"), "instance_id")
        task = nonempty_text(value.get("task"), "task")
        trajectory = value.get("trajectory")
        if not isinstance(trajectory, list) or not trajectory or any(not isinstance(x, dict) for x in trajectory):
            raise DataError(f"{identifier}: trajectory must be a nonempty array of objects")
        obligations = value.get("obligations")
        if not isinstance(obligations, list):
            raise DataError(f"{identifier}: obligations must be an array (empty for negative instances)")
        parsed = tuple(Obligation.from_dict(x) for x in obligations)
        actions = [x.required_safety_action.strip() for x in parsed]
        if len(actions) != len(set(actions)):
            raise DataError(f"{identifier}: duplicate ground-truth obligation")
        for key in ("scenario_id", "task_domain", "pair_id"):
            if value.get(key) is not None:
                nonempty_text(value[key], key)
        sources = value.get("source_scenario_ids", [])
        if not isinstance(sources, list) or any(not isinstance(x, str) or not x for x in sources):
            raise DataError(f"{identifier}: source_scenario_ids must be an array of identifiers")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, dict):
            raise DataError("metadata: expected an object")
        return cls(identifier, task, tuple(trajectory), parsed, value.get("scenario_id"), tuple(sources), value.get("task_domain"), value.get("pair_id"), metadata)

    def guard_input(self) -> dict:
        return {"task_id": MODEL_TASK_ID, "task": self.task, "trajectory": list(self.trajectory)}

    def target(self) -> dict:
        return {"task_id": MODEL_TASK_ID, "obligation_count": len(self.obligations), "obligations": [x.target_dict(i) for i, x in enumerate(self.obligations, 1)]}

    def to_dict(self) -> dict:
        value = {"instance_id": self.instance_id, "task": self.task, "trajectory": list(self.trajectory), "obligations": [x.to_dict(i) for i, x in enumerate(self.obligations, 1)]}
        for key in ("scenario_id", "task_domain", "pair_id"):
            if getattr(self, key) is not None:
                value[key] = getattr(self, key)
        if self.source_scenario_ids:
            value["source_scenario_ids"] = list(self.source_scenario_ids)
        if self.metadata:
            value["metadata"] = self.metadata
        return value


def parse_prediction(value: dict, expected_id: str | None = None) -> tuple[Obligation, ...]:
    if not isinstance(value, dict) or not isinstance(value.get("obligations"), list):
        raise ModelOutputError("prediction must contain an obligations array")
    if expected_id is not None and value.get("task_id") != expected_id:
        raise ModelOutputError("prediction task_id does not match the input")
    obligations = value["obligations"]
    count = value.get("obligation_count")
    if type(count) is not int or count != len(obligations):
        raise ModelOutputError("obligation_count must equal the obligations array length")
    try:
        return tuple(Obligation.from_dict(x) for x in obligations)
    except DataError as exc:
        raise ModelOutputError(str(exc)) from exc
