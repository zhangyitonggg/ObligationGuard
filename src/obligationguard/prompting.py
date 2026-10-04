from __future__ import annotations

from importlib.resources import files
import json
import re

from .errors import ConfigurationError
from .schema import Instance


PLACEHOLDERS = re.compile(r"\{([A-Z][A-Z_]*|guard_input\.json)\}")


def prompt_text(name: str) -> str:
    if not re.fullmatch(r"[a-z_]+", name):
        raise ConfigurationError("invalid prompt name")
    path = files("obligationguard").joinpath("prompts", name + ".txt")
    if not path.is_file():
        raise ConfigurationError(f"unknown prompt: {name}")
    return path.read_text(encoding="utf-8")


def render(name: str, **values) -> str:
    text = prompt_text(name)
    required = set(PLACEHOLDERS.findall(text))
    if required != set(values):
        raise ConfigurationError(f"{name}: placeholder mismatch; expected {sorted(required)}, received {sorted(values)}")
    def replace(match):
        value = values[match.group(1)]
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    return PLACEHOLDERS.sub(replace, text)


def identification(instance: Instance) -> str:
    return render("obligation_identification", **{"guard_input.json": instance.guard_input()})


def semantic_matching(instance: Instance, truth: str, prediction: str) -> str:
    return render("semantic_matching", USER_TASK=instance.task, TRAJECTORY=list(instance.trajectory), GROUND_TRUTH_OBLIGATION=truth, PREDICTED_OBLIGATION=prediction)
