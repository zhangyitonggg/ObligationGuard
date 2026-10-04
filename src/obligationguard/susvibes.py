from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from .errors import ConfigurationError, DataError
from .guidance import outcome, summarize_outcomes, transition_matrix
from .io import read_jsonl


SUSVIBES_REPOSITORY = "https://github.com/LeiLiLab/susvibes"
SUSVIBES_COMMIT = "520f16f9b3f39b06c6013bc0eb57a38280b4deee"


def load_tasks(path: str | Path) -> list[dict]:
    rows = list(read_jsonl(path))
    if len(rows) != 186 or len({row.get("instance_id") for row in rows}) != 186:
        raise DataError("SusVibes requires 186 unique task instances")
    for row in rows:
        for key in ("instance_id", "image_name", "problem_statement"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise DataError(f"SusVibes task is missing {key}")
    return rows


def evaluate_patches(checkout: str | Path, predictions: str | Path, run_id: str, workers: int) -> None:
    checkout = Path(checkout).resolve()
    if not (checkout / "susvibes" / "eval" / "core.py").is_file():
        raise ConfigurationError("checkout must point to the official SusVibes repository")
    if workers < 1:
        raise ConfigurationError("workers must be positive")
    command = [sys.executable, "-m", "susvibes.eval.core", "--run_id", run_id, "--predictions_path", str(Path(predictions).resolve()), "--max_workers", str(workers), "--strategy", "none"]
    subprocess.run(command, cwd=checkout, check=True)


def read_condition(reports_path: str | Path, tasks: list[dict]) -> dict:
    reports = json.loads(Path(reports_path).read_text(encoding="utf-8"))
    if not isinstance(reports, dict):
        raise DataError("condition reports must map instance identifiers to harness reports")
    expected = {x["instance_id"] for x in tasks}
    if set(reports) != expected:
        raise DataError("condition reports must contain all 186 tasks")
    values = {}
    for identifier, report in reports.items():
        if report.get("error"):
            raise DataError(f"{identifier}: harness evaluation is incomplete")
        status = report.get("eval_status")
        if status in ("empty_patch", "patch_error"):
            values[identifier] = outcome(False, False)
        elif status == "tested":
            values[identifier] = outcome(report["run"]["func"]["pass"], report["run"]["sec"]["pass"])
        else:
            raise DataError(f"{identifier}: harness evaluation is incomplete")
    return values


def report_guidance(tasks: list[dict], conditions: dict[str, dict]) -> dict:
    if "no_guidance" not in conditions:
        raise DataError("guidance comparison requires a no_guidance condition")
    expected = {x["instance_id"] for x in tasks}
    baseline = summarize_outcomes(conditions["no_guidance"], expected)
    result = {}
    for name, values in conditions.items():
        summary = summarize_outcomes(values, expected)
        result[name] = {**summary, "delta_sec": summary["sec_pass"] - baseline["sec_pass"], "transitions": transition_matrix(conditions["no_guidance"], values)}
    return result
