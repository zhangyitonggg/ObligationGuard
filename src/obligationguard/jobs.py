from __future__ import annotations

from pathlib import Path

from .backends import Backend, decode_json_output
from .errors import DataError, ModelOutputError
from .io import append_jsonl, fingerprint, read_jsonl
from .prompting import render


STAGES = {"scenario_planning", "trajectory_synthesis", "direct_synthesis", "trajectory_filtering", "trajectory_annotation", "safety_failure_attribution", "creation_step_annotation"}


def validate_job_output(stage: str, value: dict, inputs: dict) -> None:
    if stage == "scenario_planning":
        for key, kind in (("user_task", str), ("task_scenario", str), ("safety_requirements", list), ("ground_truth_obligation_set", list)):
            if not isinstance(value.get(key), kind):
                raise ModelOutputError(f"scenario planning is missing {key}")
    elif stage == "trajectory_synthesis":
        if value.get("status") == "REJECT":
            if not isinstance(value.get("reason"), str):
                raise ModelOutputError("rejected synthesis must provide a reason")
            return
        if value.get("status") != "ACCEPT" or not isinstance(value.get("trajectory"), list) or not value["trajectory"]:
            raise ModelOutputError("trajectory synthesis must ACCEPT a nonempty trajectory or REJECT")
    elif stage == "direct_synthesis":
        if not isinstance(value.get("user_task"), str) or not isinstance(value.get("trajectory"), list) or not isinstance(value.get("ground_truth_obligation_set"), list):
            raise ModelOutputError("direct synthesis must return task, trajectory, and obligation set")
    elif stage == "trajectory_filtering":
        if value.get("decision") not in ("RETAIN", "DISCARD"):
            raise ModelOutputError("filtering decision must be RETAIN or DISCARD")
    elif stage == "trajectory_annotation":
        if not isinstance(value.get("obligations"), list):
            raise ModelOutputError("annotation must return an obligations array")
    elif stage == "creation_step_annotation":
        step = value.get("creation_step")
        if type(step) is not int or step < 1 or step > len(inputs["TRAJECTORY"]):
            raise ModelOutputError("creation step must refer to the provided trajectory")
    elif stage == "safety_failure_attribution":
        for key in ("forbidden_action", "unfulfilled_obligation"):
            if not isinstance(value.get(key), dict) or type(value[key].get("present")) is not bool:
                raise ModelOutputError(f"attribution must contain boolean {key}.present")


def run_jobs(stage: str, input_path: str | Path, output_path: str | Path, backend: Backend) -> None:
    if stage not in STAGES:
        raise DataError(f"unknown synthesis/annotation stage: {stage}")
    output_path = Path(output_path)
    saved = list(read_jsonl(output_path)) if output_path.exists() else []
    previous = {x["job_id"]: x for x in saved}
    if len(previous) != len(saved):
        raise DataError("existing job outputs contain duplicate identifiers")
    seen = set()
    for job in read_jsonl(input_path):
        identifier = job.get("job_id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise DataError("each job must have a unique nonempty job_id")
        seen.add(identifier)
        inputs = job.get("inputs")
        if not isinstance(inputs, dict):
            raise DataError("job inputs must be an object of prompt placeholders")
        prompt = render(stage, **inputs)
        identity = fingerprint({"stage": stage, "prompt": prompt, "model": backend.identity()})
        if identifier in previous:
            if previous[identifier]["fingerprint"] != identity:
                raise DataError("existing job result uses different inputs or model settings")
            value = decode_json_output(previous[identifier]["text"]) if "text" in previous[identifier] else previous[identifier]["output"]
            validate_job_output(stage, value, inputs)
            if "output" not in previous[identifier]:
                raise ModelOutputError("existing job output did not pass schema validation")
            continue
        completion = backend.complete(prompt)
        row = {"job_id": identifier, "scenario_id": job.get("scenario_id"), "inputs": inputs, "fingerprint": identity, "stage": stage, "text": completion.text, "raw_response": completion.raw, "request": completion.request, "model_settings": backend.identity(), "completion_error": completion.error}
        try:
            if completion.error:
                raise ModelOutputError(completion.error)
            value = decode_json_output(completion.text)
            validate_job_output(stage, value, inputs)
        except ModelOutputError as exc:
            append_jsonl(output_path, {**row, "parse_error": str(exc)})
            raise
        append_jsonl(output_path, {**row, "output": value})
    if not set(previous) <= seen:
        raise DataError("existing job outputs contain identifiers outside the job list")
