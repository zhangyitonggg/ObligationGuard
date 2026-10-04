from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .backends import Backend, decode_json_output
from .errors import DataError, ModelOutputError
from .identifiers import MODEL_TASK_ID
from .io import append_jsonl, fingerprint, read_jsonl, write_json
from .matching import maximum_matching
from .metrics import InstanceScore, aggregate
from .prompting import identification, semantic_matching
from .schema import Instance, parse_prediction


def parse_record(row: dict, identifier: str):
    if row.get("completion_error"):
        return (), row["completion_error"]
    try:
        return parse_prediction(decode_json_output(row["text"]), identifier), None
    except ModelOutputError as exc:
        return (), str(exc)


def predict(instances: list[Instance], backend: Backend, output: str | Path) -> None:
    output = Path(output)
    saved = list(read_jsonl(output)) if output.exists() else []
    previous = {x["instance_id"]: x for x in saved}
    if len(previous) != len(saved):
        raise DataError("existing predictions contain duplicate instance identifiers")
    if not set(previous) <= {x.instance_id for x in instances}:
        raise DataError("existing predictions contain instances outside the input dataset")
    for instance in instances:
        prompt = identification(instance)
        identity = fingerprint({"input": instance.guard_input(), "prompt": prompt, "backend": backend.identity(), "parse_failure_policy": "empty_set"})
        if instance.instance_id in previous:
            if previous[instance.instance_id]["fingerprint"] != identity:
                raise DataError("existing predictions use different inputs, prompts, or model settings")
            _, error = parse_record(previous[instance.instance_id], MODEL_TASK_ID)
            if previous[instance.instance_id].get("parse_error") != error:
                raise DataError("cached prediction parse status does not match the raw output")
            continue
        result = backend.complete(prompt)
        row = {"instance_id": instance.instance_id, "fingerprint": identity, "text": result.text, "raw_response": result.raw, "request": result.request, "model_settings": backend.identity(), "completion_error": result.error}
        _, error = parse_record(row, MODEL_TASK_ID)
        row["parse_error"] = error
        append_jsonl(output, row)


def evaluate(instances: list[Instance], predictions_path: str | Path, judge: Backend, judgments_path: str | Path, output: str | Path) -> dict:
    predictions = {}
    for row in read_jsonl(predictions_path):
        identifier = row["instance_id"]
        if identifier in predictions:
            raise DataError(f"duplicate prediction for {identifier}")
        predictions[identifier] = row
    expected = {x.instance_id for x in instances}
    if set(predictions) != expected:
        raise DataError("prediction identifiers must exactly match the evaluation dataset")
    scores = []
    rows = []
    judgments_path = Path(judgments_path)
    cache = {x["fingerprint"]: x for x in read_jsonl(judgments_path)} if judgments_path.exists() else {}
    for instance in instances:
        prediction, parse_error = parse_record(predictions[instance.instance_id], MODEL_TASK_ID)
        matrix = []
        for i, obligation in enumerate(prediction):
            decisions = []
            for j, truth in enumerate(instance.obligations):
                prompt = semantic_matching(instance, truth.required_safety_action, obligation.required_safety_action)
                identity = fingerprint({"prompt": prompt, "judge": judge.identity()})
                if identity not in cache:
                    completion = judge.complete(prompt)
                    try:
                        if completion.error:
                            raise ModelOutputError(completion.error)
                        decision = decode_json_output(completion.text)
                        if type(decision.get("match")) is not bool:
                            raise ModelOutputError("semantic judge must return a boolean match")
                    except ModelOutputError as exc:
                        append_jsonl(judgments_path.with_name(judgments_path.stem + "_errors.jsonl"), {"fingerprint": identity, "instance_id": instance.instance_id, "completion_error": str(exc), "text": completion.text, "raw_response": completion.raw, "request": completion.request})
                        raise
                    row = {"fingerprint": identity, "instance_id": instance.instance_id, "prediction_index": i, "truth_index": j, "match": decision["match"], "rationale": decision.get("rationale", ""), "raw_response": completion.raw, "request": completion.request, "judge_settings": judge.identity()}
                    append_jsonl(judgments_path, row)
                    cache[identity] = row
                decisions.append(cache[identity]["match"])
            matrix.append(decisions)
        pairs = maximum_matching(matrix, len(instance.obligations))
        score = InstanceScore(instance.instance_id, len(prediction), len(instance.obligations), len(pairs))
        scores.append(score)
        rows.append({**asdict(score), "exact_match": score.exact_match, "matched_pairs": pairs, "parse_error": parse_error})
    result = {"metrics": aggregate(scores), "parse_failures": sum(row["parse_error"] is not None for row in rows), "parse_failure_policy": "empty_set", "instances": rows}
    write_json(output, result)
    return result
