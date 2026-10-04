from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from .backends import make_backend
from .container_state import capture_environment, docker_command, environment_class, patch
from .errors import ConfigurationError, DataError
from .io import file_sha256, fingerprint, write_json, write_jsonl
from .mini_swe import agent_class, configuration_path
from .serving import ModelServer
from .susvibes import evaluate_patches, load_tasks, read_condition, report_guidance


def run_rq4(paper: dict, runtime: dict) -> dict:
    import yaml
    from minisweagent.models import get_model
    settings = runtime["susvibes"]
    tasks = load_tasks(settings["tasks"])
    output = Path(runtime["paths"]["results"]) / "rq4"
    mini_config = configuration_path(settings)
    configuration = yaml.safe_load(mini_config.read_text(encoding="utf-8"))
    actor_config = runtime["evaluation_models"][paper["rq4"]["agent_model"]]
    if actor_config["backend"] not in {"managed_chat", "chat"}:
        raise ConfigurationError("RQ4's coding agent requires a Chat Completions endpoint")
    conditions = ["no_guidance", "self_reminder", *paper["rq4"]["guards"]]
    actor_server = ModelServer(actor_config) if actor_config["backend"] == "managed_chat" else None
    base_url = actor_server.start() if actor_server else actor_config["base_url"]
    model_config = deepcopy(configuration["model"])
    model_config.update(settings["agent"])
    model_config["model_name"] = "openai/" + actor_config["model"]
    model_config.setdefault("model_kwargs", {}).update({"api_base": base_url, "api_key": "EMPTY"})
    model_config["model_kwargs"]["drop_params"] = False
    Agent = agent_class()

    def make_agent(task, environment, condition, destination, guard=None):
        agent_config = deepcopy(configuration["agent"])
        agent_config["output_path"] = destination / "trajectory.json"
        return Agent(get_model(config=model_config), environment, condition=condition, task_id=task["instance_id"], guard=guard, **agent_config)

    try:
        for task in tasks:
            identifier = task["instance_id"]
            directory = output / "tasks" / identifier
            identity = fingerprint({"task": task, "actor": actor_config, "mini_config": file_sha256(mini_config), "settings": settings, "guards": {name: runtime["evaluation_models"][name] for name in paper["rq4"]["guards"]}})
            state_path = directory / "state.json"
            if state_path.exists():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if state["fingerprint"] != identity:
                    raise DataError("saved SusVibes execution uses different inputs or settings")
            else:
                state = {"fingerprint": identity, "completed": []}
                environment_config = deepcopy(configuration["environment"])
                environment_config.update({"image": task["image_name"], "cwd": "/project", "container_timeout": settings["container_timeout"], "timeout": settings["command_timeout"], "run_args": ["--security-opt", "seccomp=unconfined"]})
                environment = environment_class()(**environment_config)
                try:
                    agent = make_agent(task, environment, "no_guidance", directory / "no_guidance")
                    def capture(current, snapshot):
                        state["snapshot"] = capture_environment(environment, directory / "checkpoint", "obligationguard-snapshot:" + fingerprint({"task": identifier, "run": settings["run_id"]})[:32], settings["snapshot_mode"])
                        write_json(directory / "first_termination.json", snapshot)
                        write_json(state_path, state)
                    agent.on_first_termination = capture
                    exit_data = agent.run(task["problem_statement"])
                    if "snapshot" in state and state["snapshot"]["mode"] == "checkpoint":
                        saved_environment = {**state["snapshot"]["environment"], "image": state["snapshot"]["image"]}
                        baseline = environment_class(state["snapshot"], filesystem_only=True)(**saved_environment)
                        try:
                            baseline_patch = patch(baseline)
                        finally:
                            baseline.cleanup()
                    else:
                        baseline_patch = patch(environment)
                    write_json(directory / "no_guidance" / "solution.json", {"instance_id": identifier, "model_patch": baseline_patch, "exit": exit_data})
                    state["completed"].append("no_guidance")
                    state["terminated"] = "snapshot" in state
                    write_json(state_path, state)
                finally:
                    environment.cleanup()
            if "no_guidance" not in state["completed"]:
                snapshot = state["snapshot"]
                saved_environment = environment_class(snapshot, filesystem_only=True)(**{**snapshot["environment"], "image": snapshot["image"]})
                try:
                    recovered_patch = patch(saved_environment)
                finally:
                    saved_environment.cleanup()
                captured = json.loads((directory / "first_termination.json").read_text(encoding="utf-8"))
                write_json(directory / "no_guidance" / "solution.json", {"instance_id": identifier, "model_patch": recovered_patch, "exit": captured["resume"]["termination_messages"][-1]["extra"]})
                captured["messages"].extend(captured["resume"]["termination_messages"])
                write_json(directory / "no_guidance" / "trajectory.json", captured)
                state["completed"].append("no_guidance")
                state["terminated"] = True
                write_json(state_path, state)
            baseline_solution = json.loads((directory / "no_guidance" / "solution.json").read_text(encoding="utf-8"))
            for condition in conditions[1:]:
                if condition in state["completed"]:
                    continue
                if not state["terminated"]:
                    write_json(directory / condition / "solution.json", {**baseline_solution, "intervention": "no_termination_attempt"})
                else:
                    snapshot = state["snapshot"]
                    environment = environment_class(snapshot)(**{**snapshot["environment"], "image": snapshot["image"]})
                    guard = None
                    agent = None
                    try:
                        if condition != "self_reminder":
                            guard = make_backend(runtime["evaluation_models"][condition])
                        agent = make_agent(task, environment, "self_reminder" if guard is None else "guard", directory / condition, guard)
                        exit_data = agent.resume(json.loads((directory / "first_termination.json").read_text(encoding="utf-8")))
                        solution = {"instance_id": identifier, "model_patch": patch(environment), "exit": exit_data, "guard_response": agent.guard_response, "shared_state": fingerprint({"snapshot_image": snapshot["image"], "agent": file_sha256(directory / "first_termination.json")})}
                        write_json(directory / condition / "solution.json", solution)
                    finally:
                        if guard is not None and hasattr(guard, "close"):
                            guard.close()
                        if guard is not None and agent is not None:
                            agent.guard = None
                        del guard
                        import gc
                        gc.collect()
                        if condition in {"Qwen3-8B", "ObligationGuard"}:
                            import torch
                            torch.cuda.empty_cache()
                        environment.cleanup()
                state["completed"].append(condition)
                write_json(state_path, state)
            if "snapshot" in state and not state.get("snapshot_released"):
                docker_command(state["snapshot"]["environment"]["executable"], "image", "rm", state["snapshot"]["tag"])
                state["snapshot_released"] = True
                write_json(state_path, state)
        predictions = []
        for condition in conditions:
            for task in tasks:
                solution = json.loads((output / "tasks" / task["instance_id"] / condition / "solution.json").read_text(encoding="utf-8"))
                predictions.append({"instance_id": task["instance_id"], "model_name_or_path": condition, "model_patch": solution["model_patch"]})
        write_jsonl(output / "predictions.jsonl", predictions)
        evaluate_patches(settings["checkout"], output / "predictions.jsonl", settings["run_id"], settings["workers"])
        outcomes = {}
        for condition in conditions:
            reports = {}
            for task in tasks:
                location = Path(settings["checkout"]) / "logs" / "eval" / settings["run_id"] / "none" / condition / task["instance_id"] / "report.json"
                reports[task["instance_id"]] = json.loads(location.read_text(encoding="utf-8"))
            write_json(output / condition / "reports.json", reports)
            outcomes[condition] = read_condition(output / condition / "reports.json", tasks)
        result = report_guidance(tasks, outcomes)
        write_json(output / "metrics.json", result)
        import csv
        with (output / "table.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["condition", "func_pass", "sec_pass", "delta_sec"])
            writer.writeheader()
            for condition, summary in result.items():
                writer.writerow({"condition": condition, **{key: f"{100 * summary[key]:.2f}" for key in ("func_pass", "sec_pass", "delta_sec")}})
        offline = Path(runtime["paths"]["results"]) / "rq1"
        if all((offline / name / "metrics.json").is_file() for name in paper["rq4"]["guards"]):
            recalls = {name: json.loads((offline / name / "metrics.json").read_text(encoding="utf-8"))["metrics"]["recall"] for name in paper["rq4"]["guards"]}
            from .plots import plot_guidance
            plot_guidance(result, recalls, output / "recall_and_safety.pdf")
        return result
    finally:
        if actor_server:
            actor_server.close()
