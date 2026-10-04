from __future__ import annotations

from pathlib import Path

from .config import check_judge, readiness
from .errors import ConfigurationError
from .experiments import launch_training, run_rq1, run_rq2, run_rq3
from .io import file_sha256, fingerprint, write_json


def pipeline(paper: dict, runtime: dict) -> None:
    stages = runtime.get("pipeline", {}).get("stages", ["rq1"])
    if not isinstance(stages, list) or any(not isinstance(stage, str) or stage not in {"rq1", "rq2", "rq3", "rq4"} for stage in stages) or len(stages) != len(set(stages)):
        raise ConfigurationError("pipeline stages must be unique entries from rq1, rq2, rq3, rq4")
    checks = readiness(paper, runtime)
    missing = [name for name, valid in checks.items() if not valid]
    if missing:
        raise ConfigurationError("prepare these requirements before starting: " + ", ".join(missing))
    check_judge(runtime["models"]["judge"])
    paths = runtime["paths"]
    output = Path(paths["results"])
    identity = fingerprint({"paper": paper, "runtime": runtime, "train": file_sha256(paths["train"]), "validation": file_sha256(paths["validation"]), "benchmark": file_sha256(paths["benchmark"])})
    state_path = output / "pipeline_state.json"
    if state_path.exists():
        import json
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["fingerprint"] != identity:
            raise ConfigurationError("existing pipeline results use different settings or data; select a new results directory")
    else:
        state = {"fingerprint": identity, "training_complete": False}
    write_json(state_path, state)
    launch_training(paper, runtime, output / "main_training")
    state["training_complete"] = True
    write_json(state_path, state)
    results = run_rq1(paper, runtime) if "rq1" in stages else None
    if "rq2" in stages:
        run_rq2(paper, runtime, results)
    if "rq3" in stages:
        run_rq3(paper, runtime)
    if "rq4" in stages:
        from .rq4 import run_rq4
        run_rq4(paper, runtime)
