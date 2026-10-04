from __future__ import annotations

from .errors import DataError
from .schema import Instance, Obligation


def trajectory_jobs(plans: list[dict], variants_per_scenario: int) -> list[dict]:
    if variants_per_scenario < 1:
        raise DataError("variants_per_scenario must be positive")
    jobs = []
    for plan in plans:
        template = plan["output"]
        scenario = plan["job_id"]
        for variant in range(1, variants_per_scenario + 1):
            jobs.append({"job_id": f"{scenario}-trajectory-{variant}", "scenario_id": scenario, "inputs": {"USER_TASK": template["user_task"], "TASK_SCENARIO": template["task_scenario"], "SAFETY_REQUIREMENTS": template["safety_requirements"], "GROUND_TRUTH_OBLIGATION_SET": template["ground_truth_obligation_set"]}})
    return jobs


def assemble_examples(plans: list[dict], trajectories: list[dict]) -> list[Instance]:
    templates = {row["job_id"]: row["output"] for row in plans}
    if len(templates) != len(plans):
        raise DataError("duplicate scenario identifier")
    instances = []
    for row in trajectories:
        if row["output"].get("status") == "REJECT":
            continue
        scenario = row.get("scenario_id")
        if scenario not in templates:
            raise DataError("accepted trajectory must identify its planned scenario")
        template = templates[scenario]
        inputs = row["inputs"]
        expected = {"USER_TASK": template["user_task"], "TASK_SCENARIO": template["task_scenario"], "SAFETY_REQUIREMENTS": template["safety_requirements"], "GROUND_TRUTH_OBLIGATION_SET": template["ground_truth_obligation_set"]}
        if inputs != expected:
            raise DataError("trajectory synthesis inputs changed the planned scenario or obligation set")
        obligations = tuple(Obligation(action, id=f"O{i}") for i, action in enumerate(template["ground_truth_obligation_set"], 1))
        instances.append(Instance(row["job_id"], template["user_task"], tuple(row["output"]["trajectory"]), obligations, scenario_id=scenario))
    return instances
