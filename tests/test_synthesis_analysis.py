import unittest

from obligationguard.analysis import distance_group, rq2
from obligationguard.errors import DataError
from obligationguard.schema import Instance, Obligation
from obligationguard.synthesis import assemble_examples, trajectory_jobs


class SynthesisTests(unittest.TestCase):
    def test_planned_labels_are_attached_without_asking_generator_to_relabel(self):
        plans = [{"job_id": "s", "output": {"user_task": "Task", "task_scenario": "Scenario", "safety_requirements": [], "ground_truth_obligation_set": ["Revoke the token"]}}]
        job = trajectory_jobs(plans, 1)[0]
        result = {**job, "output": {"status": "ACCEPT", "trajectory": [{"step": 1, "action": "Use token", "observation": "Active token"}]}}
        examples = assemble_examples(plans, [result])
        self.assertEqual(examples[0].obligations[0].required_safety_action, "Revoke the token")
        self.assertEqual(examples[0].scenario_id, "s")

    def test_rejection_does_not_become_a_training_example(self):
        self.assertEqual(assemble_examples([], [{"output": {"status": "REJECT"}}]), [])

    def test_changed_plan_is_rejected(self):
        plans = [{"job_id": "s", "output": {"user_task": "Task", "task_scenario": "Scenario", "safety_requirements": [], "ground_truth_obligation_set": []}}]
        job = trajectory_jobs(plans, 1)[0]
        job["inputs"]["USER_TASK"] = "Changed task"
        with self.assertRaises(DataError):
            assemble_examples(plans, [{**job, "output": {"status": "ACCEPT", "trajectory": [{"step": 1}]}}])


class AnalysisTests(unittest.TestCase):
    def sample(self, total, creation):
        return Instance("a", "Task", tuple({"step": i} for i in range(1, total + 1)), (Obligation("Required action", creation_step=creation),))

    def test_distance_boundaries(self):
        for distance, expected in ((4, "<=4"), (5, "5-8"), (8, "5-8"), (9, "9-16"), (16, "9-16"), (17, ">16")):
            self.assertEqual(distance_group(self.sample(distance + 1, 1)), expected)

    def test_earliest_creation_step_is_used(self):
        sample = Instance("a", "Task", tuple({"step": i} for i in range(1, 12)), (Obligation("Action A", creation_step=2), Obligation("Action B", creation_step=10)))
        self.assertEqual(distance_group(sample), "9-16")

    def test_rq2_averages_model_metrics_after_group_aggregation(self):
        sample = self.sample(3, 1)
        models = {"a": {"instances": [{"instance_id": "a", "prediction_count": 1, "truth_count": 1, "matched_count": 1}]}, "b": {"instances": [{"instance_id": "a", "prediction_count": 0, "truth_count": 1, "matched_count": 0}]}}
        result = rq2([sample], models)
        self.assertEqual(result["creation_distance"]["<=4"]["mean"]["recall"], 0.5)
