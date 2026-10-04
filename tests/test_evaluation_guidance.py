import json
import os
from pathlib import Path
import tempfile
import unittest

from obligationguard.backends import Completion, decode_json_output, final_answer
from obligationguard.errors import DataError, ModelOutputError
from obligationguard.evaluation import evaluate, predict
from obligationguard.guidance import FirstTerminationGuidance, Outcome, outcome, summarize_outcomes, transition_matrix
from obligationguard.io import read_jsonl, write_jsonl
from obligationguard.schema import Instance, Obligation


class Judge:
    def __init__(self): self.calls = 0
    def identity(self): return {"model": "test-judge"}
    def complete(self, prompt):
        self.calls += 1
        return Completion('{"match":true,"rationale":"test decision"}', {"test": True}, {"prompt": prompt})


class EvaluationTests(unittest.TestCase):
    def test_identical_model_placeholders_keep_separate_local_predictions(self):
        class IdentifierBackend(Judge):
            def complete(self, prompt):
                self.calls += 1
                self.assertion = "source-U" not in prompt and "source-negative" not in prompt
                return Completion('{"task_id":"task","obligation_count":0,"obligations":[]}', {}, {})
        with tempfile.TemporaryDirectory() as location:
            root = Path(location)
            samples = [Instance("source-U", "Task", ({"step": 1},), (Obligation("Truth"),)), Instance("source-negative", "Task", ({"step": 1},), ())]
            backend = IdentifierBackend()
            path = root / "predictions.jsonl"
            predict(samples, backend, path)
            predict(samples, backend, path)
            rows = list(read_jsonl(path))
            self.assertEqual(backend.calls, 2)
            self.assertTrue(backend.assertion)
            self.assertEqual([row["instance_id"] for row in rows], [sample.instance_id for sample in samples])
            self.assertTrue(all(row["parse_error"] is None for row in rows))
            result = evaluate(samples, path, Judge(), root / "judgments.jsonl", root / "metrics.json")
            self.assertEqual(result["parse_failures"], 0)
            self.assertEqual(result["metrics"]["counts"]["positive_instances"], 1)
            self.assertEqual(result["metrics"]["counts"]["negative_instances"], 1)

    def test_judge_cache_and_maximum_matching(self):
        scratch = Path(os.environ.get("OG_TEST_SCRATCH", "D:/Codex/scratch/obligationguard-tests" if os.name == "nt" else "/tmp/obligationguard-tests"))
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            directory = Path(directory)
            sample = Instance("a", "Task", ({"step": 1},), (Obligation("Truth"),))
            prediction = {"task_id": "task", "obligation_count": 2, "obligations": [{"required_safety_action": "Prediction A"}, {"required_safety_action": "Prediction B"}]}
            path = directory / "predictions.jsonl"
            write_jsonl(path, [{"instance_id": "a", "text": json.dumps(prediction)}])
            judge = Judge()
            result = evaluate([sample], path, judge, directory / "judgments.jsonl", directory / "metrics.json")
            self.assertEqual(judge.calls, 2)
            self.assertEqual(result["metrics"]["recall"], 1.0)
            self.assertEqual(result["metrics"]["precision"], 0.5)
            self.assertEqual(result["metrics"]["exact_match"], 0.0)
            evaluate([sample], path, judge, directory / "judgments.jsonl", directory / "metrics.json")
            self.assertEqual(judge.calls, 2)

    def test_json_decoder_rejects_classification_only_outputs(self):
        with self.assertRaises(ModelOutputError):
            decode_json_output("safe")

    def test_parse_failure_is_scored_as_empty_and_generation_is_not_repeated(self):
        class ClassificationBackend(Judge):
            def complete(self, prompt):
                self.calls += 1
                return Completion("safe", {"original": "safe"}, {})
        scratch = Path(os.environ.get("OG_TEST_SCRATCH", "D:/Codex/scratch/obligationguard-tests" if os.name == "nt" else "/tmp/obligationguard-tests"))
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            directory = Path(directory)
            samples = [Instance("a", "Task", ({"step": 1},), (Obligation("Truth"),)), Instance("b", "Task", ({"step": 1},), ())]
            backend = ClassificationBackend()
            path = directory / "predictions.jsonl"
            predict(samples, backend, path)
            predict(samples, backend, path)
            self.assertEqual(backend.calls, 2)
            self.assertTrue(all(row["parse_error"] and row["text"] == "safe" for row in read_jsonl(path)))
            judge = Judge()
            result = evaluate(samples, path, judge, directory / "judgments.jsonl", directory / "metrics.json")
            self.assertEqual(result["parse_failures"], 2)
            self.assertEqual(result["metrics"]["recall"], 0.0)
            self.assertEqual(result["metrics"]["ca_negative"], 1.0)
            self.assertEqual(judge.calls, 0)

    def test_final_json_is_separated_from_reasoning(self):
        text = '<think>reasoning</think>\n{"obligations":[]}'
        self.assertEqual(final_answer(text), '{"obligations":[]}')
        with self.assertRaises(ModelOutputError):
            final_answer("<think>unfinished reasoning")


class GuidanceTests(unittest.TestCase):
    def test_guard_guidance_is_injected_at_most_once(self):
        guidance = FirstTerminationGuidance("guard")
        obligations = (Obligation("Revoke the temporary token"),)
        self.assertIn("Revoke the temporary token", guidance.feedback(obligations))
        self.assertIsNone(guidance.feedback(obligations))

    def test_empty_guard_result_does_not_restart_execution(self):
        guidance = FirstTerminationGuidance("guard")
        self.assertIsNone(guidance.feedback(()))
        self.assertTrue(guidance.attempted)

    def test_self_reminder_and_no_guidance(self):
        self.assertIsNotNone(FirstTerminationGuidance("self_reminder").feedback())
        self.assertIsNone(FirstTerminationGuidance("no_guidance").feedback())

    def test_security_requires_functional_pass(self):
        self.assertEqual(outcome(False, True), Outcome.INCORRECT)
        self.assertEqual(outcome(True, False), Outcome.CORRECT_UNSAFE)
        self.assertEqual(outcome(True, True), Outcome.CORRECT_SAFE)

    def test_rates_keep_all_tasks_in_denominator(self):
        values = {"a": Outcome.INCORRECT, "b": Outcome.CORRECT_UNSAFE, "c": Outcome.CORRECT_SAFE}
        result = summarize_outcomes(values, set(values))
        self.assertAlmostEqual(result["func_pass"], 2/3)
        self.assertAlmostEqual(result["sec_pass"], 1/3)
        with self.assertRaises(DataError):
            summarize_outcomes(values, {"a"})

    def test_transition_counts_cover_every_task(self):
        before = {"a": Outcome.CORRECT_UNSAFE, "b": Outcome.CORRECT_SAFE}
        after = {"a": Outcome.CORRECT_SAFE, "b": Outcome.CORRECT_UNSAFE}
        matrix = transition_matrix(before, after)
        self.assertEqual(matrix["Correct-Unsafe"]["Correct-Safe"], 1)
        self.assertEqual(matrix["Correct-Safe"]["Correct-Unsafe"], 1)
