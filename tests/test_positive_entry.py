import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from obligationguard.benchmark import import_benchmark, positive_instances
from obligationguard.analysis import action_observation_steps
from obligationguard.cli import main
from obligationguard.errors import ConfigurationError
from obligationguard.experiments import run_positive, run_rq1
from obligationguard.io import file_sha256, write_json, write_jsonl
from obligationguard.schema import Instance, Obligation


def benchmark():
    examples = []
    sizes = [size for size, count in ((2, 53), (3, 36), (4, 30), (5, 1)) for _ in range(count)]
    for index, size in enumerate(sizes):
        examples.append(Instance(f"positive-{index}", "Fixture task", ({"role": "user", "content": "Fixture task"},), tuple(Obligation(f"Required action {j}") for j in range(size))))
        examples.append(Instance(f"negative-{index}", "Fixture task", ({"role": "user", "content": "Fixture task"},), ()))
    return examples


class PositiveEntryTests(unittest.TestCase):
    def test_rq2_counts_actions_and_preserves_all_message_references(self):
        messages = [
            {"role": "user", "content": "Fixture task"},
            {"role": "assistant", "content": "Inspect", "tool_calls": [{"id": "a", "function": {"name": "execute", "arguments": "inspect"}}, {"id": "b", "function": {"name": "execute", "arguments": "verify"}}]},
            {"role": "tool", "tool_call_id": "a", "content": "Visible state"},
            {"role": "tool", "tool_call_id": "b", "content": "Verified"},
            {"role": "assistant", "content": "Submitted"},
        ]
        original = Instance("fixture", "Fixture task", tuple(messages), (Obligation("Close access"),))
        normalized = action_observation_steps(original)
        self.assertEqual(len(normalized.trajectory), 2)
        self.assertEqual(normalized.trajectory[0]["observation"][0]["message_index"], 2)
        self.assertEqual(normalized.trajectory[1]["observation"][0]["message_index"], 3)
        self.assertEqual(normalized.trajectory[1]["context"][0]["message_index"], 4)
        self.assertEqual(list(original.trajectory), messages)

    def test_benchmark_import_preserves_labels_and_message_indices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "source", root / "output"
            for index, item in enumerate(benchmark()):
                folder = source / ("positive" if item.obligations else "negative") / item.instance_id
                write_json(folder / "guard_input.json", {"messages": [{"role": "user", "content": item.task}, {"role": "assistant", "content": "Inspect `/ho" + "me/PrivateUser/file.txt`"}]})
                truth = item.target()
                truth["task_id"] = item.instance_id
                for obligation in truth["obligations"]:
                    obligation["evidence"] = [{"message_index": 1, "quote": "`/ho" + "me/PrivateUser/file.txt`"}]
                write_json(folder / "ground_truth.json", truth)
                write_json(folder / "provenance.json", {"source_benchmark": ["FeatureBench", "SWE-Bench Pro", "Terminal-Bench 2.0", "Terminal-Bench", "TerminalWorld"][index % 5], "source_task_id": item.instance_id})
            reference = next(source.rglob("guard_input.json"))
            original_digest = file_sha256(reference)
            report = import_benchmark(source, output)
            self.assertEqual(report["obligations"], 339)
            self.assertEqual(file_sha256(reference), original_digest)
            imported = positive_instances({"paths": {"benchmark": str(output / "obligationbench.jsonl")}})
            self.assertEqual(len(imported), 120)
            self.assertEqual(imported[0].obligations[0].evidence[0], {"message_index": 1, "quote": "`/file.txt`"})
            self.assertEqual(len(imported[0].trajectory), 2)
            self.assertEqual(imported[0].trajectory[1]["content"], "Inspect `/file.txt`")

    def test_only_positive_instances_reach_model_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_jsonl(root / "benchmark.jsonl", (x.to_dict() for x in benchmark()))
            runtime = {"paths": {"benchmark": str(root / "benchmark.jsonl"), "results": str(root)}, "evaluation_models": {"fixture-model": {}}}
            seen = []
            def evaluate(instances, *args):
                seen.extend(instances)
                return {"metrics": {key: 0.0 for key in ("precision", "recall", "exact_match", "ca_positive")}}
            with patch("obligationguard.experiments.evaluate_model", side_effect=evaluate):
                run_rq1({}, runtime)
            self.assertEqual(len(seen), 120)
            self.assertTrue(all(item.obligations for item in seen))
            self.assertEqual(sum(len(x.obligations) for x in seen), 339)
            self.assertNotIn("ca_negative", (root / "positive" / "table.csv").read_text())

    def test_entry_returns_after_positive_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = {"models": {"judge": {"model": "gpt-5.6-sol", "parameters": {"temperature": 0}}}, "paths": {"results": temporary}}
            results = {"fixture-model": {"metrics": {"recall": 0.5}}}
            with patch("obligationguard.experiments.readiness", return_value={"benchmark": True}) as check, patch("obligationguard.experiments.run_rq1", return_value=results), patch("obligationguard.experiments.launch_training") as train, patch("obligationguard.experiments.run_rq2") as rq2, patch("obligationguard.experiments.run_rq3") as rq3:
                self.assertEqual(run_positive({}, runtime), results)
            check.assert_called_once_with({}, runtime, scope="positive")
            train.assert_not_called()
            rq2.assert_not_called()
            rq3.assert_not_called()
            report = json.loads((Path(temporary) / "positive" / "summary.json").read_text())
            self.assertEqual(report["instances_per_model"], 120)

    def test_missing_requirements_prevent_all_model_calls(self):
        runtime = {"models": {"judge": {"model": "gpt-5.6-sol", "parameters": {"temperature": 0}}}}
        with patch("obligationguard.experiments.readiness", return_value={"guard_checkpoint": False}), patch("obligationguard.experiments.run_rq1") as evaluate:
            with self.assertRaises(ConfigurationError):
                run_positive({}, runtime)
        evaluate.assert_not_called()

    def test_cli_uses_the_positive_entry(self):
        with patch("obligationguard.cli.load_config", return_value={}), patch("obligationguard.experiments.run_positive") as positive:
            self.assertEqual(main(["positive"]), 0)
        positive.assert_called_once_with({}, {})

    def test_both_one_click_scripts_use_the_positive_command(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("run_all.sh", "run_all.bat"):
            text = (root / "scripts" / name).read_text()
            self.assertIn("-m obligationguard positive ", text)
            self.assertNotIn("-m obligationguard pipeline ", text)
