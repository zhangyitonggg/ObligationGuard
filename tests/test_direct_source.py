import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from obligationguard.direct_data import materialize_without_templates
from obligationguard.errors import DataError
from obligationguard.experiments import direct_training_set
from obligationguard.io import file_sha256, read_jsonl, write_jsonl
from obligationguard.schema import Instance


class DirectSourceTests(unittest.TestCase):
    def test_split_preserves_every_supplied_label_and_resumes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [{"user_task": f"Fixture task {i}", "trajectory": [{"step": 1, "action": "Inspect", "observation": "Visible state"}], "ground_truth_obligation_set": [f"Required action {i}"]} for i in range(6)]
            source = root / "source.jsonl"
            write_jsonl(source, rows)
            digest = file_sha256(source)
            path = materialize_without_templates(source, root / "split", 123, 4, 2)
            samples = list(read_jsonl(path)) + list(read_jsonl(path.parent / "validation.jsonl"))
            self.assertEqual({x["obligations"][0]["required_safety_action"] for x in samples}, {x["ground_truth_obligation_set"][0] for x in rows})
            self.assertEqual(file_sha256(source), digest)
            self.assertEqual(materialize_without_templates(source, root / "split", 123, 4, 2), path)
            with self.assertRaises(DataError):
                materialize_without_templates(source, root / "split", 124, 4, 2)

    def test_supplied_source_prevents_generator_calls(self):
        runtime = {"paths": {"without_templates": "fixture-source.jsonl"}}
        with patch("obligationguard.direct_data.materialize_without_templates", return_value=Path("fixture-train.jsonl")) as prepare, patch("obligationguard.experiments.run_jobs") as generate:
            self.assertEqual(direct_training_set(runtime, Path("fixture-output"), 123), Path("fixture-train.jsonl"))
        prepare.assert_called_once_with("fixture-source.jsonl", Path("fixture-output/without_templates"), 123)
        generate.assert_not_called()

    def test_prepared_ablation_rejects_training_validation_overlap(self):
        runtime = {"paths": {"direct_train": "fixture-train.jsonl", "direct_validation": "fixture-validation.jsonl", "without_templates": "fixture-source.jsonl"}}
        train = [Instance(f"train-{i}", "Fixture task", ({"action": "Inspect"},), ()) for i in range(40000)]
        validation = [Instance(f"validation-{i}", "Fixture task", ({"action": "Inspect"},), ()) for i in range(2000)]
        with patch("obligationguard.experiments.load_instances", side_effect=[train, validation]), patch("obligationguard.experiments.run_jobs") as generate, patch("obligationguard.direct_data.materialize_without_templates") as prepare:
            self.assertEqual(direct_training_set(runtime, Path("fixture-output"), 123), Path("fixture-train.jsonl"))
        generate.assert_not_called()
        prepare.assert_not_called()
        validation[0] = train[0]
        with patch("obligationguard.experiments.load_instances", side_effect=[train, validation]):
            with self.assertRaisesRegex(DataError, "share instance identifiers"):
                direct_training_set(runtime, Path("fixture-output"), 123)
